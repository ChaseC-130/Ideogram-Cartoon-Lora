#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_DIR = PROJECT_ROOT / "output" / "img2img_assets"
DEFAULT_ORIGINAL_DIR = PROJECT_ROOT / "input" / "images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "img2img_assets_refined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Non-destructively refine regenerated image assets with mild "
            "sharpening and optional tiny dark-detail restoration from the originals."
        )
    )
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-detail-restore", action="store_true")

    parser.add_argument("--contrast", type=float, default=1.04)
    parser.add_argument("--sharpness", type=float, default=1.10)
    parser.add_argument("--unsharp-radius", type=float, default=1.15)
    parser.add_argument("--unsharp-percent", type=int, default=175)
    parser.add_argument("--unsharp-threshold", type=int, default=3)

    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=58.0,
        help="Original-image luma threshold for eye/line fragments to restore.",
    )
    parser.add_argument(
        "--min-luma-gap",
        type=float,
        default=65.0,
        help="Only restore original dark pixels if the generated image is this much lighter.",
    )
    parser.add_argument("--min-detail-area", type=int, default=3)
    parser.add_argument("--max-detail-area", type=int, default=180)
    parser.add_argument("--max-detail-span", type=int, default=34)
    parser.add_argument("--detail-opacity", type=float, default=0.68)
    parser.add_argument("--detail-dilate", type=int, default=0)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_DIR/_refine_report.json.",
    )
    return parser.parse_args()


def iter_pngs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.png") if p.is_file() and not p.name.startswith("."))


def luma(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    return rgb_f[..., 0] * 0.299 + rgb_f[..., 1] * 0.587 + rgb_f[..., 2] * 0.114


def apply_refine_sharpen(img: Image.Image, args: argparse.Namespace) -> Image.Image:
    refined = img.convert("RGB")
    refined = ImageEnhance.Contrast(refined).enhance(args.contrast)
    refined = refined.filter(
        ImageFilter.UnsharpMask(
            radius=args.unsharp_radius,
            percent=args.unsharp_percent,
            threshold=args.unsharp_threshold,
        )
    )
    refined = ImageEnhance.Sharpness(refined).enhance(args.sharpness)
    return refined


def collect_small_components(
    candidates: np.ndarray,
    min_area: int,
    max_area: int,
    max_span: int,
) -> tuple[np.ndarray, int]:
    height, width = candidates.shape
    visited = np.zeros_like(candidates, dtype=bool)
    selected = np.zeros_like(candidates, dtype=bool)
    component_count = 0

    ys, xs = np.nonzero(candidates)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x] or not candidates[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        comp_y: list[int] = []
        comp_x: list[int] = []

        while stack:
            y, x = stack.pop()
            comp_y.append(y)
            comp_x.append(x)

            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if visited[ny, nx] or not candidates[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        area = len(comp_y)
        if area < min_area or area > max_area:
            continue

        span_y = max(comp_y) - min(comp_y) + 1
        span_x = max(comp_x) - min(comp_x) + 1
        if max(span_y, span_x) > max_span:
            continue

        selected[comp_y, comp_x] = True
        component_count += 1

    return selected, component_count


def maybe_dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0 or not mask.any():
        return mask
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(iterations):
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    return np.array(mask_img) > 0


def restore_tiny_dark_details(
    refined: Image.Image,
    original: Image.Image,
    args: argparse.Namespace,
) -> tuple[Image.Image, dict[str, int]]:
    refined_rgb = refined.convert("RGB")
    original_rgba = original.convert("RGBA")
    if original_rgba.size != refined_rgb.size:
        original_rgba = original_rgba.resize(refined_rgb.size, Image.Resampling.LANCZOS)

    orig_arr = np.array(original_rgba)
    refined_arr = np.array(refined_rgb)
    orig_rgb = orig_arr[..., :3]
    alpha_live = orig_arr[..., 3] > 0

    orig_luma = luma(orig_rgb)
    refined_luma = luma(refined_arr)

    candidates = (
        alpha_live
        & (orig_luma <= args.dark_threshold)
        & ((refined_luma - orig_luma) >= args.min_luma_gap)
    )
    selected, component_count = collect_small_components(
        candidates,
        args.min_detail_area,
        args.max_detail_area,
        args.max_detail_span,
    )
    selected = maybe_dilate_mask(selected, args.detail_dilate)

    if not selected.any():
        return refined_rgb, {"detail_components": 0, "detail_pixels": 0}

    opacity = max(0.0, min(1.0, args.detail_opacity))
    selected_f = selected.astype(np.float32)[..., None] * opacity
    blended = (
        refined_arr.astype(np.float32) * (1.0 - selected_f)
        + orig_rgb.astype(np.float32) * selected_f
    )
    blended = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB"), {
        "detail_components": component_count,
        "detail_pixels": int(selected.sum()),
    }


def preserve_alpha_if_needed(refined: Image.Image, original: Image.Image | None) -> Image.Image:
    if original is None or original.mode != "RGBA":
        return refined

    alpha = original.getchannel("A")
    alpha_arr = np.array(alpha)
    if alpha_arr.min() == 255:
        return refined

    if alpha.size != refined.size:
        alpha = alpha.resize(refined.size, Image.Resampling.LANCZOS)
    out = refined.convert("RGBA")
    out.putalpha(alpha)
    return out


def refine_one(
    generated_path: Path,
    original_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    generated = Image.open(generated_path).convert("RGB")
    original = Image.open(original_path) if original_path.exists() else None

    refined = apply_refine_sharpen(generated, args)
    stats: dict[str, Any] = {
        "path": str(generated_path.relative_to(args.generated_dir)),
        "original_found": original is not None,
        "detail_components": 0,
        "detail_pixels": 0,
    }

    if original is not None and not args.no_detail_restore:
        refined, detail_stats = restore_tiny_dark_details(refined, original, args)
        stats.update(detail_stats)

    refined = preserve_alpha_if_needed(refined, original)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    refined.save(output_path, optimize=True)
    return stats


def main() -> None:
    args = parse_args()
    report_path = args.report_path or args.output_dir / "_refine_report.json"

    if not args.generated_dir.exists():
        raise SystemExit(f"Generated directory not found: {args.generated_dir}")
    if not args.original_dir.exists():
        print(
            f"Original directory not found: {args.original_dir}. "
            "Continuing with sharpening only."
        )

    generated_paths = iter_pngs(args.generated_dir)
    print(f"Found {len(generated_paths)} generated PNGs in {args.generated_dir}")

    processed = 0
    skipped = 0
    errors: list[dict[str, str]] = []
    report: list[dict[str, Any]] = []

    for generated_path in generated_paths:
        if args.limit is not None and processed >= args.limit:
            break

        rel = generated_path.relative_to(args.generated_dir)
        original_path = args.original_dir / rel
        output_path = args.output_dir / rel

        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            stats = refine_one(generated_path, original_path, output_path, args)
            report.append(stats)
            processed += 1
            if processed % 25 == 0:
                print(f"Processed {processed} images...")
        except Exception as exc:
            errors.append({"path": str(rel), "error": str(exc)})
            print(f"Error refining {rel}: {exc}")

    summary = {
        "generated_dir": str(args.generated_dir),
        "original_dir": str(args.original_dir),
        "output_dir": str(args.output_dir),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "items": report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    detail_components = sum(item["detail_components"] for item in report)
    detail_pixels = sum(item["detail_pixels"] for item in report)
    print(
        "Done. "
        f"processed={processed}, skipped={skipped}, errors={len(errors)}, "
        f"restored_components={detail_components}, restored_pixels={detail_pixels}"
    )
    print(f"Report: {report_path}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
