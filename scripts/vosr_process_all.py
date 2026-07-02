#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_EXCLUDED_DIR_PREFIXES = (
    "vosr",
    "unblur_preview",
    "test_run",
    "__pycache__",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch VOSR workflow: discover project images, stage pending files, "
            "run VOSR once, and copy outputs back into a mirrored directory tree."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Image source root. May be repeated. Defaults to output/.",
    )
    parser.add_argument(
        "--output",
        default="output/vosr_all",
        help="Destination root for processed images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess images even when the destination file already exists.",
    )
    parser.add_argument(
        "--include-compare",
        action="store_true",
        help="Include files whose names contain 'compare'.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Process at most this many pending images; useful for test batches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned batch summary.",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep staged symlinks and flat VOSR outputs after copying results.",
    )
    parser.add_argument("--upscale", default="1")
    parser.add_argument("--tile-size", default="512")
    parser.add_argument("--tile-overlap", default="32")
    parser.add_argument("--align-method", default="wavelet")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--device", default="")
    parser.add_argument("--checkpoint", default="")
    return parser.parse_args()


def absolute_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def is_excluded_path(path: Path, output_root: Path, include_compare: bool) -> bool:
    resolved = path.resolve()
    if resolved == output_root or output_root in resolved.parents:
        return True
    if any(part.startswith(DEFAULT_EXCLUDED_DIR_PREFIXES) for part in path.parts):
        return True
    if not include_compare and "compare" in path.name.lower():
        return True
    return False


def discover_images(
    source_roots: list[Path],
    output_root: Path,
    include_compare: bool,
    force: bool,
    max_images: int,
) -> list[dict[str, Path]]:
    jobs: list[dict[str, Path]] = []
    multi_source = len(source_roots) > 1

    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        if not source_root.is_dir():
            raise NotADirectoryError(f"Source root must be a directory: {source_root}")

        for src in sorted(source_root.rglob("*")):
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if is_excluded_path(src, output_root, include_compare):
                continue

            rel = src.relative_to(source_root)
            if multi_source:
                rel = Path(source_root.name) / rel
            dest = output_root / rel

            if not force and dest.exists():
                continue

            jobs.append({"src": src.resolve(), "dest": dest})
            if max_images and len(jobs) >= max_images:
                return jobs

    return jobs


def stage_jobs(jobs: list[dict[str, Path]], work_dir: Path) -> Path:
    stage_input = work_dir / "input"
    stage_input.mkdir(parents=True, exist_ok=True)

    for idx, job in enumerate(jobs, start=1):
        stage_name = f"{idx:06d}{job['src'].suffix.lower()}"
        job["stage_name"] = Path(stage_name)
        stage_path = stage_input / stage_name
        if stage_path.exists() or stage_path.is_symlink():
            stage_path.unlink()
        os.symlink(job["src"], stage_path)

    return stage_input


def write_manifest(jobs: list[dict[str, Path]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        for job in jobs:
            row = {
                "stage_name": str(job.get("stage_name", "")),
                "source": str(job["src"]),
                "destination": str(job["dest"]),
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def run_vosr(args: argparse.Namespace, stage_input: Path, raw_output: Path) -> None:
    env = os.environ.copy()
    env["VOSR_UPSCALE"] = str(args.upscale)
    env["VOSR_TILE_SIZE"] = str(args.tile_size)
    env["VOSR_TILE_OVERLAP"] = str(args.tile_overlap)
    env["VOSR_ALIGN_METHOD"] = str(args.align_method)
    env["VOSR_SEED"] = str(args.seed)
    if args.device:
        env["VOSR_DEVICE"] = args.device
    if args.checkpoint:
        env["VOSR_CHECKPOINT"] = str(absolute_project_path(args.checkpoint))

    command = [
        str(PROJECT_ROOT / "scripts" / "run_vosr.sh"),
        str(stage_input),
        str(raw_output),
        "--force_rerun",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def copy_results(jobs: list[dict[str, Path]], raw_output: Path) -> list[dict[str, Path]]:
    missing: list[dict[str, Path]] = []
    for job in jobs:
        stage_name = str(job["stage_name"])
        matches = list(raw_output.rglob(stage_name))
        if not matches:
            missing.append(job)
            continue

        dest = job["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[0], dest)

    return missing


def main() -> int:
    args = parse_args()
    source_roots = [absolute_project_path(v) for v in (args.source or ["output"])]
    output_root = absolute_project_path(args.output)

    jobs = discover_images(
        source_roots=source_roots,
        output_root=output_root,
        include_compare=args.include_compare,
        force=args.force,
        max_images=args.max_images,
    )

    print(f"Sources: {', '.join(str(p) for p in source_roots)}")
    print(f"Output:  {output_root}")
    print(f"Pending images: {len(jobs)}")

    for job in jobs[:8]:
        print(f"  {job['src'].relative_to(PROJECT_ROOT)} -> {job['dest'].relative_to(PROJECT_ROOT)}")
    if len(jobs) > 8:
        print(f"  ... {len(jobs) - 8} more")

    if args.dry_run or not jobs:
        return 0

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = output_root / ".vosr_work" / batch_id
    raw_output = work_dir / "raw"
    manifest_path = output_root / "manifests" / f"{batch_id}.jsonl"
    summary_path = output_root / "manifests" / f"{batch_id}.summary.json"

    stage_input = stage_jobs(jobs, work_dir)
    write_manifest(jobs, manifest_path)
    run_vosr(args, stage_input, raw_output)
    missing = copy_results(jobs, raw_output)

    summary = {
        "batch_id": batch_id,
        "source_roots": [str(p) for p in source_roots],
        "output_root": str(output_root),
        "requested": len(jobs),
        "completed": len(jobs) - len(missing),
        "missing": [
            {
                "stage_name": str(job["stage_name"]),
                "source": str(job["src"]),
                "destination": str(job["dest"]),
            }
            for job in missing
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print(f"Completed: {summary['completed']} / {summary['requested']}")

    if missing:
        print("Missing outputs:")
        for job in missing[:20]:
            print(f"  {job['src']}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
