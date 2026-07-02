#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_SYSTEM_PROMPT = (
    "Write a highly creative, cartoonish video game asset description for this image. "
    "If the source is mechanical, a weapon, or an inanimate object, describe it with "
    "expressive animated features. Describe the subject, pose or silhouette, setting, "
    "colors, lighting, and style in a single paragraph under 80 words. Avoid text, "
    "letters, logos, signatures, or UI labels unless they are essential."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image prompt cache with Gemini for a directory of assets."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "input" / "images",
        help="Directory of source images. Cache keys are relative to this directory.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=PROJECT_ROOT / "output" / "img2img_prompts.json",
        help="JSON prompt cache to create/update.",
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of new prompts generated.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Instruction sent with each image.",
    )
    return parser.parse_args()


def iter_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
    )


def load_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache_path: Path, prompts_cache: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(prompts_cache, f, indent=2, ensure_ascii=False, sort_keys=True)


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    if not args.source_dir.exists():
        raise SystemExit(f"Source directory not found: {args.source_dir}")

    try:
        from google import genai
    except ImportError as e:
        raise SystemExit("The 'google-genai' package is required. Run: pip install google-genai") from e

    prompts_cache = load_cache(args.cache_path)
    image_paths = iter_images(args.source_dir)
    print(f"Found {len(image_paths)} source images in {args.source_dir}.")

    client = genai.Client()
    count = 0
    skipped = 0

    for index, path in enumerate(image_paths, start=1):
        rel_key = path.relative_to(args.source_dir).as_posix()
        if rel_key in prompts_cache:
            skipped += 1
            continue

        if args.limit is not None and count >= args.limit:
            print(f"Reached limit of {args.limit} new prompts. Stopping.")
            break

        print(f"[{index}/{len(image_paths)}] Generating prompt for {rel_key}...")
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((512, 512))

            response = client.models.generate_content(
                model=args.model,
                contents=[img, args.system_prompt],
            )

            prompts_cache[rel_key] = response.text.strip().replace("\n", " ")
            count += 1
            save_cache(args.cache_path, prompts_cache)
        except Exception as e:
            save_cache(args.cache_path, prompts_cache)
            raise SystemExit(f"Error processing {rel_key}: {e}") from e

    print(f"Done. Generated {count} new prompts, skipped {skipped} already cached.")
    print(f"Cache: {args.cache_path}")


if __name__ == "__main__":
    main()
