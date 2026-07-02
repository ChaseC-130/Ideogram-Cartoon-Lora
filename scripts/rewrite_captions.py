#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an expert prompt engineer. You will rewrite image captions for a text-to-image generator. "
    "Your goal is to modify the caption so that:\n"
    "1. The core subject is represented as an animated, cartoonish living being, creature, or game asset.\n"
    "   - If the subject is a non-living object, rewrite it as a cartoonish animated object with expressive features.\n"
    "   - If the subject is already living, keep it as the subject but describe it as a cartoonish video game character.\n"
    "2. The overall style is clearly described as a cartoonish video game style illustration.\n"
    "3. Keep the original color schemes, details, and composition backgrounds described in the prompt.\n"
    "Return ONLY the rewritten caption text as plain text. Do not wrap in quotes, markdown, or JSON formatting."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite JSON caption sidecars into a cartoon game asset style.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Directory containing .json caption files.",
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--field", default="caption", help="Source caption field to rewrite.")
    parser.add_argument("--output-field", default="modified_caption")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite files that already have the output field.")
    parser.add_argument("--system-instruction", default=DEFAULT_SYSTEM_INSTRUCTION)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it with: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        print("Error: The 'google-genai' package is required.")
        print("Please run: pip install google-genai pillow")
        sys.exit(1)

    client = genai.Client()
    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    json_files = sorted(list(dataset_dir.glob("*.json")))
    json_files = [f for f in json_files if f.name != ".aitk_size.json"]

    print(f"Found {len(json_files)} JSON caption files to rewrite.")

    success_count = 0
    for idx, json_file in enumerate(json_files):
        print(f"[{idx+1}/{len(json_files)}] Processing {json_file.name}...", end="", flush=True)

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if args.output_field in data and not args.overwrite:
                print(f" Already has {args.output_field}. Skipping.")
                continue

            original_caption = data.get(args.field, "").strip()
            if not original_caption:
                print(" Empty caption. Skipping.")
                continue

            response = client.models.generate_content(
                model=args.model,
                contents=original_caption,
                config={
                    "system_instruction": args.system_instruction,
                    "temperature": 0.2,
                }
            )

            modified_caption = response.text.strip()
            # Clean up any trailing/leading quotes the model might have returned
            if modified_caption.startswith('"') and modified_caption.endswith('"'):
                modified_caption = modified_caption[1:-1].strip()
            if modified_caption.startswith("'") and modified_caption.endswith("'"):
                modified_caption = modified_caption[1:-1].strip()

            data[args.output_field] = modified_caption

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(" Done.")
            success_count += 1

        except Exception as e:
            print(f" Failed!\nError: {e}")

    print(f"\nCompleted! Rewrote {success_count} caption files.")

if __name__ == "__main__":
    main()
