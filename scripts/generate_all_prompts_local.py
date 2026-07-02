#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_SYSTEM_PROMPT = (
    "Describe this source image in a highly creative, cartoonish video game asset style. "
    "If the source is mechanical, a weapon, or an inanimate object, describe it with "
    "expressive animated features. Describe the subject's silhouette, environment, "
    "colors, lighting, and styling in a single paragraph under 80 words. Avoid text, "
    "letters, logos, signatures, or UI labels unless they are essential."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image prompt cache using a local Qwen3-VL model.")
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
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu.")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts generated.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def import_torch():
    try:
        import torch
    except ImportError as e:
        raise SystemExit("The local prompt generator requires PyTorch. Run: pip install torch") from e
    return torch


def resolve_device(value: str, torch_module) -> str:
    if value != "auto":
        return value
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


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
    if not args.source_dir.exists():
        raise SystemExit(f"Source directory not found: {args.source_dir}")

    torch = import_torch()
    device = resolve_device(args.device, torch)
    dtype = torch.bfloat16 if device in {"cuda", "mps"} else torch.float32
    prompts_cache = load_cache(args.cache_path)
    image_paths = iter_images(args.source_dir)
    images_to_process = [
        (path.relative_to(args.source_dir).as_posix(), path)
        for path in image_paths
        if path.relative_to(args.source_dir).as_posix() not in prompts_cache
    ]

    print(f"Found {len(image_paths)} source images in {args.source_dir}.")
    print(f"{len(images_to_process)} images need prompts generated.")

    if args.limit is not None:
        images_to_process = images_to_process[: args.limit]
        print(f"Limiting generation to {len(images_to_process)} prompts.")

    if not images_to_process:
        print("All prompts are already cached. Nothing to do.")
        return

    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as e:
        raise SystemExit(
            "The local prompt generator requires a Transformers version with "
            "Qwen3VLForConditionalGeneration."
        ) from e

    print(f"Loading local model {args.model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()

    count = 0
    for rel_key, path in images_to_process:
        print(f"[{count + 1}/{len(images_to_process)}] Describing {rel_key}...")
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((512, 512))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": args.system_prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=img, padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            prompts_cache[rel_key] = output_text.replace("\n", " ")
            count += 1
            save_cache(args.cache_path, prompts_cache)
        except Exception as e:
            save_cache(args.cache_path, prompts_cache)
            raise SystemExit(f"Error describing {rel_key}: {e}") from e

    print(f"Successfully generated {count} prompts locally.")
    print(f"Cache: {args.cache_path}")


if __name__ == "__main__":
    sys.exit(main())
