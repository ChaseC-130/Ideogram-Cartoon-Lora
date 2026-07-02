#!/usr/bin/env python3
import os
import sys
import json
import glob
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    parser = argparse.ArgumentParser(description="Label images in dataset/ directory using either Gemini API or a local Vision-Language Model.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Directory of images to label with matching .json sidecars.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "api", "local"],
        help="Labeling mode: 'api' (Gemini API), 'local' (local Qwen2.5-VL model), or 'auto' (detects GEMINI_API_KEY)."
    )
    parser.add_argument(
        "--local_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face repo ID of the local VLM to use (default: Qwen/Qwen2.5-VL-3B-Instruct)."
    )
    return parser.parse_args()

def extract_caption_from_text(text):
    text = text.strip()
    # Attempt to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if json_match:
        text = json_match.group(1)

    # Clean up outer characters if the model returned extra text
    try:
        data = json.loads(text)
        if "caption" in data:
            return data["caption"]
    except json.JSONDecodeError:
        # If it's not valid JSON, try to regex find the caption key
        caption_match = re.search(r'"caption"\s*:\s*"(.*?)"', text, re.DOTALL)
        if caption_match:
            return caption_match.group(1).replace('\\"', '"')

    # If all parsing fails, return the raw text as the caption
    return text

def label_with_api(image_paths, client, model_name, prompt):
    from PIL import Image
    from google.genai import types

    success_count = 0
    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        json_path = img_path.with_suffix(".json")

        if json_path.exists():
            print(f"Skipping {img_path.name} (already labeled).")
            continue

        print(f"Labeling {img_path.name} via Gemini API...", end="", flush=True)
        try:
            img = Image.open(img_path)
            response = client.models.generate_content(
                model=model_name,
                contents=[img, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            caption = extract_caption_from_text(response.text)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"caption": caption}, f, indent=2, ensure_ascii=False)
            print(" Done.")
            success_count += 1
        except Exception as e:
            print(f" Failed!\nError: {e}")

    return success_count

def label_with_local(image_paths, model_id, prompt):
    print("Loading PyTorch and Transformers (this may take a few seconds)...")
    try:
        import torch
        import torchvision
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from PIL import Image
    except ImportError:
        print("Error: PyTorch, Torchvision, and Transformers are required for local mode.")
        print("Please run: pip install torch torchvision transformers accelerate pillow")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
    print(f"Loading local VLM '{model_id}' onto {device} ({dtype})...")

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device
        )
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check that the model name is correct and you have an active internet connection to download it.")
        sys.exit(1)

    success_count = 0
    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        json_path = img_path.with_suffix(".json")

        if json_path.exists():
            print(f"Skipping {img_path.name} (already labeled).")
            continue

        print(f"Labeling {img_path.name} locally...", end="", flush=True)
        try:
            img = Image.open(img_path)

            # Construct standard chat template for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Format inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=img, padding=True, return_tensors="pt")
            inputs = inputs.to(device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                # Strip input tokens
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            caption = extract_caption_from_text(output_text)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"caption": caption}, f, indent=2, ensure_ascii=False)
            print(" Done.")
            success_count += 1
        except Exception as e:
            print(f" Failed!\nError: {e}")

    return success_count

def main():
    args = parse_args()

    dataset_dir = args.dataset_dir

    # Find all images
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.WEBP"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(dataset_dir / ext)))

    if not image_paths:
        print(f"No images found in {dataset_dir}")
        print("Please place your training images (.png, .jpg, etc.) in that folder first.")
        sys.exit(0)

    print(f"Found {len(image_paths)} images to label.")

    # Prompt for the model
    prompt = (
        "Analyze this image and write a detailed caption suitable for training a text-to-image generator (Ideogram 4). "
        "Describe the subject, style, composition, lighting, and any prominent text or colors. "
        "Return ONLY a JSON object with a single key 'caption'. Do not include markdown code block formatting. "
        "Example output:\n"
        '{"caption": "A high-resolution photo of a vintage red leather armchair placed in a cozy, sunlit reading corner."}'
    )

    # Determine mode
    mode = args.mode
    api_key = os.environ.get("GEMINI_API_KEY")

    if mode == "auto":
        if api_key:
            mode = "api"
        else:
            mode = "local"
            print("No GEMINI_API_KEY environment variable detected. Defaulting to local model mode.")

    if mode == "api":
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable is not set.")
            print("To use API mode, set it with: export GEMINI_API_KEY='your-key-here'")
            print("Or run in local mode with: python3 scripts/label_images.py --mode local")
            sys.exit(1)

        try:
            from google import genai
        except ImportError:
            print("Error: The 'google-genai' package is required for API mode.")
            print("Please run: pip install google-genai pillow")
            sys.exit(1)

        client = genai.Client()
        success_count = label_with_api(image_paths, client, "gemini-2.5-flash", prompt)
    else:
        success_count = label_with_local(image_paths, args.local_model, prompt)

    print(f"\nLabeling completed. Labeled {success_count} new images.")

if __name__ == "__main__":
    main()
