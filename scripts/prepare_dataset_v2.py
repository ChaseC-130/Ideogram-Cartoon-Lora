import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create simple Ideogram JSON caption sidecars for images in a dataset folder."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_v2",
        help="Dataset folder containing images and optional matching .txt captions.",
    )
    parser.add_argument(
        "--placeholder-template",
        default=(
            "A cartoon game asset illustration of {name}, clean outlines, "
            "flat vibrant colors, digital 2D style, no text."
        ),
        help="Template used when no matching .txt caption exists. Supports {name}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified directory: {dataset_dir}")

    # Supported image extensions
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    # Scan for images
    images = [p for p in dataset_dir.iterdir() if p.suffix.lower() in image_extensions]

    if not images:
        print("\nNo images found in dataset_v2 yet.")
        print("Please copy your new cartoon/game style images (.png, .jpg, etc.) into:")
        print(f"  {dataset_dir}")
        print("Then run this script again to generate the matching .json caption files.")
        return

    print(f"\nFound {len(images)} images. Processing...")

    json_created = 0
    json_skipped = 0

    for img_path in images:
        json_path = img_path.with_suffix(".json")
        txt_path = img_path.with_suffix(".txt")

        # If json already exists, skip
        if json_path.exists():
            json_skipped += 1
            continue

        caption_text = ""
        # 1. Check if there is a matching .txt file with the description
        if txt_path.exists():
            try:
                caption_text = txt_path.read_text(encoding="utf-8").strip()
                print(f"  Loaded caption from {txt_path.name}")
            except Exception as e:
                print(f"  Error reading {txt_path.name}: {e}")

        # 2. If no .txt file, write a placeholder
        if not caption_text:
            entity_name = img_path.stem.replace("_", " ").title()
            caption_text = args.placeholder_template.format(name=entity_name)
            print(f"  Generated placeholder caption for {img_path.name}")

        # Write json file
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"caption": caption_text}, f, indent=2)
            json_created += 1
        except Exception as e:
            print(f"  Error writing {json_path.name}: {e}")

    print(f"\nDone! Created {json_created} new .json file(s), skipped {json_skipped} existing.")

if __name__ == "__main__":
    main()
