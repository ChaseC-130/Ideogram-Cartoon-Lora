# Image Generation Runbook

This is the generic end-to-end workflow for turning a source image into a final
cartoon-style generated asset with an Ideogram 4 LoRA.

The canonical order is:

1. Start from a source image.
2. Optionally run the source through blur preprocessing.
3. Run Ideogram img2img with a trained LoRA.
4. Refine/sharpen and optionally restore small dark details from the source.
5. Optionally run VOSR to unblur/upscale the refined result.
6. Inspect the final image before using it downstream.

## Prerequisites

Run commands from the project root:

```bash
cd /path/to/Ideogram-Cartoon-Lora
```

Expected local dependencies:

- AI Toolkit checkout, supplied with `--ai-toolkit-path` or `AI_TOOLKIT_PATH`
- Trained LoRA weights, usually outside git under `weights/`
- Optional VOSR checkout at `external/VOSR`
- Optional VOSR checkpoint under `external/VOSR/preset/ckpts/`

Quick checks:

```bash
test -d "${AI_TOOLKIT_PATH:-../ai-toolkit}"
test -f weights/cartoon_lora.safetensors
```

## One Source Image

Set the inputs:

```bash
PROJECT="$PWD"
AI_TOOLKIT="${AI_TOOLKIT_PATH:-../ai-toolkit}"
LORA="$PROJECT/weights/cartoon_lora.safetensors"
SOURCE=/absolute/path/to/source.png
SLUG=my_asset_name
PROMPT='cartoon game asset of the subject, expressive silhouette, clean cel-shaded edges, vibrant colors, dramatic readable lighting, no text, no letters'
WORK="$PROJECT/output/e2e/$SLUG"
```

Create a mirrored working layout:

```bash
mkdir -p "$WORK/original" "$WORK/blurred" "$WORK/generated" "$WORK/refined" "$WORK/vosr"
cp "$SOURCE" "$WORK/original/$SLUG.png"
```

Optional blur preprocessing:

```bash
python3 - "$WORK/original/$SLUG.png" "$WORK/blurred/$SLUG.png" <<'PY'
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

source = Path(sys.argv[1])
dest = Path(sys.argv[2])

img = Image.open(source).convert("RGB")
w, h = img.size
w = max(16, (w // 16) * 16)
h = max(16, (h // 16) * 16)
img = img.resize((w, h), Image.Resampling.LANCZOS)

bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
bgr = cv2.medianBlur(bgr, 9)
bgr = cv2.bilateralFilter(bgr, 9, 80, 80)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

dest.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(rgb).save(dest)
print(dest)
PY
```

Run Ideogram img2img against the blurred image. If you skip preprocessing,
change `--image_path` to the original source.

```bash
python3 scripts/run_image_to_image.py \
  --ai-toolkit-path "$AI_TOOLKIT" \
  --image_path "$WORK/blurred/$SLUG.png" \
  --prompt "$PROMPT" \
  --lora_path "$LORA" \
  --strength 0.70 \
  --steps 28 \
  --guidance 3.5 \
  --device mps \
  --seed 42 \
  --output_path "$WORK/generated/$SLUG.png"
```

Refine the generated image:

```bash
python3 scripts/refine_img2img_assets.py \
  --generated-dir "$WORK/generated" \
  --original-dir "$WORK/original" \
  --output-dir "$WORK/refined" \
  --overwrite
```

Optional VOSR:

```bash
python3 scripts/vosr_process_all.py \
  --source "$WORK/refined" \
  --output "$WORK/vosr" \
  --force \
  --upscale 1 \
  --tile-size 512 \
  --tile-overlap 32 \
  --align-method wavelet \
  --seed 42
```

Final output:

```bash
open "$WORK/vosr/$SLUG.png"
```

If VOSR is skipped, use:

```bash
open "$WORK/refined/$SLUG.png"
```

## Batch Workflow

Create a prompt cache whose keys are paths relative to your source directory:

```json
{
  "asset_a.png": "cartoon game asset of asset A, no text",
  "folder/asset_b.png": "cartoon game asset of asset B, no text"
}
```

You can create this manually or generate it:

```bash
python3 scripts/generate_all_prompts.py \
  --source-dir input/images \
  --cache-path output/img2img_prompts.json
```

Run the batch img2img job:

```bash
python3 scripts/batch_img2img_all.py \
  --ai-toolkit-path "$AI_TOOLKIT" \
  --source-dir input/images \
  --prompt-cache output/img2img_prompts.json \
  --lora-path "$LORA" \
  --output-dir output/img2img_assets \
  --limit 5
```

Run refinement:

```bash
python3 scripts/refine_img2img_assets.py \
  --generated-dir output/img2img_assets \
  --original-dir input/images \
  --output-dir output/img2img_assets_refined \
  --overwrite
```

Optional VOSR:

```bash
python3 scripts/vosr_process_all.py \
  --source output/img2img_assets_refined \
  --output output/vosr_img2img_assets_refined \
  --force
```

## Optional LoRA Training Refresh

Only do this when the LoRA itself needs to be rebuilt.

Make sure training images and JSON captions exist in:

```text
dataset
```

Generate missing captions:

```bash
python3 scripts/label_images.py --mode auto
```

Create a local AI Toolkit config from an example:

```bash
cp config/train_lora.example.yaml config/train_lora.yaml
```

Edit local paths, then train from AI Toolkit:

```bash
cd "$AI_TOOLKIT"
python run.py /path/to/Ideogram-Cartoon-Lora/config/train_lora.yaml
```

## Tuning Defaults

Start with these values:

- `strength=0.70`
- `steps=28`
- `guidance=3.5`
- `seed=42`
- VOSR `upscale=1`
- VOSR `tile-size=512`
- VOSR `tile-overlap=32`
- VOSR `align-method=wavelet`

If the result keeps too much of the original source, increase `strength`
slightly. If it drifts too far from the source silhouette, lower `strength`.

## Quality Checklist

- The final image came from the refined or VOSR output, not the raw generated output.
- The result is not over-smoothed after VOSR.
- Small important dark details survived the refine pass.
- No unwanted text, letters, signatures, or UI artifacts were introduced.
- Transparent sources still have correct alpha after refinement.
- The image reads clearly at the target display size.

## Troubleshooting

If img2img fails to import toolkit modules, pass `--ai-toolkit-path` or set
`AI_TOOLKIT_PATH`.

If VOSR says its Python environment is missing, set up `external/VOSR/.venv`
first.

If VOSR output is missing, rerun with:

```bash
python3 scripts/vosr_process_all.py --source "$WORK/refined" --output "$WORK/vosr" --force --keep-work
```

Then inspect:

```text
$WORK/vosr/manifests
$WORK/vosr/.vosr_work
```

If the final image is too blurry, confirm that the refine pass ran before VOSR.
The intended order is raw generated image, then refine, then VOSR.
