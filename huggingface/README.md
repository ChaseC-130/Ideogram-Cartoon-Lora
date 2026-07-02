---
license: other
base_model: ideogram-ai/ideogram-4-fp8
pipeline_tag: image-to-image
tags:
  - ideogram4
  - lora
  - adapter
  - image-to-image
  - cartoon
  - game-art
  - safetensors
---

# Ideogram Cartoon LoRA

This is a LoRA adapter trained for a clean, readable cartoon game-asset style on
top of `ideogram-ai/ideogram-4-fp8`.

Repository and scripts:
[github.com/ChaseC-130/Ideogram-Cartoon-Lora](https://github.com/ChaseC-130/Ideogram-Cartoon-Lora)

## Sample Outputs

| With LoRA: clockwork beetle | With LoRA: crystal lantern | With LoRA: storm dragon |
| --- | --- | --- |
| <img src="samples/generated/clockwork_beetle.png" alt="Clockwork beetle" width="256"> | <img src="samples/generated/crystal_lantern.png" alt="Crystal lantern" width="256"> | <img src="samples/generated/storm_dragon.png" alt="Storm dragon" width="256"> |

Base model comparison, generated with the same source sketches and prompt cache
but with `--no-lora`:

| No LoRA: clockwork beetle | No LoRA: crystal lantern | No LoRA: storm dragon |
| --- | --- | --- |
| <img src="samples/generated_base/clockwork_beetle.png" alt="Clockwork beetle generated without LoRA" width="256"> | <img src="samples/generated_base/crystal_lantern.png" alt="Crystal lantern generated without LoRA" width="256"> | <img src="samples/generated_base/storm_dragon.png" alt="Storm dragon generated without LoRA" width="256"> |

## Sample Prompts

```text
custom_subject {"caption": "A whimsical clockwork beetle hero made from polished brass plates and tiny blue glass lenses, standing on mossy stone in a cozy workshop, chunky readable silhouette, clean cel-shaded edges, vibrant cartoon game asset style, no text or letters."}
custom_subject {"caption": "A cheerful enchanted crystal lantern creature with small metal feet, warm golden light glowing through teal glass, posed on a forest path at dusk, crisp 2D fantasy game illustration, expressive face, no text or logos."}
custom_subject {"caption": "A tiny full-body storm dragon mascot centered with plenty of margin, rounded dark scales, bright yellow lightning horns, playful eyes, and fluffy cloud puffs around its feet, dramatic readable lighting, colorful cartoon game card art, no words or symbols."}
```

## Quick Usage

Clone the workflow repo and install dependencies:

```bash
git clone https://github.com/ChaseC-130/Ideogram-Cartoon-Lora.git
cd Ideogram-Cartoon-Lora
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download this LoRA adapter from Hugging Face, then run img2img with a local AI
Toolkit checkout:

```bash
python scripts/run_image_to_image.py \
  --ai-toolkit-path ../ai-toolkit \
  --image_path samples/source_images/crystal_lantern.png \
  --prompt 'A cheerful enchanted crystal lantern creature with small metal feet, warm golden light glowing through teal glass, posed on a forest path at dusk, crisp 2D fantasy game illustration, expressive face, no text or logos.' \
  --lora_path weights/ideogram_cartoon_lora.safetensors \
  --output_path output/crystal_lantern_sample.png \
  --strength 0.88 \
  --steps 20 \
  --guidance 3.6 \
  --device mps \
  --seed 21
```

## Files

- `ideogram_cartoon_lora.safetensors`: LoRA adapter weights.
- `samples/prompts.txt`: sample JSON-style prompts.
- `samples/prompt_cache.json`: prompt cache for the batch img2img script.
- `samples/source_images/`: synthetic source sketches used for examples.
- `samples/generated/`: generated sample outputs.
- `samples/generated_base/`: base-model outputs generated without the LoRA.

## License Notes

This adapter is a derivative of `ideogram-ai/ideogram-4-fp8`. Use and
redistribution of this adapter must remain compatible with the base model
license and your use case. The public Ideogram 4 weights are distributed under
Ideogram's model terms, which are separate from the MIT-licensed workflow code
in the GitHub repository.

This model card is not legal advice.
