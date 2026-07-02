# Model Card: <LoRA Name>

## Summary

- Base model: `ideogram-ai/ideogram-4-fp8`
- Adapter type: LoRA
- Trigger word: `<trigger>`
- Intended use: non-commercial research/prototyping unless you have commercial
  rights for the base model and derivative weights.

## Training Data

Describe the dataset source, ownership/permission status, number of images,
captioning method, and any excluded private or unreleased assets.

## Training Details

- Training tool: AI Toolkit
- Resolution: 1024 x 1024
- Rank/alpha: 16/16
- Steps/epochs: `<fill in>`
- Optimizer/scheduler: `<fill in>`

## Usage

```bash
python scripts/run_image_to_image.py \
  --ai-toolkit-path ../ai-toolkit \
  --image_path input/source.png \
  --prompt "<trigger> cartoon game asset, clean readable silhouette, no text" \
  --lora_path weights/<file>.safetensors \
  --output_path output/example.png
```

## Limitations

Describe failure modes, style bias, text artifacts, source-image drift, and any
known unsuitable uses.

## License Notes

This LoRA is a derivative of the named base model. The adapter's distribution
and use must be compatible with the base model license and your training data
rights. Do not rely on this template as legal advice.
