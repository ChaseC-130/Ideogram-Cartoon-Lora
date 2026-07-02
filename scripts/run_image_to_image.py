#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_ai_toolkit_path(ai_toolkit_path: str | None) -> Path:
    candidates = []
    if ai_toolkit_path:
        candidates.append(Path(ai_toolkit_path))
    if os.environ.get("AI_TOOLKIT_PATH"):
        candidates.append(Path(os.environ["AI_TOOLKIT_PATH"]))
    candidates.append(PROJECT_ROOT.parent / "ai-toolkit")

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if not resolved.exists():
            continue
        sys.path.insert(0, str(resolved))
        diffusion_root = resolved / "extensions_built_in" / "diffusion_models"
        if diffusion_root.exists():
            sys.path.insert(0, str(diffusion_root))
        return resolved

    searched = ", ".join(str(path) for path in candidates)
    raise SystemExit(
        "Could not find AI Toolkit. Pass --ai-toolkit-path or set "
        f"AI_TOOLKIT_PATH. Searched: {searched}"
    )


def import_ai_toolkit_modules() -> tuple[object, object, object]:
    global pad_text_features, predict_velocity

    try:
        from toolkit.config_modules import ModelConfig
        from toolkit.lora_special import LoRASpecialNetwork
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
        from extensions_built_in.diffusion_models.ideogram4.src.pipeline import (
            pad_text_features,
            predict_velocity,
        )
    except ImportError as e:
        raise SystemExit(
            f"Error importing required modules from AI Toolkit: {e}\n"
            "Please verify --ai-toolkit-path and your Python environment."
        ) from e

    return ModelConfig, LoRASpecialNetwork, Ideogram4Model

def parse_args():
    parser = argparse.ArgumentParser(description="Image-to-Image generation using Ideogram 4 (FP8) and a custom LoRA.")
    parser.add_argument(
        "--ai-toolkit-path",
        "--ai_toolkit_path",
        type=str,
        default=os.environ.get("AI_TOOLKIT_PATH"),
        help="Path to a local AI Toolkit checkout. Defaults to AI_TOOLKIT_PATH or ../ai-toolkit.",
    )
    parser.add_argument("--image_path", type=str, required=True, help="Path to the source input image.")
    parser.add_argument("--model_id", type=str, default="ideogram-ai/ideogram-4-fp8", help="Hugging Face model ID or local path to base model.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt describing the cartoonish output (include trigger word).")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the trained LoRA .safetensors file.")
    parser.add_argument("--strength", type=float, default=0.6, help="Strength of the transformation (0.0 = original image, 1.0 = pure noise). Default: 0.6.")
    parser.add_argument("--output_path", type=str, default="output/img2img_output.png", help="Path to save the output image.")
    parser.add_argument("--steps", type=int, default=28, help="Number of total inference steps (default: 28).")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale (default: 3.5).")
    parser.add_argument("--device", type=str, default="mps", help="Device to run on (mps, cuda, cpu). Default: mps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    return parser.parse_args()

@torch.no_grad()
def run_img2img(
    model,
    network,
    image_path,
    prompt,
    strength,
    num_steps,
    guidance_scale,
    device,
    generator
):
    dtype = model.torch_dtype
    transformer = model.model
    patch = model.patch_size

    # 1. Load and preprocess source image
    src_img = Image.open(image_path).convert("RGB")
    # Resize to multiples of bucket size (16 for Ideogram 4)
    w, h = src_img.size
    w = (w // 16) * 16
    h = (h // 16) * 16
    src_img = src_img.resize((w, h), Image.Resampling.LANCZOS)
    print(f"Resized source image to {w}x{h} for processing.")

    # Normalize to [-1, 1]
    img_tensor = torch.from_numpy(np.array(src_img)).permute(2, 0, 1).float() / 127.5 - 1.0

    # 2. Encode source image into latent space
    print("Encoding source image into latent space...")
    clean_latents = model.encode_images([img_tensor], device=device, dtype=model.vae_torch_dtype).to(torch.float32)

    # 3. Create noise and blend based on strength (Flow Matching convention)
    print(f"Blending noise with strength={strength}...")
    noise = randn_tensor(clean_latents.shape, generator=generator, device=device, dtype=torch.float32)

    # x_t = (1 - t) * x_0 + t * x_1
    # where t = strength (1.0 = pure noise, 0.0 = clean image)
    latents = (1.0 - strength) * clean_latents + strength * noise

    # 4. Prepare Scheduler and Timesteps
    scheduler = model.get_train_scheduler()
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    # Filter timesteps to start at the requested strength
    start_timestep_val = strength * 1000.0
    timesteps = [t for t in timesteps if t <= start_timestep_val]
    print(f"Running inference for {len(timesteps)} / {num_steps} denoising steps.")

    # 5. Encode Prompts
    # Wrap prompt in JSON-style to match LoRA training format
    wrapped_prompt = f'custom_subject {{\n  "caption": "{prompt}"\n}}'
    print(f"Using prompt: {repr(wrapped_prompt)}")

    conditional_embeds = model.get_prompt_embeds(wrapped_prompt)
    unconditional_embeds = model.get_prompt_embeds("")

    cond_feats, cond_mask = pad_text_features(conditional_embeds.text_embeds, device, dtype)
    uncond_feats, uncond_mask = pad_text_features(unconditional_embeds.text_embeds, device, dtype)

    do_cfg = guidance_scale != 1.0

    # 6. Denoising Loop
    with network:
        for t in timesteps:
            t01 = (t / 1000.0).to(device).expand(latents.shape[0])
            v_cond = predict_velocity(transformer, latents.to(dtype), t01, cond_feats, cond_mask)
            if do_cfg:
                v_uncond = predict_velocity(transformer, latents.to(dtype), t01, uncond_feats, uncond_mask)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            latents = scheduler.step(v.to(torch.float32), t, latents, return_dict=False)[0]

    # 7. Decode Latents
    print("Decoding generated latents...")
    images = model.decode_latents(latents, device=device, dtype=dtype)
    images = images.float().clamp(-1.0, 1.0)
    images = ((images + 1.0) * 127.5).round().to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    return Image.fromarray(images[0])

def main():
    args = parse_args()
    ai_toolkit_path = configure_ai_toolkit_path(args.ai_toolkit_path)
    ModelConfig, LoRASpecialNetwork, Ideogram4Model = import_ai_toolkit_modules()

    device = args.device
    print(f"Using AI Toolkit at {ai_toolkit_path}")

    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(
        name_or_path=args.model_id,
        arch="ideogram4",
        quantize=False,
        dtype="bf16",
        device=device
    )

    print("Loading base model...")
    model = Ideogram4Model(device=device, model_config=model_config, dtype="bf16")
    model.load_model()
    model.model.to(device, dtype=torch.bfloat16)

    print("Loading LoRA weights...")
    network = LoRASpecialNetwork(
        text_encoder=model.text_encoder,
        unet=model.model,
        lora_dim=16,
        alpha=16,
        multiplier=1.0,
        train_text_encoder=False,
        train_unet=True,
        is_transformer=model.is_transformer,
        target_lin_modules=model.target_lora_modules,
        base_model=model,
        transformer_only=True
    )
    network.apply_to(model.text_encoder, model.model, apply_text_encoder=False, apply_unet=True)
    network.load_weights(args.lora_path)
    network.force_to(device, dtype=torch.bfloat16)
    network._update_torch_multiplier()
    network.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)

    output_image = run_img2img(
        model=model,
        network=network,
        image_path=args.image_path,
        prompt=args.prompt,
        strength=args.strength,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        device=device,
        generator=generator
    )

    output_image.save(output_file)
    print(f"\nImage-to-Image generation complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
