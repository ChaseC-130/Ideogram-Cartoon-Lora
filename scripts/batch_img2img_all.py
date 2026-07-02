from __future__ import annotations

import sys
import os
import gc
import argparse
import json
import cv2
from pathlib import Path
import torch
import numpy as np
from PIL import Image
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
        from ideogram4.ideogram4 import Ideogram4Model
        from ideogram4.src.pipeline import pad_text_features, predict_velocity
    except ImportError as e:
        raise SystemExit(
            f"Error importing required modules from AI Toolkit: {e}\n"
            "Please verify --ai-toolkit-path and your Python environment."
        ) from e

    return ModelConfig, LoRASpecialNetwork, Ideogram4Model

@torch.no_grad()
def run_img2img(
    model,
    transformer,
    scheduler,
    cond_feats,
    cond_mask,
    uncond_feats,
    uncond_mask,
    img_tensor,
    strength,
    num_steps,
    guidance_scale,
    device,
    generator
):
    dtype = model.torch_dtype
    clean_latents = model.encode_images([img_tensor], device=device, dtype=model.vae_torch_dtype).to(torch.float32)
    noise = randn_tensor(clean_latents.shape, generator=generator, device=device, dtype=torch.float32)
    latents = (1.0 - strength) * clean_latents + strength * noise

    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps
    start_timestep_val = strength * 1000.0
    timesteps = [t for t in timesteps if t <= start_timestep_val]

    do_cfg = guidance_scale != 1.0
    for t in timesteps:
        t01 = (t / 1000.0).to(device).expand(latents.shape[0])
        v_cond = predict_velocity(transformer, latents.to(dtype), t01, cond_feats, cond_mask)
        if do_cfg:
            v_uncond = predict_velocity(transformer, latents.to(dtype), t01, uncond_feats, uncond_mask)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        latents = scheduler.step(v.to(torch.float32), t, latents, return_dict=False)[0]

    images = model.decode_latents(latents, device=device, dtype=dtype)
    images = images.float().clamp(-1.0, 1.0)
    images = ((images + 1.0) * 127.5).round().to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    return Image.fromarray(images[0])

def apply_guide_b_preprocessing(pil_img):
    open_cv_image = np.array(pil_img)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # 1. Median 9 Blur to erase high-frequency scales/quills
    med = cv2.medianBlur(open_cv_image, 9)

    # 2. Bilateral Filter (d=9, sigma=80) to smooth colors while preserving edges
    filtered = cv2.bilateralFilter(med, 9, 80, 80)

    filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    return Image.fromarray(filtered_rgb)

def parse_args():
    parser = argparse.ArgumentParser(description="Batch process images through blur preprocessing + Ideogram img2img.")
    parser.add_argument(
        "--ai-toolkit-path",
        "--ai_toolkit_path",
        type=str,
        default=os.environ.get("AI_TOOLKIT_PATH"),
        help="Path to a local AI Toolkit checkout. Defaults to AI_TOOLKIT_PATH or ../ai-toolkit.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images processed for testing.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "input" / "images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "img2img_assets",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=None,
        help="Optional directory to save Guide B blurred intermediates.",
    )
    parser.add_argument(
        "--prompt-cache",
        type=Path,
        default=PROJECT_ROOT / "output" / "img2img_prompts.json",
    )
    parser.add_argument(
        "--clean-prompt-cache",
        type=Path,
        default=PROJECT_ROOT / "output" / "img2img_prompts_clean.json",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=PROJECT_ROOT / "weights" / "cartoon_lora.safetensors",
    )
    parser.add_argument("--model-id", default="ideogram-ai/ideogram-4-fp8")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--strength", type=float, default=0.70)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Regenerate outputs that already exist.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    ai_toolkit_path = configure_ai_toolkit_path(args.ai_toolkit_path)
    ModelConfig, LoRASpecialNetwork, Ideogram4Model = import_ai_toolkit_modules()

    device = args.device
    model_id = args.model_id
    lora_path = str(args.lora_path)
    print(f"Using AI Toolkit at {ai_toolkit_path}")

    source_dir = args.source_dir
    output_base_dir = args.output_dir
    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}")
        sys.exit(1)
    if not args.lora_path.exists():
        print(f"Error: LoRA file not found: {args.lora_path}")
        sys.exit(1)

    cache_path = args.prompt_cache
    clean_cache_path = args.clean_prompt_cache

    # Load prompts cache (fallback to original prompts if clean file is not fully built yet)
    if clean_cache_path.exists():
        print(f"Loading cleaned prompts from {clean_cache_path}...")
        prompts = load_json(clean_cache_path)
    elif cache_path.exists():
        print(f"Warning: Clean prompts not found. Falling back to original prompts from {cache_path}...")
        prompts = load_json(cache_path)
    else:
        print(f"Error: Prompt cache not found: {cache_path}")
        sys.exit(1)

    # Also load the original prompts as a fallback for missing keys in the cleaned cache
    original_prompts = {}
    if cache_path.exists():
        original_prompts = load_json(cache_path)

    # Find all PNG files (excluding .import files and outputs)
    image_paths = []
    for root, dirs, files in os.walk(source_dir):
        if "img2img_assets" in root or "test_run" in root:
            continue
        for file in files:
            if file.endswith(".png") and not file.startswith("."):
                image_paths.append(Path(root) / file)
    image_paths.sort()

    print(f"Found {len(image_paths)} source images.")

    # 1. Load model
    print("Loading base model...")
    model_config = ModelConfig(
        name_or_path=model_id,
        arch="ideogram4",
        quantize=False,
        dtype="bf16",
        device=device
    )
    model = Ideogram4Model(device=device, model_config=model_config, dtype="bf16")
    model.load_model()
    model.model.to(device, dtype=torch.bfloat16)

    # 2. Load LoRA
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
    network.load_weights(lora_path)
    network.force_to(device, dtype=torch.bfloat16)
    network._update_torch_multiplier()
    network.eval()

    transformer = model.model
    scheduler = model.get_train_scheduler()

    processed = 0
    skipped = 0
    errors = 0

    for path in image_paths:
        rel_key = str(path.relative_to(source_dir))

        # Check if prompt exists in cleaned prompts, else fallback to original
        prompt_text = prompts.get(rel_key, original_prompts.get(rel_key))
        if not prompt_text:
            print(f"Skipping {rel_key}: No prompt found in cache.")
            continue

        # Determine output path, keeping folder structure
        out_path = output_base_dir / rel_key

        # Skip if already generated
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        if args.limit is not None and processed >= args.limit:
            print(f"Limit of {args.limit} reached. Stopping.")
            break

        print(f"[{processed+1}] Processing {rel_key}...")

        try:
            # Ensure parent directories exist
            out_path.parent.mkdir(parents=True, exist_ok=True)

            src_img = Image.open(path).convert("RGB")
            w, h = src_img.size
            w = (w // 16) * 16
            h = (h // 16) * 16
            src_img = src_img.resize((w, h), Image.Resampling.LANCZOS)

            # Apply Guide B Preprocessing (Median 9 + Bilateral 9,80,80)
            preprocessed_img = apply_guide_b_preprocessing(src_img)
            if args.preprocessed_dir is not None:
                preprocessed_path = args.preprocessed_dir / rel_key
                preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
                preprocessed_img.save(preprocessed_path)

            img_tensor = torch.from_numpy(np.array(preprocessed_img)).permute(2, 0, 1).float() / 127.5 - 1.0

            wrapped_prompt = "custom_subject {\n  \"caption\": " + json.dumps(prompt_text) + "\n}"

            conditional_embeds = model.get_prompt_embeds(wrapped_prompt)
            unconditional_embeds = model.get_prompt_embeds("")

            cond_feats, cond_mask = pad_text_features(conditional_embeds.text_embeds, device, model.torch_dtype)
            uncond_feats, uncond_mask = pad_text_features(unconditional_embeds.text_embeds, device, model.torch_dtype)

            generator = torch.Generator(device=device).manual_seed(args.seed)
            with network:
                out_img = run_img2img(
                    model, transformer, scheduler, cond_feats, cond_mask, uncond_feats, uncond_mask,
                    img_tensor, args.strength, args.steps, args.guidance, device, generator
                )

            # Save output
            out_img.save(out_path)
            processed += 1

            # Garbage collection & free memory to keep MPS stable
            del img_tensor, conditional_embeds, unconditional_embeds, cond_feats, cond_mask, uncond_feats, uncond_mask, generator, out_img
            gc.collect()
            if device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception as e:
            errors += 1
            print(f"Error processing {rel_key}: {e}")
            gc.collect()
            if device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print(f"Successfully processed {processed} images. skipped={skipped}, errors={errors}")
    if errors:
        sys.exit(1)

if __name__ == "__main__":
    main()
