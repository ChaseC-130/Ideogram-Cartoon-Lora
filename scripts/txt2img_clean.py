from __future__ import annotations

import sys
import os
import gc
import json
import argparse
import contextlib
from pathlib import Path

import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_ai_toolkit_path(ai_toolkit_path):
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
    raise SystemExit("Could not find AI Toolkit; pass --ai-toolkit-path.")


def import_ai_toolkit_modules():
    global pad_text_features, predict_velocity
    from toolkit.config_modules import ModelConfig
    from toolkit.lora_special import LoRASpecialNetwork
    from ideogram4.ideogram4 import Ideogram4Model
    from ideogram4.src.pipeline import pad_text_features, predict_velocity
    return ModelConfig, LoRASpecialNetwork, Ideogram4Model


# Same clean caption wrapping the img2img batch uses (this encoding stays clear
# of the base model's baked-in safety placeholder; encode_prompt(force_all) does not).
def wrap(caption: str) -> str:
    return "custom_subject {\n  \"caption\": " + json.dumps(caption) + "\n}"


@torch.no_grad()
def generate(model, transformer, scheduler, cond, uncond, size, steps, guidance, device, generator):
    dtype = model.torch_dtype
    patch = model.patch_size
    ae_scale = model.vae_scale_factor
    gh = size // (ae_scale * patch)
    gw = size // (ae_scale * patch)
    latent_channels = transformer.config.in_channels

    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps

    latents = randn_tensor((1, latent_channels, gh, gw), generator=generator, device=device, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma

    cond_feats, cond_mask = pad_text_features(cond.text_embeds, device, dtype)
    uncond_feats, uncond_mask = pad_text_features(uncond.text_embeds, device, dtype)
    do_cfg = guidance != 1.0

    for t in timesteps:
        t01 = (t / 1000.0).to(device).expand(latents.shape[0])
        v_cond = predict_velocity(transformer, latents.to(dtype), t01, cond_feats, cond_mask)
        if do_cfg:
            v_uncond = predict_velocity(transformer, latents.to(dtype), t01, uncond_feats, uncond_mask)
            v = v_uncond + guidance * (v_cond - v_uncond)
        else:
            v = v_cond
        latents = scheduler.step(v.to(torch.float32), t, latents, return_dict=False)[0]

    images = model.decode_latents(latents, device=device, dtype=dtype)
    images = images.float().clamp(-1.0, 1.0)
    images = ((images + 1.0) * 127.5).round().to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    return Image.fromarray(images[0])


def parse_args():
    p = argparse.ArgumentParser(description="Text-to-image for Ideogram 4 + LoRA on the clean caption path (no init image).")
    p.add_argument("--ai-toolkit-path", "--ai_toolkit_path", default=os.environ.get("AI_TOOLKIT_PATH"))
    p.add_argument("--prompt-json", type=Path, default=None, help="JSON file: {slug: caption} or {slug: {caption, seed}}.")
    p.add_argument("--prompt", type=str, default=None, help="A single caption (alternative to --prompt-json).")
    p.add_argument("--slug", type=str, default="output", help="Output filename stem when using --prompt.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lora-path", type=Path, default=PROJECT_ROOT / "weights" / "ideogram_cartoon_lora.safetensors")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--model-id", default="ideogram-ai/ideogram-4-fp8")
    p.add_argument("--device", default="mps")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=21)
    return p.parse_args()


def main():
    args = parse_args()
    ai_toolkit_path = configure_ai_toolkit_path(args.ai_toolkit_path)
    ModelConfig, LoRASpecialNetwork, Ideogram4Model = import_ai_toolkit_modules()
    device = args.device
    print(f"Using AI Toolkit at {ai_toolkit_path}")

    if args.prompt_json:
        with args.prompt_json.open() as f:
            prompts = json.load(f)
    elif args.prompt:
        prompts = {args.slug: args.prompt}
    else:
        raise SystemExit("Pass --prompt-json or --prompt.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(name_or_path=args.model_id, arch="ideogram4", quantize=False, dtype="bf16", device=device)
    model = Ideogram4Model(device=device, model_config=model_config, dtype="bf16")
    model.load_model()
    model.model.to(device, dtype=torch.bfloat16)

    network = None
    if not args.no_lora:
        print("Loading LoRA...")
        network = LoRASpecialNetwork(
            text_encoder=model.text_encoder, unet=model.model, lora_dim=16, alpha=16, multiplier=1.0,
            train_text_encoder=False, train_unet=True, is_transformer=model.is_transformer,
            target_lin_modules=model.target_lora_modules, base_model=model, transformer_only=True)
        network.apply_to(model.text_encoder, model.model, apply_text_encoder=False, apply_unet=True)
        network.load_weights(str(args.lora_path))
        network.force_to(device, dtype=torch.bfloat16)
        network._update_torch_multiplier()
        network.eval()

    transformer = model.model
    scheduler = model.get_train_scheduler()
    uncond = model.get_prompt_embeds("")

    for slug, entry in prompts.items():
        if isinstance(entry, dict):
            caption, seed = entry["caption"], int(entry.get("seed", args.seed))
        else:
            caption, seed = entry, args.seed
        print(f"Generating {slug} (seed={seed})...")
        cond = model.get_prompt_embeds(wrap(caption))
        generator = torch.Generator(device=device).manual_seed(seed)
        ctx = network if network is not None else contextlib.nullcontext()
        with ctx:
            img = generate(model, transformer, scheduler, cond, uncond, args.size, args.steps, args.guidance, device, generator)
        out = args.output_dir / f"{slug}.png"
        img.save(out)
        print(f"Saved {out}")
        gc.collect()
        if device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("DONE")


if __name__ == "__main__":
    main()
