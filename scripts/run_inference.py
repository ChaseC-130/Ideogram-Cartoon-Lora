#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import os
import sys
import argparse
from pathlib import Path
from PIL import Image

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
    try:
        import torch
        from toolkit.config_modules import ModelConfig
        from toolkit.lora_special import LoRASpecialNetwork
        from extensions_built_in.diffusion_models.ideogram4.ideogram4 import Ideogram4Model
    except ImportError as e:
        raise SystemExit(
            f"Error importing required modules from AI Toolkit: {e}\n"
            "Please verify --ai-toolkit-path and your Python environment."
        ) from e

    return torch, ModelConfig, LoRASpecialNetwork, Ideogram4Model

def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate 1024x1024 images using Ideogram 4 (FP8) and optionally apply a custom LoRA.")
    parser.add_argument(
        "--ai-toolkit-path",
        "--ai_toolkit_path",
        type=str,
        default=os.environ.get("AI_TOOLKIT_PATH"),
        help="Path to a local AI Toolkit checkout. Defaults to AI_TOOLKIT_PATH or ../ai-toolkit.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="ideogram-ai/ideogram-4-fp8",
        help="Hugging Face model ID or local path to base model."
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the trained LoRA .safetensors file (leave empty to run base model)."
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Weight/scale of the LoRA model (default: 1.0)."
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to a text file containing prompts (one per line) for batch generation."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="A single prompt to generate (ignored if --prompt_file is provided)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory where generated images will be saved."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps (default: 28)."
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=3.5,
        help="Guidance scale (default: 3.5)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run on (cuda, mps, cpu). Default: mps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to use."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    ai_toolkit_path = configure_ai_toolkit_path(args.ai_toolkit_path)
    torch, ModelConfig, LoRASpecialNetwork, Ideogram4Model = import_ai_toolkit_modules()
    device = args.device
    print(f"Using AI Toolkit at {ai_toolkit_path}")

    # Gather prompts
    prompts = []
    if args.prompt_file:
        file_path = Path(args.prompt_file)
        if not file_path.exists():
            print(f"Error: Prompt file not found at {args.prompt_file}")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "custom_subject style illustration of a retro robot playing chess, 8k, photorealistic"
        ]

    print(f"Loaded {len(prompts)} prompts for generation.")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base model config
    model_config = ModelConfig(
        name_or_path=args.model_id,
        arch="ideogram4",
        quantize=False,
        dtype="bf16",
        device=device
    )

    print(f"Loading Ideogram 4 base model from {args.model_id}...")
    model = Ideogram4Model(device=device, model_config=model_config, dtype="bf16")
    model.load_model()
    model.model.to(device, dtype=torch.bfloat16)

    network = None
    if args.lora_path:
        lora_file = Path(args.lora_path)
        if not lora_file.exists():
            print(f"Error: LoRA file not found at {args.lora_path}")
            sys.exit(1)

        print(f"Instantiating LoRA network and loading weights from {lora_file}...")
        network = LoRASpecialNetwork(
            text_encoder=model.text_encoder,
            unet=model.model,
            lora_dim=16,
            alpha=16,
            multiplier=args.lora_scale,
            train_text_encoder=False,
            train_unet=True,
            is_transformer=model.is_transformer,
            target_lin_modules=model.target_lora_modules,
            base_model=model,
            transformer_only=True
        )
        network.apply_to(
            model.text_encoder,
            model.model,
            apply_text_encoder=False,
            apply_unet=True
        )
        network.load_weights(str(lora_file))
        network.force_to(device, dtype=torch.bfloat16)
        network._update_torch_multiplier()
        network.eval()

    pipeline = model.get_generation_pipeline()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print("\nStarting generation...")
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Prompt: {prompt}")

        # Helper context for LoRA active state
        lora_ctx = network if network is not None else contextlib.nullcontext()

        with lora_ctx:
            with torch.no_grad():
                # Encode prompt
                print("Encoding prompt...")
                conditional_embeds = model.encode_prompt(prompt, prompt, force_all=True)
                unconditional_embeds = model.encode_prompt("", "", force_all=True)

                # Run inference
                print("Running pipeline inference...")
                images = pipeline(
                    conditional_embeds=conditional_embeds,
                    unconditional_embeds=unconditional_embeds,
                    height=1024,
                    width=1024,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator
                )

                image = images[0]

                # Save output
                filename = f"gen_{idx+1:03d}_{prompt[:30].replace(' ', '_').replace('/', '_')}.png"
                save_path = out_dir / filename
                image.save(save_path)
                print(f"Saved to {save_path}")

    print("\nAll generations completed.")

if __name__ == "__main__":
    main()
