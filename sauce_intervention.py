import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image

from sae_training.hooked_vit import Hook, HookedVisionTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.sauce import build_sauce_intervention_hook, get_token_index


def parse_args():
    parser = argparse.ArgumentParser(description="Apply SAUCE feature intervention during ViT forward.")
    parser.add_argument("--sae-path", type=str, required=True, help="Path to SAE checkpoint (.pt)")
    parser.add_argument("--image-paths", nargs="+", required=True, help="One or more image file paths")
    parser.add_argument("--text", type=str, default="Please describe this figure", help="Text prompt for CLIP processor")
    parser.add_argument("--feature-indices", type=str, default=None, help="Comma-separated feature ids, e.g. 10,42,1337")
    parser.add_argument("--feature-indices-path", type=str, default=None, help="Path to .pt tensor containing selected feature indices")
    parser.add_argument("--gamma", type=float, default=-0.5, help="Intervention scaling factor for selected features")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    return parser.parse_args()


def load_feature_indices(args) -> List[int]:
    if args.feature_indices_path is not None:
        indices = torch.load(args.feature_indices_path, map_location="cpu").tolist()
        return [int(x) for x in indices]
    if args.feature_indices is None:
        raise ValueError("Provide --feature-indices or --feature-indices-path.")
    return [int(x.strip()) for x in args.feature_indices.split(",") if x.strip()]


@torch.no_grad()
def main():
    args = parse_args()
    sae = SparseAutoencoder.load_from_pretrained(args.sae_path).eval()
    if args.device is not None:
        sae = sae.to(args.device)

    # 按训练配置选择对应的 VLM 家族（CLIP/LLaVA）。
    model = HookedVisionTransformer(
        sae.cfg.model_name,
        device=str(sae.cfg.device),
        vlm_family=sae.cfg.vlm_family,
    )
    model.eval()

    token_index = get_token_index(
        class_token=sae.cfg.class_token,
        sae_target_token=sae.cfg.sae_target_token,
    )
    feature_indices = load_feature_indices(args)
    intervention_hook_fn = build_sauce_intervention_hook(
        sparse_autoencoder=sae,
        selected_feature_indices=feature_indices,
        gamma=args.gamma,
        token_index=token_index,
    )
    hooks = [Hook(sae.cfg.block_layer, sae.cfg.module_name, intervention_hook_fn, return_module_output=False)]

    images = [Image.open(Path(p)).convert("RGB") for p in args.image_paths]
    text_inputs = [args.text] * len(images)
    # 统一输入构造，支持 CLIP/LLaVA。
    inputs = model.prepare_inputs(images=images, text=text_inputs, device=sae.cfg.device)

    base_output = model(return_type="output", **inputs)
    edited_output = model.run_with_hooks(hooks, return_type="output", **inputs)

    base_logits = base_output.logits_per_image.detach().cpu()
    edited_logits = edited_output.logits_per_image.detach().cpu()
    diff = edited_logits - base_logits

    print("Base logits_per_image:")
    print(base_logits)
    print("\nEdited logits_per_image:")
    print(edited_logits)
    print("\nDelta logits_per_image:")
    print(diff)


if __name__ == "__main__":
    main()
