import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import trange

from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.sauce import get_token_index


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare positive/negative token activations for SAUCE feature selection.")
    parser.add_argument("--sae-path", type=str, required=True, help="Path to SAE checkpoint (.pt), used for config/model metadata")
    parser.add_argument("--pos-dir", type=str, required=True, help="Directory of positive images (contains target concept)")
    parser.add_argument("--neg-dir", type=str, required=True, help="Directory of negative images (without target concept)")
    parser.add_argument("--out-dir", type=str, default="sauce_outputs", help="Output directory")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt used for multimodal forward")
    parser.add_argument("--batch-size", type=int, default=256, help="Forward pass image batch size")
    parser.add_argument("--max-pos", type=int, default=0, help="If >0, limit number of positive images")
    parser.add_argument("--max-neg", type=int, default=0, help="If >0, limit number of negative images")
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


@torch.no_grad()
def collect_token_activations(model, cfg, image_paths: List[Path], prompt: str, batch_size: int) -> torch.Tensor:
    token_index = get_token_index(class_token=cfg.class_token, sae_target_token=cfg.sae_target_token)
    all_acts: List[torch.Tensor] = []
    hook_locs = [(cfg.block_layer, cfg.module_name)]

    for start in trange(0, len(image_paths), batch_size, desc="Extracting token activations"):
        # 分批前向，避免显存溢出。
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        text_inputs = [prompt] * len(images)
        # 统一输入构造，支持 CLIP/LLaVA。
        inputs = model.prepare_inputs(images=images, text=text_inputs, device=cfg.device)
        cache = model.run_with_cache(hook_locs, **inputs)[1]
        acts = cache[(cfg.block_layer, cfg.module_name)][:, token_index, :].detach().cpu()
        all_acts.append(acts)
    return torch.cat(all_acts, dim=0)


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = SparseAutoencoder.load_from_pretrained(args.sae_path).eval()
    cfg = sae.cfg
    prompt = args.prompt if args.prompt is not None else (cfg.image_caption_prompt or "Please describe this figure")

    # 按训练配置选择对应的 VLM 家族（CLIP/LLaVA），并使用相同精度。
    model = HookedVisionTransformer(
        cfg.model_name,
        device=str(cfg.device),
        vlm_family=cfg.vlm_family,
        torch_dtype=cfg.dtype,
    )
    model.eval()

    pos_paths = list_images(Path(args.pos_dir))
    neg_paths = list_images(Path(args.neg_dir))
    if args.max_pos > 0:
        pos_paths = pos_paths[: args.max_pos]
    if args.max_neg > 0:
        neg_paths = neg_paths[: args.max_neg]

    if len(pos_paths) == 0 or len(neg_paths) == 0:
        raise ValueError("Positive/negative image folders must contain at least one image.")

    pos_acts = collect_token_activations(model, cfg, pos_paths, prompt=prompt, batch_size=args.batch_size)
    neg_acts = collect_token_activations(model, cfg, neg_paths, prompt=prompt, batch_size=args.batch_size)

    torch.save(pos_acts, out_dir / "pos_token_activations.pt")
    torch.save(neg_acts, out_dir / "neg_token_activations.pt")

    print(f"Saved pos activations: {(out_dir / 'pos_token_activations.pt').resolve()}")
    print(f"Saved neg activations: {(out_dir / 'neg_token_activations.pt').resolve()}")
    print(f"pos shape: {tuple(pos_acts.shape)}, neg shape: {tuple(neg_acts.shape)}")


if __name__ == "__main__":
    main()
