import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset

from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.sauce import get_token_index
from sae_training.sparse_autoencoder import SparseAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check SAE reconstruction quality (EV) and global sparsity ratio (nz)."
    )
    parser.add_argument("--sae-path", type=str, required=True, help="Path to SAE checkpoint.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="evanarlian/imagenet_1k_resized_256",
        help="HuggingFace dataset path.",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split, e.g. val/train/test.")
    parser.add_argument("--max-images", type=int, default=500, help="Number of images to sample.")
    parser.add_argument("--batch-size", type=int, default=8, help="Forward batch size.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please describe this figure",
        help="Prompt used for multimodal forward.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to save summary JSON.",
    )
    return parser.parse_args()


def summarize_tensor(x: torch.Tensor):
    x = x.float()
    return {
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "std": float(x.std().item()),
        "q05": float(torch.quantile(x, 0.05).item()),
        "q95": float(torch.quantile(x, 0.95).item()),
    }


@torch.no_grad()
def main():
    args = parse_args()

    sae = SparseAutoencoder.load_from_pretrained(args.sae_path).eval()
    cfg = sae.cfg
    model = HookedVisionTransformer(
        cfg.model_name,
        device=str(cfg.device),
        vlm_family=cfg.vlm_family,
        torch_dtype=cfg.dtype,
    )
    model.eval()

    # 只抽样前 max_images 张，作为健康检查集。
    ds = load_dataset(args.dataset_path, split=args.split).select(range(args.max_images))
    token_idx = get_token_index(cfg.class_token, cfg.sae_target_token)
    hook_loc = (cfg.block_layer, cfg.module_name)

    ev_all = []
    nz_all = []
    for start in range(0, len(ds), args.batch_size):
        images = ds[start : start + args.batch_size]["image"]
        inputs = model.prepare_inputs(
            images=images,
            text=[args.prompt] * len(images),
            device=cfg.device,
        )
        _, cache = model.run_with_cache([hook_loc], **inputs)
        x = cache[hook_loc][:, token_idx, :]
        x_hat, z, _, _, _, _ = sae(x)

        per_recon = (x_hat - x).float().pow(2).sum(dim=-1)
        total_var = x.float().pow(2).sum(dim=-1).clamp_min(1e-8)
        ev = 1 - (per_recon / total_var)
        ev_all.append(ev.detach().cpu())
        # 统计 batch 级别的全局非零比例（所有 SAE 维度）
        nz_all.append((z > 0).float().mean().detach().cpu())

    ev_all = torch.cat(ev_all, dim=0)
    nz_all = torch.stack(nz_all, dim=0)

    ev_summary = summarize_tensor(ev_all)
    nz_summary = summarize_tensor(nz_all)

    print(f"EV mean: {ev_summary['mean']:.6f}")
    print(f"EV median: {ev_summary['median']:.6f}")
    print(f"EV q05/q95: {ev_summary['q05']:.6f} / {ev_summary['q95']:.6f}")
    print(f"NZ ratio mean: {nz_summary['mean']:.6f}")
    print(f"NZ ratio median: {nz_summary['median']:.6f}")
    print(f"NZ ratio q05/q95: {nz_summary['q05']:.6f} / {nz_summary['q95']:.6f}")

    result = {
        "sae_path": args.sae_path,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "max_images": args.max_images,
        "batch_size": args.batch_size,
        "prompt": args.prompt,
        "token_index": token_idx,
        "hook_loc": {"block_layer": cfg.block_layer, "module_name": cfg.module_name},
        "ev": ev_summary,
        "nz_ratio": nz_summary,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
