import argparse
import csv
from pathlib import Path
from typing import List

import torch

from sae_training.sparse_autoencoder import SparseAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect activation distribution of selected SAUCE features on pos/neg sets."
    )
    parser.add_argument("--sae-path", type=str, required=True, help="Path to SAE checkpoint.")
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory containing selected_feature_indices.pt.",
    )
    parser.add_argument(
        "--acts-dir",
        type=str,
        default=None,
        help="Directory containing pos_token_activations.pt and neg_token_activations.pt. Default: out-dir",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for SAE encoding.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="selected_feature_distribution",
        help="Prefix for output files.",
    )
    return parser.parse_args()


@torch.no_grad()
def encode_selected_features(
    sae: SparseAutoencoder,
    token_acts: torch.Tensor,
    selected_indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    # 分批前向，提取选中特征的激活值，避免显存峰值过高。
    outs: List[torch.Tensor] = []
    for start in range(0, token_acts.shape[0], batch_size):
        batch = token_acts[start : start + batch_size].to(sae.device)
        _, z, _, _, _, _ = sae(batch)
        outs.append(z[:, selected_indices].float().cpu())
    return torch.cat(outs, dim=0)


def describe_flat(name: str, x: torch.Tensor):
    q = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float32)
    qs = torch.quantile(x.float(), q)
    print(f"[{name}] mean={x.mean().item():.6f} std={x.std().item():.6f} min={x.min().item():.6f} max={x.max().item():.6f}")
    print(
        f"[{name}] q01={qs[0].item():.6f} q05={qs[1].item():.6f} q50={qs[2].item():.6f} q95={qs[3].item():.6f} q99={qs[4].item():.6f}"
    )


def save_feature_stats_csv(
    save_path: Path,
    selected_indices: torch.Tensor,
    mu_pos: torch.Tensor,
    mu_neg: torch.Tensor,
    nz_pos: torch.Tensor,
    nz_neg: torch.Tensor,
):
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "feature_id",
                "mu_pos",
                "mu_neg",
                "margin_mu_pos_minus_mu_neg",
                "ratio_mu_pos_over_mu_neg",
                "nz_rate_pos",
                "nz_rate_neg",
                "margin_nz_pos_minus_nz_neg",
            ]
        )
        for i in range(selected_indices.numel()):
            fid = int(selected_indices[i].item())
            p = float(mu_pos[i].item())
            n = float(mu_neg[i].item())
            writer.writerow(
                [
                    fid,
                    p,
                    n,
                    p - n,
                    (p + 1e-8) / (n + 1e-8),
                    float(nz_pos[i].item()),
                    float(nz_neg[i].item()),
                    float(nz_pos[i].item() - nz_neg[i].item()),
                ]
            )


def maybe_save_plot(pos_vals: torch.Tensor, neg_vals: torch.Tensor, save_png: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skip png plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(pos_vals.numpy(), bins=80, alpha=0.55, label="pos", density=True)
    plt.hist(neg_vals.numpy(), bins=80, alpha=0.55, label="neg", density=True)
    plt.xlabel("Selected feature activation")
    plt.ylabel("Density")
    plt.title("Activation Distribution of Selected Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_png, dpi=160)
    plt.close()
    print(f"Saved plot: {save_png}")


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    acts_dir = Path(args.acts_dir) if args.acts_dir is not None else out_dir

    idx_path = out_dir / "selected_feature_indices.pt"
    pos_acts_path = acts_dir / "pos_token_activations.pt"
    neg_acts_path = acts_dir / "neg_token_activations.pt"

    if not idx_path.exists():
        raise FileNotFoundError(f"Missing file: {idx_path}")
    if not pos_acts_path.exists():
        raise FileNotFoundError(f"Missing file: {pos_acts_path}")
    if not neg_acts_path.exists():
        raise FileNotFoundError(f"Missing file: {neg_acts_path}")

    sae = SparseAutoencoder.load_from_pretrained(args.sae_path).eval()
    selected_indices = torch.load(idx_path, map_location="cpu").long()
    pos_acts = torch.load(pos_acts_path, map_location="cpu")
    neg_acts = torch.load(neg_acts_path, map_location="cpu")

    if selected_indices.numel() == 0:
        raise ValueError("selected_feature_indices is empty.")
    if pos_acts.ndim != 2 or neg_acts.ndim != 2:
        raise ValueError("pos/neg activations must be rank-2 tensors [N, d_in].")
    if pos_acts.shape[1] != sae.d_in or neg_acts.shape[1] != sae.d_in:
        raise ValueError(
            f"Activation d_in mismatch. SAE d_in={sae.d_in}, pos={pos_acts.shape[1]}, neg={neg_acts.shape[1]}"
        )

    print(f"Selected k = {selected_indices.numel()}")
    print(f"POS shape = {tuple(pos_acts.shape)}, NEG shape = {tuple(neg_acts.shape)}")

    pos_sel = encode_selected_features(sae, pos_acts, selected_indices, args.batch_size)
    neg_sel = encode_selected_features(sae, neg_acts, selected_indices, args.batch_size)

    # 每个特征的统计
    mu_pos = pos_sel.mean(dim=0)
    mu_neg = neg_sel.mean(dim=0)
    nz_pos = (pos_sel > 0).float().mean(dim=0)
    nz_neg = (neg_sel > 0).float().mean(dim=0)

    margin_mu = mu_pos - mu_neg
    margin_nz = nz_pos - nz_neg

    print(f"mean(mu_pos)={mu_pos.mean().item():.6f}, mean(mu_neg)={mu_neg.mean().item():.6f}, mean(margin_mu)={margin_mu.mean().item():.6f}")
    print(f"mean(nz_pos)={nz_pos.mean().item():.6f}, mean(nz_neg)={nz_neg.mean().item():.6f}, mean(margin_nz)={margin_nz.mean().item():.6f}")

    # 扁平化分布（所有样本 x 所选特征）
    pos_flat = pos_sel.reshape(-1)
    neg_flat = neg_sel.reshape(-1)
    describe_flat("pos_flat", pos_flat)
    describe_flat("neg_flat", neg_flat)

    # 显示 margin 最大/最小的特征，快速判断“是不是大多特征都没区分度”。
    top_vals, top_idx = torch.topk(margin_mu, k=min(10, margin_mu.numel()))
    bot_vals, bot_idx = torch.topk(-margin_mu, k=min(10, margin_mu.numel()))
    print("Top-10 by margin_mu (mu_pos - mu_neg):")
    for rank, (v, i) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), start=1):
        print(f"  {rank:02d}. fid={int(selected_indices[i].item())} margin_mu={v:.6f} nz_pos={nz_pos[i].item():.4f} nz_neg={nz_neg[i].item():.4f}")
    print("Bottom-10 by margin_mu:")
    for rank, (v, i) in enumerate(zip(bot_vals.tolist(), bot_idx.tolist()), start=1):
        print(f"  {rank:02d}. fid={int(selected_indices[i].item())} margin_mu={-v:.6f} nz_pos={nz_pos[i].item():.4f} nz_neg={nz_neg[i].item():.4f}")

    csv_path = out_dir / f"{args.save_prefix}.csv"
    save_feature_stats_csv(csv_path, selected_indices, mu_pos, mu_neg, nz_pos, nz_neg)
    print(f"Saved csv: {csv_path}")

    png_path = out_dir / f"{args.save_prefix}.png"
    maybe_save_plot(pos_flat, neg_flat, png_path)


if __name__ == "__main__":
    main()
