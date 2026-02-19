import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize FS/FMAV tensors produced by sauce_feature_stats.py.")
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="sauce_outputs/sae_health",
        help="Directory containing feature_sparsity_fs.pt and feature_mean_activation_fmav.pt",
    )
    parser.add_argument("--out-json", type=str, default=None, help="Optional summary json path.")
    return parser.parse_args()


def summarize(x: torch.Tensor):
    x = x.float()
    return {
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
        "std": float(x.std().item()),
        "q05": float(torch.quantile(x, 0.05).item()),
        "q95": float(torch.quantile(x, 0.95).item()),
    }


def main():
    args = parse_args()
    stats_dir = Path(args.stats_dir)
    fs_path = stats_dir / "feature_sparsity_fs.pt"
    fmav_path = stats_dir / "feature_mean_activation_fmav.pt"

    if not fs_path.exists():
        raise FileNotFoundError(f"Missing file: {fs_path}")
    if not fmav_path.exists():
        raise FileNotFoundError(f"Missing file: {fmav_path}")

    fs = torch.load(fs_path, map_location="cpu")
    fmav = torch.load(fmav_path, map_location="cpu")

    fs_summary = summarize(fs)
    fmav_summary = summarize(fmav)
    fs_over_01 = float((fs.float() > 0.1).float().mean().item())

    print(f"FS mean/median: {fs_summary['mean']:.6f} / {fs_summary['median']:.6f}")
    print(f"FS > 0.1 ratio: {fs_over_01:.6f}")
    print(f"FMAV mean/median: {fmav_summary['mean']:.6f} / {fmav_summary['median']:.6f}")

    result = {
        "stats_dir": str(stats_dir),
        "fs": fs_summary,
        "fs_over_0_1_ratio": fs_over_01,
        "fmav": fmav_summary,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
