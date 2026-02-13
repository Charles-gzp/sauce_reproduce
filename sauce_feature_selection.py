import argparse
from pathlib import Path

import torch

from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.sauce import (
    get_feature_activations_for_tokens,
    select_features_by_pos_neg_correlation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SAUCE correlation-based feature selection.")
    parser.add_argument("--sae-path", type=str, required=True, help="Path to trained SAE checkpoint (.pt)")
    parser.add_argument("--pos-acts", type=str, required=True, help="Path to positive token activations tensor [N, d_in]")
    parser.add_argument("--neg-acts", type=str, required=True, help="Path to negative token activations tensor [N, d_in]")
    parser.add_argument("--top-k", type=int, required=True, help="Number of features to keep")
    parser.add_argument("--delta", type=float, default=1e-6, help="Stability constant in score normalization")
    parser.add_argument("--out-dir", type=str, default="sauce_outputs", help="Output folder")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size for SAE feature activation pass")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = SparseAutoencoder.load_from_pretrained(args.sae_path).eval()

    pos_acts = torch.load(args.pos_acts, map_location=sae.device).to(sae.device)
    neg_acts = torch.load(args.neg_acts, map_location=sae.device).to(sae.device)
    if pos_acts.ndim != 2 or neg_acts.ndim != 2:
        raise ValueError("Expected rank-2 activation tensors [N, d_in].")
    if pos_acts.shape[1] != sae.d_in or neg_acts.shape[1] != sae.d_in:
        raise ValueError(
            f"Activation dimension mismatch. Expected d_in={sae.d_in}, got pos={pos_acts.shape[1]}, neg={neg_acts.shape[1]}"
        )

    pos_feature_acts = get_feature_activations_for_tokens(sae, pos_acts, batch_size=args.batch_size)
    neg_feature_acts = get_feature_activations_for_tokens(sae, neg_acts, batch_size=args.batch_size)

    result = select_features_by_pos_neg_correlation(
        pos_feature_acts=pos_feature_acts,
        neg_feature_acts=neg_feature_acts,
        top_k=args.top_k,
        delta=args.delta,
    )

    torch.save(result.topk_indices.detach().cpu(), out_dir / "selected_feature_indices.pt")
    torch.save(result.scores.detach().cpu(), out_dir / "feature_scores.pt")
    torch.save(result.mu_pos.detach().cpu(), out_dir / "mu_pos.pt")
    torch.save(result.mu_neg.detach().cpu(), out_dir / "mu_neg.pt")

    print(f"Saved selection outputs to: {out_dir.resolve()}")
    print(f"Top-{args.top_k} feature indices: {result.topk_indices.detach().cpu().tolist()[:20]} ...")


if __name__ == "__main__":
    main()
