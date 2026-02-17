import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAUCE pipeline stages with explicit, reproducible commands.")
    parser.add_argument("--sae-path", type=str, required=True, help="Trained SAE checkpoint.")
    parser.add_argument("--pos-dir", type=str, required=True, help="Positive concept image folder.")
    parser.add_argument("--neg-dir", type=str, required=True, help="Negative concept image folder.")
    parser.add_argument("--top-k", type=int, default=128, help="Top-k features to select.")
    parser.add_argument("--out-dir", type=str, default="sauce_outputs", help="Output directory.")
    parser.add_argument("--prompt", type=str, default="Please describe this figure", help="Prompt for multimodal forward.")
    parser.add_argument("--gamma", type=float, default=-0.5, help="Intervention scale.")
    parser.add_argument("--example-image", type=str, default=None, help="Optional test image path for intervention smoke test.")
    return parser.parse_args()


def run_cmd(cmd):
    # Print and execute command for reproducibility.
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: extract token activations for positive and negative sets.
    run_cmd(
        [
            sys.executable,
            "sauce_prepare_concept_activations.py",
            "--sae-path",
            args.sae_path,
            "--pos-dir",
            args.pos_dir,
            "--neg-dir",
            args.neg_dir,
            "--out-dir",
            str(out_dir),
            "--prompt",
            args.prompt,
        ]
    )

    # Stage 2: select top-k features by SAUCE score.
    run_cmd(
        [
            sys.executable,
            "sauce_feature_selection.py",
            "--sae-path",
            args.sae_path,
            "--pos-acts",
            str(out_dir / "pos_token_activations.pt"),
            "--neg-acts",
            str(out_dir / "neg_token_activations.pt"),
            "--top-k",
            str(args.top_k),
            "--out-dir",
            str(out_dir),
        ]
    )

    # Stage 3: optional intervention smoke test on one image.
    if args.example_image is not None:
        run_cmd(
            [
                sys.executable,
                "sauce_intervention.py",
                "--sae-path",
                args.sae_path,
                "--image-paths",
                args.example_image,
                "--text",
                args.prompt,
                "--feature-indices-path",
                str(out_dir / "selected_feature_indices.pt"),
                "--gamma",
                str(args.gamma),
            ]
        )

    print(f"Pipeline complete. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
