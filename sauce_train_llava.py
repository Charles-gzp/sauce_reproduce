import argparse
from pathlib import Path

import torch

from sae_training.config import ViTSAERunnerConfig
from sae_training.vit_runner import vision_transformer_sae_runner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAE on LLaVA vision encoder with SAUCE-style settings."
    )
    parser.add_argument("--model-name", type=str, default="llava-hf/llava-1.5-7b-hf", help="LLaVA model name.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="evanarlian/imagenet_1k_resized_256",
        help="Training dataset path.",
    )
    parser.add_argument("--d-in", type=int, default=1024, help="Vision feature width at hook layer.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1_281_167,
        help="Used to estimate total_training_tokens = epochs * dataset_size.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="SAE optimization batch size.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--l1-coef", type=float, default=8e-5, help="L1 coefficient (alpha).")
    parser.add_argument("--expansion-factor", type=int, default=64, help="SAE expansion factor.")
    parser.add_argument(
        "--b-dec-init-method",
        type=str,
        default="geometric_median",
        choices=["geometric_median", "mean", "zeros"],
        help="b_dec initialization method; paper-aligned default is geometric_median.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    parser.add_argument("--checkpoint-root", type=str, default="checkpoints", help="Checkpoint root directory.")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument(
        "--max-vit-batch",
        type=int,
        default=64,
        help="Max image count per VLM forward pass (for memory control).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 中文注释：按论文口径，训练 token 数按 epochs * ImageNet-1k 训练样本数估算。
    total_training_tokens = args.epochs * args.dataset_size

    # 中文注释：这里集中定义 SAE 训练配置，后续 checkpoint 会完整保存该 cfg。
    cfg = ViTSAERunnerConfig(
        vlm_family="llava",
        class_token=False,
        sae_target_token="last",
        image_caption_prompt="Please describe this figure",
        image_width=224,
        image_height=224,
        model_name=args.model_name,
        module_name="resid",
        block_layer=-2,
        dataset_path=args.dataset_path,
        use_cached_activations=False,
        cached_activations_path=None,
        d_in=args.d_in,
        expansion_factor=args.expansion_factor,
        b_dec_init_method=args.b_dec_init_method,
        lr=args.lr,
        l1_coefficient=args.l1_coef,
        reconstruction_loss_type="l2",
        lr_scheduler_name="constantwithwarmup",
        batch_size=args.batch_size,
        lr_warm_up_steps=500,
        total_training_tokens=total_training_tokens,
        n_batches_in_store=15,
        max_batch_size_for_vit_forward_pass=args.max_vit_batch,
        use_ghost_grads=False,
        feature_sampling_method=None,
        feature_sampling_window=64,
        dead_feature_window=64,
        dead_feature_threshold=1e-6,
        log_to_wandb=args.wandb,
        wandb_project="sauce-repro",
        wandb_entity=None,
        wandb_log_frequency=20,
        device=args.device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path=args.checkpoint_root,
        # 中文注释：A800 上通常使用 bf16 以降低显存压力。
        dtype=torch.bfloat16,
    )

    sparse_autoencoder, _ = vision_transformer_sae_runner(cfg)
    final_path = Path(cfg.checkpoint_path) / f"final_{sparse_autoencoder.get_name()}.pt"
    print(f"SAE checkpoint: {final_path}")


if __name__ == "__main__":
    main()
