import argparse
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import trange

from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.sauce import get_token_index
from sae_training.sparse_autoencoder import SparseAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(description="计算 SAE 特征统计量（FS / FMAV）。")
    parser.add_argument("--sae-path", type=str, required=True, help="训练好的 SAE checkpoint。")
    parser.add_argument("--dataset-path", type=str, default="evanarlian/imagenet_1k_resized_256", help="数据集路径。")
    parser.add_argument("--split", type=str, default="train", help="数据集 split。")
    parser.add_argument("--prompt", type=str, default="Please describe this figure", help="图像对应的文本 prompt。")
    parser.add_argument("--max-images", type=int, default=0, help="最多处理多少张图片（0 表示全量）。")
    parser.add_argument("--batch-size", type=int, default=64, help="前向 batch size。")
    parser.add_argument("--out-dir", type=str, default="sauce_outputs", help="输出目录。")
    return parser.parse_args()


def list_images_from_dataset(dataset, image_key: str, max_images: int) -> List[Image.Image]:
    # 拉取数据集中的图像对象列表。
    if max_images > 0:
        dataset = dataset.select(range(min(max_images, len(dataset))))
    return [item[image_key] for item in dataset]


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 SAE 与对应 VLM。
    sae = SparseAutoencoder.load_from_pretrained(args.sae_path).eval()
    cfg = sae.cfg
    model = HookedVisionTransformer(cfg.model_name, device=str(cfg.device), vlm_family=cfg.vlm_family)
    model.eval()

    # 加载数据集（ImageNet-1k）。
    dataset = load_dataset(args.dataset_path, split=args.split)
    image_key = "image"

    images = list_images_from_dataset(dataset, image_key=image_key, max_images=args.max_images)
    if len(images) == 0:
        raise ValueError("数据集为空，无法计算统计量。")

    token_index = get_token_index(class_token=cfg.class_token, sae_target_token=cfg.sae_target_token)
    hook_locs = [(cfg.block_layer, cfg.module_name)]

    # FS 与 FMAV 的累计统计量。
    d_sae = sae.d_sae
    nonzero_counts = torch.zeros(d_sae, dtype=torch.long, device=cfg.device)
    sum_acts = torch.zeros(d_sae, dtype=torch.float32, device=cfg.device)
    total_images = 0

    for start in trange(0, len(images), args.batch_size, desc="计算特征统计"):
        batch_images = images[start : start + args.batch_size]
        text_inputs = [args.prompt] * len(batch_images)
        inputs = model.prepare_inputs(images=batch_images, text=text_inputs, device=cfg.device)
        cache = model.run_with_cache(hook_locs, **inputs)[1]
        token_acts = cache[(cfg.block_layer, cfg.module_name)][:, token_index, :].to(cfg.device)
        _, feature_acts, _, _, _, _ = sae(token_acts)

        # 统计 FS / FMAV。
        nonzero = feature_acts != 0
        nonzero_counts += nonzero.sum(dim=0)
        sum_acts += feature_acts.sum(dim=0).float()
        total_images += feature_acts.shape[0]

    # 计算最终 FS / FMAV。
    fs = nonzero_counts.float() / float(total_images)
    fmav = sum_acts / nonzero_counts.clamp_min(1).float()

    torch.save(fs.detach().cpu(), out_dir / "feature_sparsity_fs.pt")
    torch.save(fmav.detach().cpu(), out_dir / "feature_mean_activation_fmav.pt")

    print(f"FS 保存于: {(out_dir / 'feature_sparsity_fs.pt').resolve()}")
    print(f"FMAV 保存于: {(out_dir / 'feature_mean_activation_fmav.pt').resolve()}")


if __name__ == "__main__":
    main()
