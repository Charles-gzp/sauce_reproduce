import argparse
import csv
import re
from pathlib import Path
from typing import List

import torch
from PIL import Image

from sae_training.hooked_vit import Hook, HookedVisionTransformer
from sae_training.sauce import build_sauce_intervention_hook, get_token_index
from sae_training.sparse_autoencoder import SparseAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(description="Quick sanity-check: describe images and save CSV.")
    parser.add_argument("--sae-path", type=str, required=True, help="Path to trained SAE checkpoint.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of evaluation images.")
    parser.add_argument("--out-csv", type=str, default="quick_eval.csv", help="Output CSV path.")
    parser.add_argument("--num-images", type=int, default=50, help="Number of images to evaluate.")
    parser.add_argument("--prompt", type=str, default="Please describe this figure.", help="Generation prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Max new tokens for generation.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size (use 1 to avoid OOM).")
    parser.add_argument("--feature-indices-path", type=str, default=None, help="Optional selected feature file (.pt).")
    parser.add_argument("--gamma", type=float, default=-0.5, help="Intervention scale if feature indices are provided.")
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    images.sort()
    return images


def contains_dog(text: str) -> bool:
    # 快速关键词匹配：统计描述中是否明确出现 dog/dogs。
    return re.search(r"\bdogs?\b", text.lower()) is not None


@torch.no_grad()
def generate_captions(model: HookedVisionTransformer, image_paths: List[Path], prompt: str, max_new_tokens: int, batch_size: int):
    captions = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        text_inputs = [prompt] * len(images)
        inputs = model.prepare_inputs(images=images, text=text_inputs)

        # 使用模型原生 generate，适配 LLaVA 文本输出。
        generated_ids = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        decoded = model.processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions.extend([t.strip() for t in decoded])
    return captions


@torch.no_grad()
def main():
    args = parse_args()
    sae = SparseAutoencoder.load_from_pretrained(args.sae_path).eval()
    model = HookedVisionTransformer(
        sae.cfg.model_name,
        device=str(sae.cfg.device),
        vlm_family=sae.cfg.vlm_family,
        torch_dtype=sae.cfg.dtype,
    )
    model.eval()

    image_dir = Path(args.image_dir)
    image_paths = list_images(image_dir)[: args.num_images]
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Evaluating {len(image_paths)} images from: {image_dir}")
    baseline_captions = generate_captions(
        model=model,
        image_paths=image_paths,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    edited_captions = None
    if args.feature_indices_path is not None:
        # 如果传入 feature 索引，则在同一批图片上做干预生成。
        token_index = get_token_index(
            class_token=sae.cfg.class_token,
            sae_target_token=sae.cfg.sae_target_token,
        )
        selected_indices = torch.load(args.feature_indices_path, map_location="cpu").tolist()
        hook_fn = build_sauce_intervention_hook(
            sparse_autoencoder=sae,
            selected_feature_indices=[int(i) for i in selected_indices],
            gamma=args.gamma,
            token_index=token_index,
        )
        hooks = [Hook(sae.cfg.block_layer, sae.cfg.module_name, hook_fn, return_module_output=False)]

        edited_captions = []
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start : start + args.batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            text_inputs = [args.prompt] * len(images)
            inputs = model.prepare_inputs(images=images, text=text_inputs)
            with model.hooks(hooks):
                generated_ids = model.model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            decoded = model.processor.batch_decode(generated_ids, skip_special_tokens=True)
            edited_captions.extend([t.strip() for t in decoded])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 保存 CSV，方便直接丢给 AI 或人工检查。
    fieldnames = [
        "image_path",
        "baseline_caption",
        "baseline_contains_dog",
    ]
    if edited_captions is not None:
        fieldnames += [
            "edited_caption",
            "edited_contains_dog",
        ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, path in enumerate(image_paths):
            row = {
                "image_path": str(path),
                "baseline_caption": baseline_captions[i],
                "baseline_contains_dog": contains_dog(baseline_captions[i]),
            }
            if edited_captions is not None:
                row["edited_caption"] = edited_captions[i]
                row["edited_contains_dog"] = contains_dog(edited_captions[i])
            writer.writerow(row)

    baseline_hits = sum(contains_dog(t) for t in baseline_captions)
    print(f"Saved CSV: {out_csv.resolve()}")
    print(f"Baseline contains 'dog': {baseline_hits}/{len(baseline_captions)}")
    if edited_captions is not None:
        edited_hits = sum(contains_dog(t) for t in edited_captions)
        print(f"Edited contains 'dog': {edited_hits}/{len(edited_captions)}")


if __name__ == "__main__":
    main()
