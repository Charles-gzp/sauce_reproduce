import argparse
import json
import random
from pathlib import Path
from typing import List

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Export positive/negative images for one concept from ImageNet validation split.")
    parser.add_argument("--concept", type=str, required=True, help="Concept keyword, e.g. dog")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="evanarlian/imagenet_1k_resized_256",
        help="HF dataset path",
    )
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--pos-train", type=int, default=200, help="Positive train images")
    parser.add_argument("--pos-test", type=int, default=50, help="Positive test images")
    parser.add_argument("--neg-train", type=int, default=200, help="Negative train images")
    parser.add_argument("--neg-test", type=int, default=50, help="Negative test images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-root", type=str, default="concepts", help="Output root directory")
    return parser.parse_args()


def label_match(label_names: List[str], keyword: str) -> List[bool]:
    keyword = keyword.lower().strip()
    return [keyword in name.lower() for name in label_names]


def save_images(ds, indices: List[int], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(indices):
        image = ds[idx]["image"].convert("RGB")
        image.save(out_dir / f"{i:04d}.jpg", quality=95)


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load ImageNet validation split and collect class names.
    ds = load_dataset(args.dataset_path, split=args.split)
    if "label" not in ds.features or "image" not in ds.features:
        raise ValueError("Dataset must contain both 'label' and 'image' columns.")

    label_names = ds.features["label"].names
    is_pos_label = label_match(label_names, args.concept)
    if not any(is_pos_label):
        raise ValueError(f"No label names contain keyword '{args.concept}'.")

    need_pos = args.pos_train + args.pos_test
    need_neg = args.neg_train + args.neg_test
    pos_idx, neg_idx = [], []

    # Shuffle and scan indices until enough positive/negative samples are collected.
    all_idx = list(range(len(ds)))
    random.shuffle(all_idx)
    for idx in all_idx:
        label = ds[idx]["label"]
        if is_pos_label[label]:
            pos_idx.append(idx)
        else:
            neg_idx.append(idx)
        if len(pos_idx) >= need_pos and len(neg_idx) >= need_neg:
            break

    if len(pos_idx) < need_pos:
        raise ValueError(f"Insufficient positives: need {need_pos}, got {len(pos_idx)}")
    if len(neg_idx) < need_neg:
        raise ValueError(f"Insufficient negatives: need {need_neg}, got {len(neg_idx)}")

    concept_dir = Path(args.out_root) / args.concept
    pos_train_dir = concept_dir / "pos_train"
    pos_test_dir = concept_dir / "pos_test"
    neg_train_dir = concept_dir / "neg_train"
    neg_test_dir = concept_dir / "neg_test"

    # Save images to folders used by SAUCE pipeline.
    save_images(ds, pos_idx[: args.pos_train], pos_train_dir)
    save_images(ds, pos_idx[args.pos_train : args.pos_train + args.pos_test], pos_test_dir)
    save_images(ds, neg_idx[: args.neg_train], neg_train_dir)
    save_images(ds, neg_idx[args.neg_train : args.neg_train + args.neg_test], neg_test_dir)

    manifest = {
        "concept": args.concept,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "seed": args.seed,
        "pos_train": args.pos_train,
        "pos_test": args.pos_test,
        "neg_train": args.neg_train,
        "neg_test": args.neg_test,
        "output": {
            "pos_train": str(pos_train_dir.resolve()),
            "pos_test": str(pos_test_dir.resolve()),
            "neg_train": str(neg_train_dir.resolve()),
            "neg_test": str(neg_test_dir.resolve()),
        },
    }
    (concept_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Done: {concept_dir.resolve()}")
    print(f"POS train: {pos_train_dir.resolve()}")
    print(f"NEG train: {neg_train_dir.resolve()}")


if __name__ == "__main__":
    main()
