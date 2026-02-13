import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Build SAUCE-style concept datasets from local image pools.")
    parser.add_argument("--concept-spec", type=str, required=True, help="JSON file defining concepts and source folders.")
    parser.add_argument("--out-dir", type=str, default="sauce_datasets", help="Output directory for train/test splits.")
    parser.add_argument("--train-per-concept", type=int, default=200, help="Train images per concept.")
    parser.add_argument("--test-per-concept", type=int, default=50, help="Test images per concept.")
    parser.add_argument("--copy-files", action="store_true", help="Copy files instead of symlink (Windows-safe default is symlink off).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    # Centralized image extension filter for reproducible sampling.
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def ensure_link_or_copy(src: Path, dst: Path, copy_files: bool):
    # Keep data storage efficient by default (symlink); allow copy fallback.
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    random.seed(args.seed)

    spec_path = Path(args.concept_spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    concepts: List[Dict] = spec["concepts"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "train_per_concept": args.train_per_concept,
        "test_per_concept": args.test_per_concept,
        "seed": args.seed,
        "concepts": [],
    }

    for concept in concepts:
        # Required fields: name/domain/source_dir.
        name = concept["name"]
        domain = concept["domain"]
        source_dir = Path(concept["source_dir"])
        images = list_images(source_dir)
        needed = args.train_per_concept + args.test_per_concept
        if len(images) < needed:
            raise ValueError(
                f"Concept '{name}' has only {len(images)} images in {source_dir}, need at least {needed}."
            )

        # Deterministic subsampling and split.
        random.shuffle(images)
        selected = images[:needed]
        train_images = selected[: args.train_per_concept]
        test_images = selected[args.train_per_concept :]

        train_dir = out_dir / domain / name / "train"
        test_dir = out_dir / domain / name / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(train_images):
            dst = train_dir / f"{idx:04d}{src.suffix.lower()}"
            ensure_link_or_copy(src, dst, copy_files=args.copy_files)
        for idx, src in enumerate(test_images):
            dst = test_dir / f"{idx:04d}{src.suffix.lower()}"
            ensure_link_or_copy(src, dst, copy_files=args.copy_files)

        manifest["concepts"].append(
            {
                "name": name,
                "domain": domain,
                "source_dir": str(source_dir),
                "train_count": len(train_images),
                "test_count": len(test_images),
            }
        )
        print(f"Built concept split: {domain}/{name} (train={len(train_images)}, test={len(test_images)})")

    # Save exact split metadata for reproducibility.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {(out_dir / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
