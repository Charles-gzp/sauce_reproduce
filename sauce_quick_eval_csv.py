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
    parser = argparse.ArgumentParser(description="Quick sanity-check: evaluate SAUCE with captions or Yes/No prompts.")
    parser.add_argument("--sae-path", type=str, required=True, help="Path to trained SAE checkpoint.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of evaluation images.")
    parser.add_argument("--out-csv", type=str, default="quick_eval.csv", help="Output CSV path.")
    parser.add_argument("--num-images", type=int, default=50, help="Number of images to evaluate.")
    parser.add_argument("--prompt", type=str, default="Please describe this figure.", help="Prompt used for evaluation.")
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["gen", "disc"],
        default="gen",
        help="gen=caption generation, disc=Yes/No discriminative prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for generation.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (use 1 to avoid OOM).")
    parser.add_argument("--feature-indices-path", type=str, default=None, help="Optional selected feature file (.pt).")
    parser.add_argument("--gamma", type=float, default=-0.5, help="Intervention scale if feature indices are provided.")
    parser.add_argument("--keyword", type=str, default="dog", help="Keyword for gen-mode contains check.")
    parser.add_argument("--yes-token", type=str, default="Yes", help="Yes token text for discriminative scoring.")
    parser.add_argument("--no-token", type=str, default="No", help="No token text for discriminative scoring.")
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    images.sort()
    return images


def contains_keyword(text: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword.lower())}s?\b", text.lower()) is not None


@torch.no_grad()
def generate_captions(
    model: HookedVisionTransformer,
    image_paths: List[Path],
    prompt: str,
    max_new_tokens: int,
    batch_size: int,
):
    captions = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        text_inputs = [prompt] * len(images)
        inputs = model.prepare_inputs(images=images, text=text_inputs)

        generated_ids = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        decoded = model.processor.batch_decode(generated_ids, skip_special_tokens=True)
        captions.extend([t.strip() for t in decoded])
    return captions


def pick_token_id(tokenizer, token_text: str):
    candidates = [token_text]
    if not token_text.startswith(" "):
        candidates.insert(0, f" {token_text}")
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], cand, True
    ids = tokenizer.encode(candidates[0], add_special_tokens=False)
    return ids[0], candidates[0], False


@torch.no_grad()
def score_yes_no(
    model: HookedVisionTransformer,
    image_paths: List[Path],
    prompt: str,
    batch_size: int,
    yes_token_id: int,
    no_token_id: int,
):
    yes_logits, no_logits, preds = [], [], []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        text_inputs = [prompt] * len(images)
        inputs = model.prepare_inputs(images=images, text=text_inputs)
        outputs = model.model(**inputs)
        logits = outputs.logits[:, -1, :]
        batch_yes = logits[:, yes_token_id].detach().cpu().tolist()
        batch_no = logits[:, no_token_id].detach().cpu().tolist()
        yes_logits.extend(batch_yes)
        no_logits.extend(batch_no)
        preds.extend([y > n for y, n in zip(batch_yes, batch_no)])
    return yes_logits, no_logits, preds


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

    baseline_captions = None
    baseline_yes = baseline_no = baseline_preds = None

    if args.eval_mode == "gen":
        baseline_captions = generate_captions(
            model=model,
            image_paths=image_paths,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    else:
        tokenizer = getattr(model.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Tokenizer not found; cannot run discriminative scoring.")
        yes_id, yes_text, yes_single = pick_token_id(tokenizer, args.yes_token)
        no_id, no_text, no_single = pick_token_id(tokenizer, args.no_token)
        if not (yes_single and no_single):
            print(f"Warning: tokenization is multi-token for '{yes_text}' or '{no_text}', using first token only.")
        baseline_yes, baseline_no, baseline_preds = score_yes_no(
            model=model,
            image_paths=image_paths,
            prompt=args.prompt,
            batch_size=args.batch_size,
            yes_token_id=yes_id,
            no_token_id=no_id,
        )

    edited_captions = None
    edited_yes = edited_no = edited_preds = None

    if args.feature_indices_path is not None:
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

        if args.eval_mode == "gen":
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
        else:
            with model.hooks(hooks):
                edited_yes, edited_no, edited_preds = score_yes_no(
                    model=model,
                    image_paths=image_paths,
                    prompt=args.prompt,
                    batch_size=args.batch_size,
                    yes_token_id=yes_id,
                    no_token_id=no_id,
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_path"]
    if args.eval_mode == "gen":
        fieldnames += ["baseline_caption", "baseline_contains_dog"]
        if edited_captions is not None:
            fieldnames += ["edited_caption", "edited_contains_dog"]
    else:
        fieldnames += [
            "baseline_yes_logit",
            "baseline_no_logit",
            "baseline_margin",
            "baseline_pred_yes",
        ]
        if edited_preds is not None:
            fieldnames += [
                "edited_yes_logit",
                "edited_no_logit",
                "edited_margin",
                "edited_pred_yes",
            ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, path in enumerate(image_paths):
            row = {"image_path": str(path)}
            if args.eval_mode == "gen":
                row["baseline_caption"] = baseline_captions[i]
                row["baseline_contains_dog"] = contains_keyword(baseline_captions[i], args.keyword)
                if edited_captions is not None:
                    row["edited_caption"] = edited_captions[i]
                    row["edited_contains_dog"] = contains_keyword(edited_captions[i], args.keyword)
            else:
                row["baseline_yes_logit"] = baseline_yes[i]
                row["baseline_no_logit"] = baseline_no[i]
                row["baseline_margin"] = baseline_yes[i] - baseline_no[i]
                row["baseline_pred_yes"] = bool(baseline_preds[i])
                if edited_preds is not None:
                    row["edited_yes_logit"] = edited_yes[i]
                    row["edited_no_logit"] = edited_no[i]
                    row["edited_margin"] = edited_yes[i] - edited_no[i]
                    row["edited_pred_yes"] = bool(edited_preds[i])
            writer.writerow(row)

    print(f"Saved CSV: {out_csv.resolve()}")
    if args.eval_mode == "gen":
        baseline_hits = sum(contains_keyword(t, args.keyword) for t in baseline_captions)
        print(f"Baseline contains '{args.keyword}': {baseline_hits}/{len(baseline_captions)}")
        if edited_captions is not None:
            edited_hits = sum(contains_keyword(t, args.keyword) for t in edited_captions)
            print(f"Edited contains '{args.keyword}': {edited_hits}/{len(edited_captions)}")
    else:
        baseline_yes_count = sum(bool(x) for x in baseline_preds)
        print(f"Baseline predicted Yes: {baseline_yes_count}/{len(baseline_preds)}")
        if edited_preds is not None:
            edited_yes_count = sum(bool(x) for x in edited_preds)
            print(f"Edited predicted Yes: {edited_yes_count}/{len(edited_preds)}")
            mean_delta_yes = sum(ey - by for ey, by in zip(edited_yes, baseline_yes)) / len(edited_yes)
            mean_delta_no = sum(en - bn for en, bn in zip(edited_no, baseline_no)) / len(edited_no)
            mean_delta_margin = sum((ey - en) - (by - bn) for ey, en, by, bn in zip(edited_yes, edited_no, baseline_yes, baseline_no)) / len(edited_yes)
            print(f"Mean delta yes_logit: {mean_delta_yes:.6f}")
            print(f"Mean delta no_logit: {mean_delta_no:.6f}")
            print(f"Mean delta margin (yes-no): {mean_delta_margin:.6f}")


if __name__ == "__main__":
    main()
