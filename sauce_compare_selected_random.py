import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from sae_training.hooked_vit import Hook, HookedVisionTransformer
from sae_training.sauce import build_sauce_intervention_hook, get_token_index
from sae_training.sparse_autoencoder import SparseAutoencoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare selected features vs random features on discriminative Yes/No evaluation."
    )
    parser.add_argument("--sae-path", type=str, required=True, help="Path to SAE checkpoint.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of evaluation images.")
    parser.add_argument(
        "--selected-feature-indices-path",
        type=str,
        required=True,
        help="Path to selected_feature_indices.pt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Does this image contain a dog? Please answer Yes or No.",
        help="Discriminative evaluation prompt.",
    )
    parser.add_argument("--num-images", type=int, default=50, help="Number of images to evaluate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--gamma", type=float, default=-0.5, help="Intervention gamma.")
    parser.add_argument("--yes-token", type=str, default="Yes", help="Yes token text.")
    parser.add_argument("--no-token", type=str, default="No", help="No token text.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random feature indices.")
    parser.add_argument(
        "--random-indices-out",
        type=str,
        default=None,
        help="Optional output path to save random feature indices.",
    )
    parser.add_argument("--out-csv", type=str, default=None, help="Optional per-image output csv.")
    parser.add_argument("--out-json", type=str, default=None, help="Optional summary output json.")
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    images.sort()
    return images


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
    hooks: Optional[List[Hook]] = None,
):
    yes_logits, no_logits, preds = [], [], []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        text_inputs = [prompt] * len(images)
        inputs = model.prepare_inputs(images=images, text=text_inputs)
        if hooks is None:
            outputs = model.model(**inputs)
        else:
            with model.hooks(hooks):
                outputs = model.model(**inputs)
        logits = outputs.logits[:, -1, :]
        batch_yes = logits[:, yes_token_id].detach().cpu().tolist()
        batch_no = logits[:, no_token_id].detach().cpu().tolist()
        yes_logits.extend(batch_yes)
        no_logits.extend(batch_no)
        preds.extend([y > n for y, n in zip(batch_yes, batch_no)])
    return yes_logits, no_logits, preds


def summarize_case(name: str, base_yes, base_no, edit_yes, edit_no, edit_pred):
    yes_count = int(sum(bool(x) for x in edit_pred))
    mean_delta_yes = sum(e - b for e, b in zip(edit_yes, base_yes)) / len(edit_yes)
    mean_delta_no = sum(e - b for e, b in zip(edit_no, base_no)) / len(edit_no)
    mean_delta_margin = sum((ey - en) - (by - bn) for ey, en, by, bn in zip(edit_yes, edit_no, base_yes, base_no)) / len(edit_yes)
    print(f"[{name}] predicted Yes: {yes_count}/{len(edit_pred)}")
    print(f"[{name}] mean delta yes_logit: {mean_delta_yes:.6f}")
    print(f"[{name}] mean delta no_logit: {mean_delta_no:.6f}")
    print(f"[{name}] mean delta margin (yes-no): {mean_delta_margin:.6f}")
    return {
        "yes_count": yes_count,
        "mean_delta_yes_logit": float(mean_delta_yes),
        "mean_delta_no_logit": float(mean_delta_no),
        "mean_delta_margin": float(mean_delta_margin),
    }


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

    image_paths = list_images(Path(args.image_dir))[: args.num_images]
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {args.image_dir}")

    tokenizer = getattr(model.processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Tokenizer not found; cannot run discriminative scoring.")
    yes_id, yes_text, yes_single = pick_token_id(tokenizer, args.yes_token)
    no_id, no_text, no_single = pick_token_id(tokenizer, args.no_token)
    if not (yes_single and no_single):
        print(f"Warning: tokenization is multi-token for '{yes_text}' or '{no_text}', using first token only.")

    selected_indices = torch.load(args.selected_feature_indices_path, map_location="cpu").long()
    k = int(selected_indices.numel())
    if k == 0:
        raise ValueError("Selected feature indices are empty.")

    # 同 k 随机采样，作为对照组。
    g = torch.Generator().manual_seed(args.seed)
    random_indices = torch.randperm(sae.d_sae, generator=g)[:k].long()
    random_out = args.random_indices_out
    if random_out is None:
        random_out = str(Path(args.selected_feature_indices_path).with_name(f"random_feature_indices_k{k}_seed{args.seed}.pt"))
    random_out_path = Path(random_out)
    random_out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(random_indices, random_out_path)
    print(f"Saved random indices: {random_out_path}")

    token_index = get_token_index(
        class_token=sae.cfg.class_token,
        sae_target_token=sae.cfg.sae_target_token,
    )
    selected_hook = build_sauce_intervention_hook(
        sparse_autoencoder=sae,
        selected_feature_indices=[int(i) for i in selected_indices.tolist()],
        gamma=args.gamma,
        token_index=token_index,
    )
    random_hook = build_sauce_intervention_hook(
        sparse_autoencoder=sae,
        selected_feature_indices=[int(i) for i in random_indices.tolist()],
        gamma=args.gamma,
        token_index=token_index,
    )
    selected_hooks = [Hook(sae.cfg.block_layer, sae.cfg.module_name, selected_hook, return_module_output=False)]
    random_hooks = [Hook(sae.cfg.block_layer, sae.cfg.module_name, random_hook, return_module_output=False)]

    base_yes, base_no, base_pred = score_yes_no(
        model=model,
        image_paths=image_paths,
        prompt=args.prompt,
        batch_size=args.batch_size,
        yes_token_id=yes_id,
        no_token_id=no_id,
        hooks=None,
    )
    print(f"[baseline] predicted Yes: {sum(bool(x) for x in base_pred)}/{len(base_pred)}")

    sel_yes, sel_no, sel_pred = score_yes_no(
        model=model,
        image_paths=image_paths,
        prompt=args.prompt,
        batch_size=args.batch_size,
        yes_token_id=yes_id,
        no_token_id=no_id,
        hooks=selected_hooks,
    )
    rnd_yes, rnd_no, rnd_pred = score_yes_no(
        model=model,
        image_paths=image_paths,
        prompt=args.prompt,
        batch_size=args.batch_size,
        yes_token_id=yes_id,
        no_token_id=no_id,
        hooks=random_hooks,
    )

    sel_summary = summarize_case("selected", base_yes, base_no, sel_yes, sel_no, sel_pred)
    rnd_summary = summarize_case("random", base_yes, base_no, rnd_yes, rnd_no, rnd_pred)

    advantage = sel_summary["mean_delta_margin"] - rnd_summary["mean_delta_margin"]
    print(f"[compare] selected_minus_random_mean_delta_margin: {advantage:.6f} (more negative is better)")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "image_path",
                    "baseline_yes_logit",
                    "baseline_no_logit",
                    "baseline_margin",
                    "selected_yes_logit",
                    "selected_no_logit",
                    "selected_margin",
                    "random_yes_logit",
                    "random_no_logit",
                    "random_margin",
                    "baseline_pred_yes",
                    "selected_pred_yes",
                    "random_pred_yes",
                ]
            )
            for i, p in enumerate(image_paths):
                writer.writerow(
                    [
                        str(p),
                        base_yes[i],
                        base_no[i],
                        base_yes[i] - base_no[i],
                        sel_yes[i],
                        sel_no[i],
                        sel_yes[i] - sel_no[i],
                        rnd_yes[i],
                        rnd_no[i],
                        rnd_yes[i] - rnd_no[i],
                        bool(base_pred[i]),
                        bool(sel_pred[i]),
                        bool(rnd_pred[i]),
                    ]
                )
        print(f"Saved CSV: {out_csv}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "sae_path": args.sae_path,
            "image_dir": args.image_dir,
            "num_images": len(image_paths),
            "prompt": args.prompt,
            "gamma": args.gamma,
            "k": k,
            "baseline_yes_count": int(sum(bool(x) for x in base_pred)),
            "selected": sel_summary,
            "random": rnd_summary,
            "selected_minus_random_mean_delta_margin": float(advantage),
            "selected_feature_indices_path": args.selected_feature_indices_path,
            "random_feature_indices_path": str(random_out_path),
        }
        out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
