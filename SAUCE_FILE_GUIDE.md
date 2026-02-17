# SAUCE Reproduction File Guide

This document maps the SAUCE pipeline to concrete files in this repo.

## 1) SAE training (A1 + B2/B3/B4/B5)

- `training_sae_on_ViT.py`
  - Main training entry script.
  - Current default config is set to SAUCE-like setup:
    - penultimate layer hook (`block_layer = -2`)
    - last image token (`sae_target_token = "last"`)
    - prompt `"Please describe this figure"`
    - SAE loss uses `l2` + L1
    - expansion factor 64, alpha 0.00008, lr 0.0004, batch 1024
- `sae_training/config.py`
  - `ViTSAERunnerConfig` includes SAUCE-critical switches:
    - `sae_target_token`
    - `image_caption_prompt`
    - `reconstruction_loss_type`
- `sae_training/vit_activations_store.py`
  - Collects activations from the configured token and prompt.
- `sae_training/sparse_autoencoder.py`
  - SAE forward/loss implementation.

## 2) Correlation-based feature selection (A2)

- `sauce_build_dataset.py`
  - Build SAUCE-style per-concept train/test split from local raw image pools.
  - Default split size matches paper setting: train 200 / test 50.
- `sauce_concept_spec.example.json`
  - Example concept spec for dataset builder input.
- `sauce_export_imagenet_concept.py`
  - Export one concept's positive/negative images directly from ImageNet validation split.
- `sauce_prepare_concept_activations.py`
  - Build positive/negative token activation tensors from image folders.
- `sauce_feature_selection.py`
  - Computes feature score:
  - `S(i) = mu_pos(i)/(sum_j mu_pos(j)+delta) - mu_neg(i)/(sum_j mu_neg(j)+delta)`
  - Saves top-k feature indices.
- `sae_training/sauce.py`
  - Core scoring implementation (`select_features_by_pos_neg_correlation`).

## 3) Feature intervention (A3)

- `sauce_intervention.py`
  - Loads selected features and applies intervention with gamma (default `-0.5`) during forward hooks.
- `sauce_pipeline.py`
  - Single command pipeline for:
  - activation extraction -> top-k feature selection -> optional intervention smoke test.
- `run_sauce_imagenet_concept.sh`
  - One-command wrapper: ImageNet concept export -> SAUCE pipeline.
- `sae_training/sauce.py`
  - Core intervention hook builder (`build_sauce_intervention_hook`).

## 4) Evaluation helpers (D)

- `sae_training/sauce.py`
  - score helpers: UAg/UAd/retain
- `sauce_eval_scores.py`
  - CLI utility to compute UAg/UAd/IRA/CRA from json/jsonl logs.
- `sauce_feature_stats.py`
  - 计算 SAE 特征统计量（FS / FMAV）。

## 5) Current scope status

- Implemented in code:
  - A1/A2/A3
  - B2/B4/B5 (and B3 prompt behavior)
  - D metric computations (utility level)
- Not fully automated yet:
  - B1 exact VLM variants from paper (current executable training path is CLIP-based ViT path)
  - C raw web-image collection pipeline (Google Image Search API ingestion is not included here)
