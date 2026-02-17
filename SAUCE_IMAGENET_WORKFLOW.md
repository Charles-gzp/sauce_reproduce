# SAUCE ImageNet Workflow

This workflow uses two scripts:

- `sauce_export_imagenet_concept.py`
- `run_sauce_imagenet_concept.sh`

## 1) `sauce_export_imagenet_concept.py`

What it does:
- Samples one concept's positive/negative images from ImageNet validation split.
- Writes folders:
  - `<out-root>/<concept>/pos_train`
  - `<out-root>/<concept>/pos_test`
  - `<out-root>/<concept>/neg_train`
  - `<out-root>/<concept>/neg_test`
- Saves `manifest.json` for reproducibility.

Example:
```bash
python sauce_export_imagenet_concept.py \
  --concept dog \
  --dataset-path evanarlian/imagenet_1k_resized_256 \
  --split validation \
  --pos-train 200 --pos-test 50 \
  --neg-train 200 --neg-test 50 \
  --out-root concepts
```

## 2) `run_sauce_imagenet_concept.sh`

What it does:
- Runs ImageNet concept export first.
- Runs SAUCE pipeline next (activation extraction + feature selection + optional intervention smoke test).

Required variable:
- `SAE_PATH`: trained SAE checkpoint path.

Optional variables:
- `CONCEPT` (default `dog`)
- `DATASET_PATH` (default `evanarlian/imagenet_1k_resized_256`)
- `OUT_ROOT` (default `concepts`)
- `PIPE_OUT_ROOT` (default `sauce_outputs`)
- `TOP_K` (default `128`)

Example:
```bash
SAE_PATH=/home/gzp/data/sauce_reproduce/checkpoints/xxx/final_*.pt \
CONCEPT=dog \
bash run_sauce_imagenet_concept.sh
```
