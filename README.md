# Supplementary README, Fold-Aware Split for GTM / Hybrid Retrieval

This repository is a small extension of the previous version.
It does not introduce a completely new modeling pipeline. Instead, it mainly adds more flexible dataset split handling, so the same GTM and hybrid-retrieval code can be used with:

- the original default split setup,
- explicitly provided train/validation CSV files,
- or predefined fold files for cross-validation-style experiments.

## What changed compared to the previous version?

### 1. New fold-aware split utilities
A new helper file, `fold_utils.py`, centralizes how train/validation/evaluation files are resolved.

It adds support for:

- `--fold`
- `--split_dir`
- `--train_csv`
- `--val_csv`
- `--eval_csv`

When `--fold` is set, the code expects files like:

- `fold_<k>_train.csv`
- `fold_<k>_validation.csv`

inside `--split_dir` (or inside `data_folder` if `split_dir` is not given).

If no fold is provided, the code falls back to the old behavior and uses the default CSV files or user-supplied paths. In `train.py` and `train_hybrid_retrieval.py`, it still supports the old automatic 85/15 date-based split when no validation CSV is given.

### 2. Baseline training now supports fold-based experiments
`train.py` was updated so the baseline GTM pipeline no longer depends only on `dataset/train.csv` plus an internal 85/15 split.

You can now:

- run a specific fold with `--fold`,
- point to an external split directory with `--split_dir`,
- or manually provide `--train_csv` and `--val_csv`.

### 3. Hybrid training received the same split logic
`train_hybrid_retrieval.py` now mirrors the same fold-aware behavior.

So instead of always taking `train.csv` and internally cutting off the last 15% as validation, the script can now:

- train on a predefined fold,
- validate on a matching fold validation file,
- or still fall back to the old 85/15 logic if no validation split is supplied.

### 4. Retrieval-memory building now uses explicit train/val split files
`build_hybrid_retrieval_memory.py` changed more substantially in terms of split handling.

In the earlier setup, retrieval memory was built from a single `train.csv` that was internally divided into subtrain and validation. In this version, memory construction explicitly resolves:

- a training split,
- a validation split,
- and optionally a test split.

This matters because retrieval candidates are still restricted to the subtrain rows, while the combined metadata can include validation and test rows as query items. That makes the retrieval-memory generation consistent with fold-based experiments.

## Typical usage differences

### Previous style

Baseline training:

```bash
python train.py \
  --data_folder dataset/ \
  --model_type GTM \
  --wandb_run Run1
```

Hybrid training:

```bash
python train_hybrid_retrieval.py \
  --data_folder dataset/ \
  --retrieval_memory_path artifacts/retrieval_memory.pth \
  --run_name Run1
```

This uses the default dataset files and may fall back to the internal validation split.

### New fold-based style

Baseline training on fold 1:

```bash
python train.py \
  --data_folder dataset/ \
  --split_dir splits/ \
  --fold 1 \
  --model_type GTM \
  --wandb_run fold1_run
```

Hybrid training on fold 1:

```bash
python train_hybrid_retrieval.py \
  --data_folder dataset/ \
  --split_dir splits/ \
  --fold 1 \
  --retrieval_memory_path artifacts/retrieval_memory_fold1.pth \
  --run_name fold1_run
```

Build retrieval memory for fold 1:

```bash
python build_hybrid_retrieval_memory.py \
  --data_folder dataset/ \
  --split_dir splits/ \
  --fold 1 \
  --checkpoint_path path/to/baseline_fold1.ckpt \
  --output_path artifacts/retrieval_memory_fold1.pth
```

## Important note

For fold-based experiments, keep the split configuration consistent across all steps:

- baseline training,
- retrieval-memory building,
- hybrid training,
- and final forecasting.

Otherwise, the retrieval lookup may no longer align with the intended data partition.

## Files most relevant to this update

- `fold_utils.py` — new split-resolution helper
- `train.py` — baseline training with fold/manual split support
- `train_hybrid_retrieval.py` — hybrid training with fold/manual split support
- `build_hybrid_retrieval_memory.py` — retrieval-memory building with explicit split resolution

