from pathlib import Path
import pandas as pd


def read_sorted_csv(path):
    df = pd.read_csv(path, parse_dates=["release_date"])
    return df.sort_values("release_date").reset_index(drop=True)


def resolve_fold_paths(args, need_val=False, default_eval_name="test.csv"):
    split_dir = Path(args.split_dir) if args.split_dir else Path(args.data_folder)

    if args.fold is not None:
        train_csv = split_dir / f"fold_{args.fold}_train.csv"
        val_csv = split_dir / f"fold_{args.fold}_validation.csv"
        eval_csv = val_csv
    else:
        train_csv = Path(args.train_csv) if args.train_csv else Path(args.data_folder) / "train.csv"
        val_csv = Path(args.val_csv) if args.val_csv else None
        eval_csv = Path(args.eval_csv) if args.eval_csv else Path(args.data_folder) / default_eval_name

    if need_val and val_csv is None:
        raise ValueError("A validation CSV is required: use --fold or provide --val_csv.")

    for path in [train_csv, val_csv, eval_csv]:
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"Could not find split file: {path}")

    return Path(train_csv), (Path(val_csv) if val_csv else None), (Path(eval_csv) if eval_csv else None)


def fold_suffix(args):
    return f"fold{args.fold}" if args.fold is not None else "default"