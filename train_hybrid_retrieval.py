import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

from models.GTM_hybrid_retrieval import HybridRetrievalGTM
from utils.data_multitrends import ZeroShotDataset
from utils.fold_utils import resolve_fold_paths, read_sorted_csv, fold_suffix

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RetrievalAugmentedDataset(Dataset):
    def __init__(self, base_dataset, retrieval_curves, retrieval_available):
        self.base_dataset = base_dataset
        self.retrieval_curves = retrieval_curves
        self.retrieval_available = retrieval_available
        if len(self.base_dataset) != len(self.retrieval_curves):
            raise ValueError("Length mismatch between base dataset and retrieval curves.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        return (*item, self.retrieval_curves[idx], self.retrieval_available[idx])


def build_retrieval_tensors_for_dataframe(df: pd.DataFrame, retrieval_memory: dict):
    metadata = retrieval_memory["metadata"]
    retrieval_curve = retrieval_memory["retrieval_curve"]
    retrieval_available = retrieval_memory["retrieval_available"]

    lookup = metadata[["external_code"]].copy()
    lookup["row_idx"] = np.arange(len(lookup))
    merged = df[["external_code"]].merge(lookup, on="external_code", how="left", validate="one_to_one")
    if merged["row_idx"].isna().any():
        missing = merged.loc[merged["row_idx"].isna(), "external_code"].tolist()
        raise ValueError(f"Missing retrieval rows for {len(missing)} products. Examples: {missing[:10]}")

    row_idx = torch.tensor(merged["row_idx"].astype(int).values, dtype=torch.long)
    return retrieval_curve[row_idx], retrieval_available[row_idx]


def build_loader_with_retrieval(df, img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len, retrieval_memory, batch_size, train):
    base_dataset = ZeroShotDataset(df.copy(), img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len).preprocess_data()
    retrieval_curves, retrieval_available = build_retrieval_tensors_for_dataframe(df, retrieval_memory)
    dataset = RetrievalAugmentedDataset(base_dataset, retrieval_curves, retrieval_available)
    return DataLoader(
        dataset,
        batch_size=batch_size if train else 1,
        shuffle=train,
        num_workers=2,
    )


def run(args):
    print(args)
    pl.seed_everything(args.seed)

    train_csv, val_csv, _ = resolve_fold_paths(args, need_val=False)

    train_df = read_sorted_csv(train_csv)

    if val_csv is not None:
        val_df = read_sorted_csv(val_csv)
    else:
        val_size = max(1, int(args.val_frac * len(train_df)))
        val_df = train_df.iloc[-val_size:].copy().reset_index(drop=True)
        train_df = train_df.iloc[:-val_size].copy().reset_index(drop=True)

    cat_dict = torch.load(Path(args.data_folder) / "category_labels.pt", weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / "color_labels.pt", weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / "fabric_labels.pt", weights_only=False)
    gtrends = pd.read_csv(Path(args.data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)
    retrieval_memory = torch.load(args.retrieval_memory_path, map_location="cpu", weights_only=False)

    img_root = Path(args.data_folder) / "images"

    train_loader = build_loader_with_retrieval(
        train_df, img_root, gtrends, cat_dict, col_dict, fab_dict,
        args.trend_len, retrieval_memory, args.batch_size, train=True,
    )

    val_loader = build_loader_with_retrieval(
        val_df, img_root, gtrends, cat_dict, col_dict, fab_dict,
        args.trend_len, retrieval_memory, batch_size=1, train=False,
    )
    model_savename = f"GTM_hybrid_{args.run_name}_{fold_suffix(args)}"

    model = HybridRetrievalGTM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_text=args.use_text,
        use_img=args.use_img,
        trend_len=args.trend_len,
        num_trends=args.num_trends,
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive,
        gpu_num=args.gpu_num,
        retrieval_seq_len=args.output_dim,
    )

    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(args.log_dir) / "GTM_hybrid",
        filename=model_savename + "---{epoch}---" + dt_string,
        monitor="val_wape",
        mode="min",
        save_top_k=1,
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_wape",
        min_delta=0.0,
        patience=5,
        verbose=True,
        mode="min",
    )
    tb_logger = pl_loggers.TensorBoardLogger(args.log_dir + "/", name=model_savename)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[args.gpu_num] if torch.cuda.is_available() else 1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=5,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the hybrid retrieval-augmented GTM model.")
    parser.add_argument("--data_folder", type=str, default="dataset/")
    parser.add_argument("--retrieval_memory_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--val_frac", type=float, default=0.15)

    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="Run1")

    # to decide which fold to use
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split_dir", type=str, default="")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--eval_csv", type=str, default="")

    run(parser.parse_args())
