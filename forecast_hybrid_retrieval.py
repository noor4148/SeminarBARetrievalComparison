import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from models.GTM_hybrid_retrieval import HybridRetrievalGTM
from utils.data_multitrends import ZeroShotDataset


def compute_forecast_metrics_np(y_true, y_pred, erp_epsilon=0.1):
    abs_err = np.abs(y_true - y_pred)
    mae = abs_err.mean()
    wape = 100.0 * abs_err.sum() / max(y_true.sum(), 1e-12)
    mae_per_series = np.maximum(abs_err.mean(axis=1), 1e-12)
    signed_error_per_series = (y_true - y_pred).sum(axis=1)
    ts = (signed_error_per_series / mae_per_series).mean()
    erp = (abs_err >= erp_epsilon).sum(axis=1).mean()
    return round(wape, 3), round(mae, 3), round(ts, 3), round(erp, 3)

def print_error_metrics(y_true, y_pred, rescaled_y_true, rescaled_y_pred, erp_epsilon=0.1):
    wape, mae, ts, erp = compute_forecast_metrics_np(y_true, y_pred, erp_epsilon=erp_epsilon)
    rwape, rmae, rts, rerp = compute_forecast_metrics_np(rescaled_y_true, rescaled_y_pred,
                                                         erp_epsilon=erp_epsilon * 1065.0)
    print("Normalized:", {"WAPE": wape, "MAE": mae, "TS": ts, "ERP": erp})
    print("Rescaled:", {"WAPE": rwape, "MAE": rmae, "TS": rts, "ERP": rerp})

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


def build_loader_with_retrieval(df, img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len, retrieval_memory):
    base_dataset = ZeroShotDataset(df.copy(), img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len).preprocess_data()
    retrieval_curves, retrieval_available = build_retrieval_tensors_for_dataframe(df, retrieval_memory)
    dataset = RetrievalAugmentedDataset(base_dataset, retrieval_curves, retrieval_available)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def run(args):
    print(args)
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(args.seed)

    test_df = pd.read_csv(Path(args.data_folder) / "test.csv", parse_dates=["release_date"])
    item_codes = test_df["external_code"].values

    cat_dict = torch.load(Path(args.data_folder) / "category_labels.pt", weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / "color_labels.pt", weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / "fabric_labels.pt", weights_only=False)
    gtrends = pd.read_csv(Path(args.data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)
    retrieval_memory = torch.load(args.retrieval_memory_path, map_location="cpu", weights_only=False)

    test_loader = build_loader_with_retrieval(
        test_df,
        Path(args.data_folder) / "images",
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len,
        retrieval_memory,
    )

    model = HybridRetrievalGTM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.model_output_dim,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_text=args.use_text,
        use_img=args.use_img,
        trend_len=args.trend_len,
        num_trends=args.num_trends,
        gpu_num=args.gpu_num,
        retrieval_seq_len=args.model_output_dim,
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive,
    )

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    gt, forecasts, attns = [], [], []
    for test_data in tqdm(test_loader, total=len(test_loader), ascii=True):
        with torch.no_grad():
            test_data = [tensor.to(device) for tensor in test_data]
            item_sales, category, color, fabric, temporal_features, gtrends_batch, images, analog_curve, analog_available = test_data
            y_pred, att = model(
                category,
                color,
                fabric,
                temporal_features,
                gtrends_batch,
                images,
                analog_curve=analog_curve,
                analog_available=analog_available,
            )
            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
            y_true_np = item_sales.detach().cpu().numpy().reshape(-1)
            forecasts.append(y_pred_np[: args.eval_horizon])
            gt.append(y_true_np[: args.eval_horizon])
            attns.append(att.detach().cpu().numpy())

    attns = np.stack(attns)
    forecasts = np.array(forecasts)
    gt = np.array(gt)

    scale = float(np.load(Path(args.data_folder) / "normalization_scale.npy"))
    rescale_vals = np.full(args.eval_horizon, scale, dtype=np.float32)
    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals
    print_error_metrics(gt, forecasts, rescaled_gt, rescaled_forecasts)

    model_savename = f"{args.run_name}_model{args.model_output_dim}_eval{args.eval_horizon}"
    Path("results").mkdir(parents=True, exist_ok=True)
    torch.save(
        {"results": rescaled_forecasts, "gts": rescaled_gt, "codes": item_codes.tolist(), "attns": attns},
        Path("results") / f"{model_savename}.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast with the hybrid retrieval-augmented GTM model.")
    parser.add_argument("--data_folder", type=str, default="dataset/")
    parser.add_argument("--retrieval_memory_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=21)

    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--model_output_dim", type=int, default=12)
    parser.add_argument("--eval_horizon", type=int, default=6)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="Run1")

    args = parser.parse_args()
    if args.eval_horizon > args.model_output_dim:
        raise ValueError(
            f"eval_horizon ({args.eval_horizon}) cannot be bigger than model_output_dim ({args.model_output_dim})."
        )
    run(args)
