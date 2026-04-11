import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.GTM import GTM
from utils.data_multitrends import ZeroShotDataset


def load_baseline_model(checkpoint_path, args, cat_dict, col_dict, fab_dict):
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    model = GTM(
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
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def prepare_metadata(train_csv: Path, test_csv: Path | None, val_frac: float):
    train_df = pd.read_csv(train_csv, parse_dates=["release_date"])
    train_df = train_df.sort_values("release_date").reset_index(drop=True)

    val_size = max(1, int(val_frac * len(train_df)))
    subtrain_df = train_df.iloc[:-val_size].copy().reset_index(drop=True)
    val_df = train_df.iloc[-val_size:].copy().reset_index(drop=True)
    subtrain_df["split"] = "subtrain"
    val_df["split"] = "val"

    dfs = [subtrain_df, val_df]
    if test_csv is not None:
        test_df = pd.read_csv(test_csv, parse_dates=["release_date"])
        test_df = test_df.sort_values("release_date").reset_index(drop=True)
        test_df["split"] = "test"
        dfs.append(test_df)

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined


def extract_embeddings(metadata_df, args, model, img_root, gtrends, cat_dict, col_dict, fab_dict):
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    dataset_df = metadata_df.drop(columns=["split"]).copy()
    dataset = ZeroShotDataset(dataset_df, img_root, gtrends, cat_dict, col_dict, fab_dict, args.trend_len)
    loader = dataset.get_loader(batch_size=1, train=False)

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting multimodal embeddings", ascii=True):
            _, category, color, fabric, _, _, images = batch
            category = category.to(device)
            color = color.to(device)
            fabric = fabric.to(device)
            images = images.to(device)
            x_i = model.encode_multimodal_embedding(category, color, fabric, images)
            embeddings.append(x_i.squeeze(0).cpu())
    return torch.stack(embeddings, dim=0)


def compute_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = embeddings.float()
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-12)
    normalized = embeddings / norms
    return normalized @ normalized.T


def build_admissibility_mask(metadata: pd.DataFrame, horizon_weeks: int) -> torch.Tensor:
    release_dates = pd.to_datetime(metadata["release_date"])
    d_i = release_dates.values[:, None]
    d_j = release_dates.values[None, :]
    horizon = pd.to_timedelta(horizon_weeks * 7, unit="D")

    temporal_mask = (d_j + horizon) <= d_i
    subtrain_mask = (metadata["split"].values == "subtrain")[None, :]
    same_code_mask = metadata["external_code"].astype(str).values[:, None] == metadata["external_code"].astype(str).values[None, :]

    admissible = temporal_mask & subtrain_mask & (~same_code_mask)
    return torch.from_numpy(admissible)


def compute_retrieval(metadata, embeddings, top_k, min_similarity, horizon_weeks):
    cosine_sim = compute_cosine_similarity(embeddings)
    admissible_mask = build_admissibility_mask(metadata, horizon_weeks)

    masked_sim = cosine_sim.clone()
    masked_sim[~admissible_mask] = -float("inf")
    if min_similarity is not None:
        masked_sim[masked_sim < float(min_similarity)] = -float("inf")

    k = min(int(top_k), max(1, masked_sim.shape[1] - 1))
    topk_scores, topk_indices = torch.topk(masked_sim, k=k, dim=1)
    topk_valid_mask = torch.isfinite(topk_scores)

    masked_scores = topk_scores.clone()
    masked_scores[~topk_valid_mask] = -float("inf")
    similarity_weights = torch.softmax(masked_scores, dim=1)
    similarity_weights[~topk_valid_mask] = 0.0

    row_sums = similarity_weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    similarity_weights = similarity_weights / row_sums
    retrieval_available = topk_valid_mask.any(dim=1)
    similarity_weights[~retrieval_available] = 0.0

    sales_cols = [str(i) for i in range(horizon_weeks)]
    if not all(col in metadata.columns for col in sales_cols):
        missing = [col for col in sales_cols if col not in metadata.columns]
        raise KeyError(f"Missing sales columns needed for retrieval summary: {missing}")
    sales_tensor = torch.tensor(metadata[sales_cols].values, dtype=torch.float32)

    n_items = len(metadata)
    neighbor_sales = torch.zeros((n_items, k, horizon_weeks), dtype=torch.float32)
    neighbor_external_codes = np.empty((n_items, k), dtype=object)
    neighbor_external_codes[:] = None

    for i in range(n_items):
        for rank in range(k):
            if bool(topk_valid_mask[i, rank].item()):
                j = int(topk_indices[i, rank].item())
                neighbor_sales[i, rank] = sales_tensor[j]
                neighbor_external_codes[i, rank] = metadata.loc[j, "external_code"]

    retrieval_curve = (neighbor_sales * similarity_weights.unsqueeze(-1)).sum(dim=1)

    return {
        "cosine_similarity_matrix": cosine_sim,
        "admissible_mask": admissible_mask,
        "topk_scores": topk_scores,
        "topk_indices": topk_indices,
        "topk_valid_mask": topk_valid_mask,
        "similarity_weights": similarity_weights,
        "neighbor_sales": neighbor_sales,
        "neighbor_external_codes": neighbor_external_codes,
        "retrieval_curve": retrieval_curve,
        "retrieval_available": retrieval_available,
    }


def build_neighbors_dataframe(metadata, retrieval_output, horizon_weeks):
    rows = []
    topk_scores = retrieval_output["topk_scores"]
    topk_indices = retrieval_output["topk_indices"]
    topk_valid_mask = retrieval_output["topk_valid_mask"]
    similarity_weights = retrieval_output["similarity_weights"]

    k = topk_scores.shape[1]
    for i in range(len(metadata)):
        query_release_date = pd.to_datetime(metadata.loc[i, "release_date"])
        for rank in range(k):
            valid = bool(topk_valid_mask[i, rank].item())
            row = {
                "query_index": i,
                "query_external_code": metadata.loc[i, "external_code"],
                "query_split": metadata.loc[i, "split"],
                "query_release_date": query_release_date,
                "query_category": metadata.loc[i, "category"] if "category" in metadata.columns else None,
                "rank": rank + 1,
                "is_valid_neighbor": valid,
                "forecast_horizon_weeks": horizon_weeks,
            }
            if valid:
                j = int(topk_indices[i, rank].item())
                neighbor_release_date = pd.to_datetime(metadata.loc[j, "release_date"])
                row.update(
                    {
                        "neighbor_index": j,
                        "neighbor_external_code": metadata.loc[j, "external_code"],
                        "neighbor_split": metadata.loc[j, "split"],
                        "neighbor_release_date": neighbor_release_date,
                        "neighbor_category": metadata.loc[j, "category"] if "category" in metadata.columns else None,
                        "cosine_similarity": float(topk_scores[i, rank].item()),
                        "similarity_weight": float(similarity_weights[i, rank].item()),
                        "neighbor_plus_horizon_date": neighbor_release_date + pd.to_timedelta(horizon_weeks * 7, unit="D"),
                        "days_between_launches": (query_release_date - neighbor_release_date).days,
                    }
                )
            else:
                row.update(
                    {
                        "neighbor_index": None,
                        "neighbor_external_code": None,
                        "neighbor_split": None,
                        "neighbor_release_date": None,
                        "neighbor_category": None,
                        "cosine_similarity": None,
                        "similarity_weight": 0.0,
                        "neighbor_plus_horizon_date": None,
                        "days_between_launches": None,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main(args):
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cat_dict = torch.load(Path(args.data_folder) / "category_labels.pt", weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / "color_labels.pt", weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / "fabric_labels.pt", weights_only=False)
    gtrends = pd.read_csv(Path(args.data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)

    metadata = prepare_metadata(Path(args.train_csv), Path(args.test_csv) if args.test_csv else None, args.val_frac)
    print(metadata["split"].value_counts(dropna=False).to_dict())

    model = load_baseline_model(args.checkpoint_path, args, cat_dict, col_dict, fab_dict)
    embeddings = extract_embeddings(
        metadata_df=metadata,
        args=args,
        model=model,
        img_root=Path(args.data_folder) / "images",
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
    )

    retrieval_output = compute_retrieval(
        metadata=metadata,
        embeddings=embeddings,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        horizon_weeks=args.horizon_weeks,
    )

    output = {
        "metadata": metadata,
        "embeddings": embeddings,
        "top_k": int(min(args.top_k, max(1, len(metadata) - 1))),
        "horizon_weeks": int(args.horizon_weeks),
        "min_similarity": float(args.min_similarity),
        **retrieval_output,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output_path)
    print(f"Saved retrieval memory to {args.output_path}")

    if args.neighbors_csv:
        neighbors_df = build_neighbors_dataframe(metadata, retrieval_output, args.horizon_weeks)
        Path(args.neighbors_csv).parent.mkdir(parents=True, exist_ok=True)
        neighbors_df.to_csv(args.neighbors_csv, index=False)
        print(f"Saved readable neighbors CSV to {args.neighbors_csv}")

    available = retrieval_output["retrieval_available"].float().mean().item()
    print(f"Fraction with at least one valid retrieved analog: {available:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the hybrid retrieval memory used by GTM.")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--neighbors_csv", type=str, default="")
    parser.add_argument("--gpu_num", type=int, default=0)

    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--horizon_weeks", type=int, default=12)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--min_similarity", type=float, default=0.92)

    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)

    main(parser.parse_args())
