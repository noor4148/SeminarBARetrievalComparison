import os
import argparse
import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from models.GTM import GTM
from models.FCN import FCN
from utils.data_multitrends import ZeroShotDataset
from utils.fold_utils import resolve_fold_paths, read_sorted_csv, fold_suffix

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run(args):
    print(args)
    pl.seed_everything(args.seed)

    train_csv, val_csv, _ = resolve_fold_paths(args, need_val=False)

    train_df = read_sorted_csv(train_csv)

    if val_csv is not None:
        val_df = read_sorted_csv(val_csv)
    else:
        # backwards compatibility: oude 85/15 split alleen als geen fold/val_csv is opgegeven
        val_size = max(1, int(0.15 * len(train_df)))
        val_df = train_df.iloc[-val_size:].copy().reset_index(drop=True)
        train_df = train_df.iloc[:-val_size].copy().reset_index(drop=True)

    cat_dict = torch.load(Path(args.data_folder) / "category_labels.pt", weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / "color_labels.pt", weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / "fabric_labels.pt", weights_only=False)
    gtrends = pd.read_csv(Path(args.data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)

    train_loader = ZeroShotDataset(
        train_df, Path(args.data_folder) / "images", gtrends, cat_dict, col_dict, fab_dict, args.trend_len
    ).get_loader(batch_size=args.batch_size, train=True)

    val_loader = ZeroShotDataset(
        val_df, Path(args.data_folder) / "images", gtrends, cat_dict, col_dict, fab_dict, args.trend_len
    ).get_loader(batch_size=1, train=False)

    model_savename = f"{args.model_type}_{args.wandb_run}_{fold_suffix(args)}"
    #__________________________________
    # Create model
    if args.model_type == 'FCN':
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_trends=args.use_trends,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num
        )
    else:
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
            gpu_num=args.gpu_num
        )

    # Model Training
    # Define model saving procedure
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + '/'+args.model_type,
        filename=model_savename+'---{epoch}---'+dt_string,
        monitor='val_wape',
        mode='min',
        save_top_k=1
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_wape',
        min_delta=0.0,
        patience=5,
        verbose=True,
        mode='min'
    )

    # wandb.init(entity=args.wandb_entity, project=args.wandb_proj, name=args.wandb_run)
    # wandb_logger = pl_loggers.WandbLogger()
    # wandb_logger.watch(model)

    # If you wish to use Tensorboard you can change the logger to:
    tb_logger = pl_loggers.TensorBoardLogger(args.log_dir+'/', name=model_savename)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[args.gpu_num],
        max_epochs=args.epochs,
        check_val_every_n_epoch=5,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # Fit model
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # Print out path of best model
    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=0)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM', help='Choose between GTM or FCN')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='username-here')
    parser.add_argument('--wandb_proj', type=str, default='GTM')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    # Type of fold to run
    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], default=None)
    parser.add_argument("--split_dir", type=str, default="")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--eval_csv", type=str, default="")

    args = parser.parse_args()
    run(args)