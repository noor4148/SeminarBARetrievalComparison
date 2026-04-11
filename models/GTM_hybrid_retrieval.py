import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GTM import (
    GTM,
    PositionalEncoding,
    TimeDistributed,
    compute_forecast_metrics,
)


class RetrievalMemoryEncoder(nn.Module):
    # Encode a retrieved 12-week analog curve as an extra decoder memory sequence.

    def __init__(self, hidden_dim, seq_len, num_heads=4):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        self.pos_embedding = PositionalEncoding(hidden_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, retrieval_curve: torch.Tensor) -> torch.Tensor:
        retrieval_curve = retrieval_curve.unsqueeze(-1)  # [batch, seq_len, 1]
        retrieval_emb = self.input_linear(retrieval_curve)
        retrieval_emb = self.pos_embedding(retrieval_emb.permute(1, 0, 2))
        return self.encoder(retrieval_emb)


class HybridRetrievalGTM(GTM):
    # GTM augmented with a retrieved analog-sales memory:
    # The decoder attends to both:
    # 1) Google Trends encoder memory
    # 2) A retrieved 12-week analog sales curve encoded as a sequence

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_heads,
        num_layers,
        use_text,
        use_img,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len,
        num_trends,
        gpu_num,
        use_encoder_mask=1,
        autoregressive=False,
        retrieval_seq_len=12,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_text=use_text,
            use_img=use_img,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            trend_len=trend_len,
            num_trends=num_trends,
            gpu_num=gpu_num,
            use_encoder_mask=use_encoder_mask,
            autoregressive=autoregressive,
        )
        self.retrieval_seq_len = int(retrieval_seq_len)
        self.retrieval_encoder = RetrievalMemoryEncoder(hidden_dim, self.retrieval_seq_len, num_heads=num_heads)
        self.validation_outputs = []

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 7:
            item_sales, category, color, fabric, temporal_features, gtrends, images = batch
            analog_curve = None
            analog_available = None
        elif len(batch) == 9:
            item_sales, category, color, fabric, temporal_features, gtrends, images, analog_curve, analog_available = batch
        else:
            raise ValueError(f"Unexpected batch size {len(batch)}.")
        return item_sales, category, color, fabric, temporal_features, gtrends, images, analog_curve, analog_available

    def _build_memory(self, gtrend_encoding, analog_curve=None, analog_available=None):
        memory = gtrend_encoding
        memory_key_padding_mask = None

        if analog_curve is None or analog_available is None:
            return memory, memory_key_padding_mask

        retrieval_encoding = self.retrieval_encoder(analog_curve)
        memory = torch.cat([gtrend_encoding, retrieval_encoding], dim=0)

        batch_size = gtrend_encoding.shape[1]
        gtrend_mask = torch.zeros(
            (batch_size, gtrend_encoding.shape[0]),
            dtype=torch.bool,
            device=gtrend_encoding.device,
        )
        retrieval_mask = (~analog_available.bool()).unsqueeze(1).expand(-1, retrieval_encoding.shape[0])
        memory_key_padding_mask = torch.cat([gtrend_mask, retrieval_mask], dim=1)
        return memory, memory_key_padding_mask

    def forward(
        self,
        category,
        color,
        fabric,
        temporal_features,
        gtrends,
        images,
        analog_curve=None,
        analog_available=None,
    ):
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        gtrend_encoding = self.gtrend_encoder(gtrends)

        static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, dummy_encoding)
        memory, memory_key_padding_mask = self._build_memory(
            gtrend_encoding,
            analog_curve=analog_curve,
            analog_available=analog_available,
        )

        if self.autoregressive == 1:
            tgt = torch.zeros(
                self.output_len,
                gtrend_encoding.shape[1],
                gtrend_encoding.shape[-1],
                device=gtrend_encoding.device,
            )
            tgt[0] = static_feature_fusion
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len)
            decoder_out, attn_weights = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            forecast = self.decoder_fc(decoder_out)
        else:
            tgt = static_feature_fusion.unsqueeze(0)
            decoder_out, attn_weights = self.decoder(
                tgt,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def training_step(self, train_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images, analog_curve, analog_available = self._unpack_batch(train_batch)
        forecasted_sales, _ = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            analog_curve=analog_curve,
            analog_available=analog_available,
        )
        loss = F.mse_loss(item_sales, forecasted_sales)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def validation_step(self, val_batch, batch_idx):
        item_sales, category, color, fabric, temporal_features, gtrends, images, analog_curve, analog_available = self._unpack_batch(val_batch)
        forecasted_sales, _ = self.forward(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            analog_curve=analog_curve,
            analog_available=analog_available,
        )
        self.validation_outputs.append(
            {
                "item_sales": item_sales.detach(),
                "forecasted_sales": forecasted_sales.detach(),
            }
        )

    def on_validation_epoch_end(self):
        if len(self.validation_outputs) == 0:
            return

        item_sales = torch.cat([x["item_sales"] for x in self.validation_outputs], dim=0)
        forecasted_sales = torch.cat([x["forecasted_sales"] for x in self.validation_outputs], dim=0)

        loss = F.mse_loss(item_sales, forecasted_sales)
        rescaled_item_sales = item_sales * 1065
        rescaled_forecasted_sales = forecasted_sales * 1065

        val_wape_norm, val_mae_norm, val_ts_norm, val_erp_norm = compute_forecast_metrics(
            item_sales,
            forecasted_sales,
            erp_epsilon=0.1,
        )
        val_wape, val_mae, val_ts, val_erp = compute_forecast_metrics(
            rescaled_item_sales,
            rescaled_forecasted_sales,
            erp_epsilon=0.1,
        )

        self.log("val_loss", loss)
        self.log("val_wape_norm", val_wape_norm, prog_bar=False)
        self.log("val_mae_norm", val_mae_norm, prog_bar=False)
        self.log("val_ts_norm", val_ts_norm, prog_bar=False)
        self.log("val_erp_norm", val_erp_norm, prog_bar=False)
        self.log("val_wape", val_wape, prog_bar=False)
        self.log("val_mae", val_mae, prog_bar=True)
        self.log("val_ts", val_ts, prog_bar=False)
        self.log("val_erp", val_erp, prog_bar=False)

        print(
            f"Validation normalized | WAPE: {val_wape_norm.item():.3f} | "
            f"MAE: {val_mae_norm.item():.3f} | TS: {val_ts_norm.item():.3f} | "
            f"ERP: {val_erp_norm.item():.3f}"
        )
        print(
            f"Validation rescaled | WAPE: {val_wape.item():.3f} | "
            f"MAE: {val_mae.item():.3f} | TS: {val_ts.item():.3f} | "
            f"ERP: {val_erp.item():.3f}"
        )
        self.validation_outputs.clear()
