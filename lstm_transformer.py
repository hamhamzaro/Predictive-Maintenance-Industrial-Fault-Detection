"""
models/lstm_transformer.py
--------------------------
Hybrid deep learning model for predictive maintenance.

Architecture:
    - LSTM encoder: captures short-term temporal dependencies
    - Transformer encoder: captures long-range patterns via multi-head attention
    - Dual heads:
        * Classification head: normal vs anomalous (binary)
        * RUL regression head: remaining useful life estimation

Usage:
    from src.models.lstm_transformer import HybridFaultDetector, RULEstimator
"""

import torch
import torch.nn as nn
import math
from typing import Optional


# ─── Positional Encoding ──────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer.
    Injects position information into the sequence embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─── LSTM Encoder ─────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for short-term temporal feature extraction.

    Args:
        input_size:  Number of input sensor channels.
        hidden_size: LSTM hidden state dimension.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout between LSTM layers.
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch, seq_len, hidden_size * 2)  — bidirectional
        """
        out, _ = self.lstm(x)
        return self.layer_norm(out)


# ─── Transformer Encoder ──────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for long-range temporal dependency modelling.

    Projects LSTM output to d_model, applies positional encoding,
    then passes through N transformer encoder layers.

    Args:
        input_size: Dimension of incoming features (LSTM output dim).
        d_model:    Transformer model dimension.
        nhead:      Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_ff:     Feed-forward dimension inside transformer.
        dropout:    Dropout rate.
    """

    def __init__(
        self,
        input_size: int = 128,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            src_key_padding_mask: (batch, seq_len) — True = ignore position
        Returns:
            out: (batch, seq_len, d_model)
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.layer_norm(out)


# ─── Hybrid Fault Detector ────────────────────────────────────────────────────

class HybridFaultDetector(nn.Module):
    """
    End-to-end fault detection model combining LSTM + Transformer.

    Architecture:
        Input (batch, seq_len, n_sensors)
          → LSTMEncoder       → (batch, seq_len, lstm_hidden * 2)
          → TransformerEncoder → (batch, seq_len, d_model)
          → Global avg pool   → (batch, d_model)
          → Classification head → (batch, 1)  [fault probability]

    Args:
        n_sensors:   Number of input sensor channels.
        lstm_hidden: LSTM hidden dimension.
        lstm_layers: Number of LSTM layers.
        d_model:     Transformer model dimension.
        nhead:       Attention heads.
        tf_layers:   Transformer encoder layers.
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        n_sensors: int = 8,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        tf_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm_encoder = LSTMEncoder(
            input_size=n_sensors,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(
            input_size=lstm_hidden * 2,  # bidirectional
            d_model=d_model,
            nhead=nhead,
            num_layers=tf_layers,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
            # No sigmoid here — use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_sensors)
        Returns:
            logits: (batch, 1) — raw fault logit
        """
        lstm_out = self.lstm_encoder(x)
        tf_out = self.transformer_encoder(lstm_out)

        # Global average pooling over sequence dimension
        pooled = tf_out.mean(dim=1)  # (batch, d_model)

        return self.classifier(pooled)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fault probability (0-1)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ─── RUL Estimator ────────────────────────────────────────────────────────────

class RULEstimator(nn.Module):
    """
    RUL (Remaining Useful Life) estimator built on top of HybridFaultDetector.

    Shares the LSTM + Transformer backbone with HybridFaultDetector,
    replacing the classification head with a regression head.

    The regression head applies an asymmetric penalty via a custom loss:
    late predictions (overestimating RUL) are penalized more heavily
    than early predictions to favour conservative maintenance scheduling.

    Args:
        backbone:    Pre-trained or fresh HybridFaultDetector instance.
        freeze_backbone: Freeze LSTM + Transformer weights during RUL training.
    """

    def __init__(self, backbone: HybridFaultDetector, freeze_backbone: bool = False):
        super().__init__()

        self.lstm_encoder = backbone.lstm_encoder
        self.transformer_encoder = backbone.transformer_encoder

        if freeze_backbone:
            for param in self.lstm_encoder.parameters():
                param.requires_grad = False
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

        d_model = backbone.transformer_encoder.layer_norm.normalized_shape[0]

        # RUL regression head
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.ReLU()  # RUL cannot be negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_sensors)
        Returns:
            rul_pred: (batch, 1) — predicted RUL in hours
        """
        lstm_out = self.lstm_encoder(x)
        tf_out = self.transformer_encoder(lstm_out)
        pooled = tf_out.mean(dim=1)
        return self.rul_head(pooled)


# ─── Asymmetric RUL Loss ──────────────────────────────────────────────────────

class AsymmetricRULLoss(nn.Module):
    """
    Custom loss for RUL estimation with asymmetric penalty.

    Penalizes overestimation of RUL (late prediction) more than
    underestimation (early prediction), to prefer conservative
    maintenance scheduling.

    Loss = mean(w * (pred - true)^2)
    where w = late_penalty if pred > true else 1.0

    Args:
        late_penalty: Weight multiplier for late predictions (default 2.0).
    """

    def __init__(self, late_penalty: float = 2.0):
        super().__init__()
        self.late_penalty = late_penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = pred - target
        weights = torch.where(errors > 0, self.late_penalty * torch.ones_like(errors), torch.ones_like(errors))
        return (weights * errors ** 2).mean()


# ─── Autoencoder (Baseline) ───────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for unsupervised anomaly detection.

    Trained on healthy sequences only. Anomaly score = reconstruction MSE.
    High reconstruction error → anomalous sequence.

    Args:
        n_sensors: Number of input channels.
        seq_len:   Sequence length (window size).
        latent_dim: Bottleneck dimension.
    """

    def __init__(self, n_sensors: int = 8, seq_len: int = 50, latent_dim: int = 16):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_sensors, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(16 * seq_len, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16 * seq_len),
            nn.Unflatten(1, (16, seq_len)),
            nn.ConvTranspose1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, n_sensors, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, n_sensors)
        Returns:
            reconstruction: (batch, seq_len, n_sensors)
            latent:         (batch, latent_dim)
        """
        x_t = x.permute(0, 2, 1)   # → (batch, n_sensors, seq_len) for Conv1d
        latent = self.encoder(x_t)
        recon_t = self.decoder(latent)
        recon = recon_t.permute(0, 2, 1)
        return recon, latent

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction MSE."""
        with torch.no_grad():
            recon, _ = self.forward(x)
            return ((x - recon) ** 2).mean(dim=(1, 2))


# ─── Model Factory ────────────────────────────────────────────────────────────

def build_model(model_type: str = "hybrid", n_sensors: int = 8, seq_len: int = 50) -> nn.Module:
    """
    Factory function for model instantiation.

    Args:
        model_type: 'hybrid' | 'autoencoder' | 'rul'
        n_sensors:  Number of sensor channels.
        seq_len:    Input sequence length.

    Returns:
        Instantiated PyTorch model.
    """
    if model_type == "hybrid":
        return HybridFaultDetector(n_sensors=n_sensors)
    elif model_type == "autoencoder":
        return ConvAutoencoder(n_sensors=n_sensors, seq_len=seq_len)
    elif model_type == "rul":
        backbone = HybridFaultDetector(n_sensors=n_sensors)
        return RULEstimator(backbone)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: hybrid, autoencoder, rul")


if __name__ == "__main__":
    # Smoke test
    batch, seq_len, n_sensors = 4, 50, 8

    print("Testing HybridFaultDetector...")
    model = HybridFaultDetector(n_sensors=n_sensors)
    x = torch.randn(batch, seq_len, n_sensors)
    out = model(x)
    print(f"  Input: {x.shape} → Output: {out.shape}")

    print("\nTesting ConvAutoencoder...")
    ae = ConvAutoencoder(n_sensors=n_sensors, seq_len=seq_len)
    recon, latent = ae(x)
    scores = ae.anomaly_score(x)
    print(f"  Input: {x.shape} → Recon: {recon.shape}, Latent: {latent.shape}")
    print(f"  Anomaly scores: {scores.shape}")

    print("\nTesting RULEstimator...")
    rul_model = RULEstimator(model)
    rul_pred = rul_model(x)
    print(f"  Input: {x.shape} → RUL: {rul_pred.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nHybrid model parameters: {total_params:,}")
    print("\nAll tests passed ✅")
