"""
Time-Series Foundation Model: Spatiotemporal Transformer for HC vs SZ
classification directly from raw ICA time series.

This bypasses causal discovery entirely and learns to classify
subjects from their ICA component time courses.  The architecture
uses spatial attention (across brain regions at each time point) and
temporal attention (across time for each region), following the
factored spatiotemporal transformer design used in recent brain-signal
foundation models (e.g. BrainLM, LaBraM).

This provides a comparison point: can a large, data-driven model
learn to separate HC from SZ without explicit causal structure?
If RASL-based classifiers (Tiers 1-3) outperform this, it validates
the value of causal discovery.  If this outperforms, it suggests
raw temporal dynamics contain information beyond what static causal
graphs capture.

Usage:
    python timeseries_foundation_classify.py --timestamp 03232026175230
    python timeseries_foundation_classify.py \\
        --data_path ../real_data/../fbirn/fbirn_sz_data.npz \\
        --n_components 10 --epochs 80
"""

import os
import sys
import argparse
import warnings
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

_REAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "real_data"
)
if _REAL_DATA_DIR not in sys.path:
    sys.path.insert(0, _REAL_DATA_DIR)
from component_config import (
    get_comp_indices, get_comp_names, INDEX_TO_DOMAIN, DOMAIN_ORDER,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Time-series foundation model for HC vs SZ.")
    p.add_argument("--data_path", default=None,
                   help="Path to fbirn_sz_data.npz")
    p.add_argument("--timestamp", default=None,
                   help="If given, save results under fbirn_results/<ts>/ml_classification/")
    p.add_argument("--results_root", default="fbirn_results")
    p.add_argument("--n_components", type=int, nargs="+", default=[10, 20],
                   help="Component set sizes to evaluate")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_spatial_layers", type=int, default=2)
    p.add_argument("--n_temporal_layers", type=int, default=2)
    p.add_argument("--temporal_pool_size", type=int, default=8,
                   help="Downsample temporal dimension by this factor")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def find_data_path():
    """Try common locations for the fMRI data file."""
    candidates = [
        "../real_data/../fbirn/fbirn_sz_data.npz",
        "../fbirn/fbirn_sz_data.npz",
        "../../fbirn/fbirn_sz_data.npz",
        os.path.join(_REAL_DATA_DIR, "..", "fbirn", "fbirn_sz_data.npz"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_fmri_data(data_path, comp_indices):
    """
    Load FBIRN data and select components.

    Returns
    -------
    ts : ndarray [n_subjects, T, N]
    labels : ndarray [n_subjects]
    """
    npz = np.load(data_path)
    data = npz["data"]  # [n_subjects, T, 53]
    if "labels" in npz.files:
        labels = npz["labels"]
    elif "label" in npz.files:
        labels = npz["label"]
    else:
        raise KeyError("No labels in NPZ")
    ts = data[:, :, comp_indices]
    return ts.astype(np.float32), labels.astype(np.int64)


class fMRIDataset(Dataset):
    def __init__(self, timeseries, domain_ids, labels,
                 augment=False, noise_std=0.05):
        """
        Parameters
        ----------
        timeseries : ndarray [n_subj, T, N]
        domain_ids : ndarray [N]
        labels     : ndarray [n_subj]
        """
        self.ts = torch.from_numpy(timeseries)
        self.domain = torch.from_numpy(domain_ids)
        self.labels = torch.from_numpy(labels)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.ts[idx].clone()  # [T, N]
        if self.augment and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.domain, self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Reshape time series [B, T, N] into patches of size P along time,
    then project each patch to d_model.
    """
    def __init__(self, n_regions, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(n_regions * patch_size, d_model)

    def forward(self, x):
        B, T, N = x.shape
        P = self.patch_size
        n_patches = T // P
        x = x[:, :n_patches * P, :]
        x = x.reshape(B, n_patches, P * N)
        return self.proj(x)  # [B, n_patches, d_model]


class SpatioTemporalTransformer(nn.Module):
    """
    Factored spatiotemporal transformer for fMRI ICA time series.

    Architecture:
      1. Per-timepoint spatial attention: each time step, attend across
         N brain regions to capture cross-regional interactions.
         (Operates on temporally-pooled segments for efficiency.)
      2. Per-region temporal attention: each region attends across the
         temporal dimension to capture dynamics.
      3. Global pooling + MLP classifier.

    For a time series [T, N]:
      - Temporal pooling: average every `pool_size` frames -> [T', N]
      - Spatial transformer: [B, T', N, d] -> attend over N per timepoint
      - Temporal transformer: [B, N, T', d] -> attend over T' per region
      - Pool and classify
    """

    def __init__(self, n_regions, n_domains, d_model=64, n_heads=4,
                 n_spatial_layers=2, n_temporal_layers=2,
                 temporal_pool_size=8, dropout=0.3):
        super().__init__()
        self.n_regions = n_regions
        self.pool_size = temporal_pool_size

        self.region_proj = nn.Linear(1, d_model)
        self.domain_emb = nn.Embedding(n_domains, d_model)
        self.region_pos = nn.Embedding(n_regions, d_model)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=n_spatial_layers)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=n_temporal_layers)

        self.temporal_pos_emb = None  # created dynamically

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def _get_temporal_pos(self, T_prime, device):
        """Sinusoidal positional encoding for the temporal dimension."""
        pos = torch.arange(T_prime, device=device).unsqueeze(1).float()
        dim = torch.arange(0, self.classifier[0].in_features, 2,
                           device=device).float()
        d = self.classifier[0].in_features
        pe = torch.zeros(T_prime, d, device=device)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (dim[:pe[:, 0::2].size(1)] / d)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (dim[:pe[:, 1::2].size(1)] / d)))
        return pe

    def forward(self, ts, domain_ids):
        """
        Parameters
        ----------
        ts         : [B, T, N]
        domain_ids : [N]
        """
        B, T, N = ts.shape
        P = self.pool_size
        T_prime = T // P

        # Temporal pooling (average)
        ts_pooled = ts[:, :T_prime * P, :].reshape(B, T_prime, P, N).mean(dim=2)
        # ts_pooled: [B, T_prime, N]

        # Z-score per subject
        mu = ts_pooled.mean(dim=1, keepdim=True)
        std = ts_pooled.std(dim=1, keepdim=True).clamp(min=1e-6)
        ts_pooled = (ts_pooled - mu) / std

        # === Spatial attention (across regions per time step) ===
        # Reshape to [B * T_prime, N, 1] -> project -> add embeddings
        x_spatial = ts_pooled.reshape(B * T_prime, N, 1)
        x_spatial = self.region_proj(x_spatial)  # [B*T', N, d]
        x_spatial = (x_spatial
                     + self.domain_emb(domain_ids)
                     + self.region_pos(torch.arange(N, device=ts.device)))
        x_spatial = self.spatial_encoder(x_spatial)  # [B*T', N, d]

        # === Temporal attention (across time per region) ===
        x_spatial = x_spatial.reshape(B, T_prime, N, -1)
        d = x_spatial.size(-1)
        # Permute to [B*N, T_prime, d]
        x_temporal = x_spatial.permute(0, 2, 1, 3).reshape(B * N, T_prime, d)
        temp_pos = self._get_temporal_pos(T_prime, ts.device)
        x_temporal = x_temporal + temp_pos.unsqueeze(0)
        x_temporal = self.temporal_encoder(x_temporal)  # [B*N, T', d]

        # Global average pooling over time and regions
        x_temporal = x_temporal.reshape(B, N, T_prime, d)
        x_pooled = x_temporal.mean(dim=(1, 2))  # [B, d]

        return self.classifier(x_pooled)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for ts, dom, labels in loader:
        ts, dom, labels = ts.to(DEVICE), dom.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ts, dom)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for ts, dom, labels in loader:
        ts, dom = ts.to(DEVICE), dom.to(DEVICE)
        logits = model(ts, dom)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return acc, auc


def run_cv(ts_data, labels, comp_indices, args):
    N = len(comp_indices)
    domain_map = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    domain_ids = np.array(
        [domain_map.get(INDEX_TO_DOMAIN.get(idx, "?"), 0) for idx in comp_indices],
        dtype=np.int64,
    )

    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(ts_data, labels)):
        print(f"    Fold {fold + 1}/{args.n_folds} ...", end=" ", flush=True)
        torch.manual_seed(args.seed + fold)

        train_ds = fMRIDataset(
            ts_data[train_idx], domain_ids, labels[train_idx],
            augment=True, noise_std=0.05)
        val_ds = fMRIDataset(
            ts_data[val_idx], domain_ids, labels[val_idx],
            augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False)

        n0 = (labels[train_idx] == 0).sum()
        n1 = (labels[train_idx] == 1).sum()
        w0 = len(train_idx) / (2.0 * max(n0, 1))
        w1 = len(train_idx) / (2.0 * max(n1, 1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = SpatioTemporalTransformer(
            n_regions=N, n_domains=len(DOMAIN_ORDER),
            d_model=args.d_model, n_heads=args.n_heads,
            n_spatial_layers=args.n_spatial_layers,
            n_temporal_layers=args.n_temporal_layers,
            temporal_pool_size=args.temporal_pool_size,
            dropout=args.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        best_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(args.epochs):
            train_one_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()
            val_acc, val_auc = evaluate(model, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        val_acc, val_auc = evaluate(model, val_loader)

        print(f"acc={val_acc:.3f}  auc={val_auc:.3f}")
        fold_results.append({"fold": fold, "accuracy": val_acc, "auc": val_auc})

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_path = args.data_path or find_data_path()
    if data_path is None or not os.path.isfile(data_path):
        print("Error: cannot find fbirn_sz_data.npz. Use --data_path.")
        sys.exit(1)

    if args.output_dir:
        out_dir = args.output_dir
    elif args.timestamp:
        out_dir = os.path.join(
            args.results_root, args.timestamp, "ml_classification")
    else:
        out_dir = "ml_classification"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("TIME-SERIES FOUNDATION MODEL — HC vs SZ CLASSIFICATION")
    print("=" * 80)
    print(f"Data:        {data_path}")
    print(f"Components:  {args.n_components}")
    print(f"Device:      {DEVICE}")
    print(f"Model:       d={args.d_model} heads={args.n_heads} "
          f"spatial_layers={args.n_spatial_layers} "
          f"temporal_layers={args.n_temporal_layers}")
    print(f"Pool size:   {args.temporal_pool_size}")
    print()

    all_rows = []

    for n_comp in args.n_components:
        comp_indices = get_comp_indices(n_comp)
        comp_names = get_comp_names(comp_indices)
        print(f"  N = {n_comp}  ({', '.join(comp_names[:5])}...)")

        ts_data, labels = load_fmri_data(data_path, comp_indices)
        n0 = (labels == 0).sum()
        n1 = (labels == 1).sum()
        print(f"    {ts_data.shape[0]} subjects, T={ts_data.shape[1]}, "
              f"N={ts_data.shape[2]}, group0={n0} group1={n1}")

        fold_results = run_cv(ts_data, labels, comp_indices, args)

        accs = [r["accuracy"] for r in fold_results]
        aucs = [r["auc"] for r in fold_results]
        row = {
            "config": f"N{n_comp}_timeseries",
            "classifier": "SpatioTemporalTransformer",
            "method": "TimeSeries",
            "n_components": n_comp,
            "scc_strategy": "N/A",
            "n_subjects": ts_data.shape[0],
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
        }
        all_rows.append(row)

    if not all_rows:
        print("No results.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    show_cols = ["config", "classifier", "accuracy_mean", "accuracy_std",
                 "auc_mean", "auc_std"]
    show = df[[c for c in show_cols if c in df.columns]].copy()
    for c in ["accuracy_mean", "accuracy_std", "auc_mean", "auc_std"]:
        if c in show.columns:
            show[c] = show[c].round(4)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(show.to_string(index=False))

    csv_path = os.path.join(out_dir, "timeseries_foundation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
