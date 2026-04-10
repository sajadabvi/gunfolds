"""
Tier 2: BrainNet Transformer — graph-level HC vs SZ classification.

A transformer-based architecture inspired by Kan et al. (NeurIPS 2022)
adapted for directed causal graphs produced by RASL / PCMCI / GCM.
Each brain region is treated as a token whose features come from its
row/column in the (mean) adjacency matrix, plus a learnable domain
embedding.  Self-attention captures global inter-region interactions,
and an orthonormal-clustering readout produces a fixed-size
representation for the classifier head.

Usage:
    python brain_transformer_classify.py --timestamp 03232026175230
    python brain_transformer_classify.py --timestamp 03232026175230 \\
        --configs N10_domain_RASL N10_none_PCMCI --epochs 80
"""

import os
import sys
import glob
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from gunfolds.utils import zickle as zkl
from gunfolds import conversions as cv

_REAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "real_data"
)
if _REAL_DATA_DIR not in sys.path:
    sys.path.insert(0, _REAL_DATA_DIR)
from component_config import INDEX_TO_DOMAIN, DOMAIN_ORDER, get_comp_indices

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tier-2 BrainNet Transformer.")
    p.add_argument("--timestamp", required=True)
    p.add_argument("--results_root", default="fbirn_results")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_clusters", type=int, default=4,
                   help="Orthonormal clustering readout clusters")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--edge_dropout", type=float, default=0.1,
                   help="Data augmentation: randomly zero edges during training")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading & dataset
# ---------------------------------------------------------------------------

def discover_configs(root_dir):
    configs = []
    if not os.path.isdir(root_dir):
        return configs
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        parts = name.split("_")
        if len(parts) >= 3 and parts[0].startswith("N"):
            configs.append(name)
    return configs


def load_config_subjects(config_dir):
    pattern = os.path.join(config_dir, "subject_*", "result.zkl")
    results = []
    for f in sorted(glob.glob(pattern)):
        try:
            results.append(zkl.load(f))
        except Exception:
            pass
    return results


def _adj_from_sol(sol):
    if "adj" in sol:
        return np.array(sol["adj"], dtype=np.float32)
    return (cv.graph2adj(sol["graph"]) > 0).astype(np.float32)


def build_subject_data(subject_results, comp_indices):
    """
    Build arrays for the transformer dataset.

    Returns
    -------
    adj_matrices : ndarray [n_subj, N, N]  (mean adjacency)
    edge_std     : ndarray [n_subj, N, N]  (solution agreement)
    domain_ids   : ndarray [N]             (int domain id per node)
    labels       : ndarray [n_subj]
    """
    N = len(comp_indices)
    domain_map = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    domain_ids = np.array(
        [domain_map.get(INDEX_TO_DOMAIN.get(idx, "?"), 0) for idx in comp_indices],
        dtype=np.int64,
    )

    adjs = []
    stds = []
    labels = []
    for info in subject_results:
        sol_adjs = np.stack([_adj_from_sol(s) for s in info["solutions"]])
        mean_a = sol_adjs.mean(axis=0)
        np.fill_diagonal(mean_a, 0)
        std_a = sol_adjs.std(axis=0) if sol_adjs.shape[0] > 1 else np.zeros_like(mean_a)
        np.fill_diagonal(std_a, 0)
        adjs.append(mean_a)
        stds.append(std_a)
        labels.append(info["group"])

    return (
        np.array(adjs, dtype=np.float32),
        np.array(stds, dtype=np.float32),
        domain_ids,
        np.array(labels, dtype=np.int64),
    )


class BrainGraphDataset(Dataset):
    def __init__(self, adj_matrices, edge_std, domain_ids, labels,
                 edge_dropout=0.0, training=False):
        self.adj = torch.from_numpy(adj_matrices)
        self.std = torch.from_numpy(edge_std)
        self.domain = torch.from_numpy(domain_ids)
        self.labels = torch.from_numpy(labels)
        self.edge_dropout = edge_dropout
        self.training = training

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a = self.adj[idx].clone()
        s = self.std[idx].clone()
        if self.training and self.edge_dropout > 0:
            mask = (torch.rand_like(a) > self.edge_dropout).float()
            a = a * mask
        return a, s, self.domain, self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BrainNetTransformer(nn.Module):
    """
    Transformer for brain connectivity classification.

    Per-node features: concatenation of outgoing edge profile (row),
    incoming edge profile (column), edge-std profile (row), and a
    learnable domain embedding.

    The transformer encoder processes the N-node sequence, followed
    by an orthonormal-clustering readout that maps N tokens to K
    cluster representations, which are concatenated and fed to an
    MLP classifier.
    """

    def __init__(self, n_nodes, n_domains, d_model=64, n_heads=4,
                 n_layers=2, n_clusters=4, dropout=0.3):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model

        # Node feature projection: in_row(N) + in_col(N) + std_row(N) => 3N
        self.input_proj = nn.Linear(3 * n_nodes, d_model)
        self.domain_emb = nn.Embedding(n_domains, d_model)
        self.pos_emb = nn.Embedding(n_nodes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.cluster_proj = nn.Linear(d_model, n_clusters)
        self.classifier = nn.Sequential(
            nn.Linear(n_clusters * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, adj, std, domain_ids):
        B, N, _ = adj.shape
        row_feat = adj
        col_feat = adj.transpose(1, 2)
        std_feat = std
        x = torch.cat([row_feat, col_feat, std_feat], dim=-1)  # [B, N, 3N]
        x = self.input_proj(x)  # [B, N, d_model]

        pos = torch.arange(N, device=adj.device)
        x = x + self.domain_emb(domain_ids) + self.pos_emb(pos)

        x = self.encoder(x)  # [B, N, d_model]

        # Orthonormal-clustering readout
        assign = F.softmax(self.cluster_proj(x), dim=-1)  # [B, N, K]
        clustered = torch.bmm(assign.transpose(1, 2), x)  # [B, K, d_model]
        pooled = clustered.reshape(B, -1)  # [B, K * d_model]

        return self.classifier(pooled)  # [B, 2]

    def get_attention_weights(self, adj, std, domain_ids):
        """Extract attention weights for interpretability."""
        B, N, _ = adj.shape
        row_feat = adj
        col_feat = adj.transpose(1, 2)
        std_feat = std
        x = torch.cat([row_feat, col_feat, std_feat], dim=-1)
        x = self.input_proj(x)

        pos = torch.arange(N, device=adj.device)
        x = x + self.domain_emb(domain_ids) + self.pos_emb(pos)

        attn_weights = []
        for layer in self.encoder.layers:
            x2, w = layer.self_attn(x, x, x, need_weights=True)
            attn_weights.append(w.detach().cpu().numpy())
            x = layer(x)
        return attn_weights


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for adj, std, dom, labels in loader:
        adj, std, dom, labels = (
            adj.to(DEVICE), std.to(DEVICE), dom.to(DEVICE), labels.to(DEVICE))
        optimizer.zero_grad()
        logits = model(adj, std, dom)
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
    all_preds = []
    all_probs = []
    all_labels = []
    for adj, std, dom, labels in loader:
        adj, std, dom = adj.to(DEVICE), std.to(DEVICE), dom.to(DEVICE)
        logits = model(adj, std, dom)
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
    return acc, auc, all_preds, all_probs, all_labels


# ---------------------------------------------------------------------------
# Cross-validation driver
# ---------------------------------------------------------------------------

def run_cv(adj_matrices, edge_std, domain_ids, labels, args):
    """Stratified K-fold cross-validation."""
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_results = []
    N = adj_matrices.shape[1]
    n_domains = len(DOMAIN_ORDER)

    for fold, (train_idx, val_idx) in enumerate(skf.split(adj_matrices, labels)):
        print(f"    Fold {fold + 1}/{args.n_folds} ...", end=" ", flush=True)
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)

        train_ds = BrainGraphDataset(
            adj_matrices[train_idx], edge_std[train_idx],
            domain_ids, labels[train_idx],
            edge_dropout=args.edge_dropout, training=True,
        )
        val_ds = BrainGraphDataset(
            adj_matrices[val_idx], edge_std[val_idx],
            domain_ids, labels[val_idx],
            edge_dropout=0.0, training=False,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False)

        # Handle class imbalance
        n0 = (labels[train_idx] == 0).sum()
        n1 = (labels[train_idx] == 1).sum()
        w0 = len(train_idx) / (2.0 * max(n0, 1))
        w1 = len(train_idx) / (2.0 * max(n1, 1))
        class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = BrainNetTransformer(
            n_nodes=N, n_domains=n_domains, d_model=args.d_model,
            n_heads=args.n_heads, n_layers=args.n_layers,
            n_clusters=args.n_clusters, dropout=args.dropout,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion)
            scheduler.step()
            val_acc, val_auc, _, _, _ = evaluate(model, val_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        val_acc, val_auc, preds, probs, true_labels = evaluate(model, val_loader)

        print(f"acc={val_acc:.3f}  auc={val_auc:.3f}")
        fold_results.append({
            "fold": fold,
            "accuracy": val_acc,
            "auc": val_auc,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        })

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root_dir = os.path.join(args.results_root, args.timestamp)
    if not os.path.isdir(root_dir):
        alt = os.path.join("..", "real_data", args.results_root, args.timestamp)
        if os.path.isdir(alt):
            root_dir = alt
        else:
            print(f"Error: {root_dir} not found")
            sys.exit(1)

    configs = args.configs or discover_configs(root_dir)
    if not configs:
        print("No configs found.")
        sys.exit(1)

    out_dir = args.output_dir or os.path.join(root_dir, "ml_classification")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("TIER 2: BRAIN-NET TRANSFORMER — HC vs SZ CLASSIFICATION")
    print("=" * 80)
    print(f"Timestamp: {args.timestamp}")
    print(f"Configs:   {configs}")
    print(f"Device:    {DEVICE}")
    print(f"Model:     d={args.d_model} heads={args.n_heads} "
          f"layers={args.n_layers} clusters={args.n_clusters}")
    print()

    all_rows = []

    for cfg in configs:
        cfg_dir = os.path.join(root_dir, cfg)
        if not os.path.isdir(cfg_dir):
            print(f"  {cfg}: not found, skipping")
            continue

        print(f"  Loading {cfg} ...")
        subjects = load_config_subjects(cfg_dir)
        if len(subjects) < 10:
            print(f"    Only {len(subjects)} subjects, skipping")
            continue

        parts = cfg.split("_")
        n_comp = int(parts[0][1:]) if parts[0].startswith("N") else 10
        comp_indices = get_comp_indices(n_comp)

        adj_matrices, edge_std, domain_ids, labels = build_subject_data(
            subjects, comp_indices)

        n0 = (labels == 0).sum()
        n1 = (labels == 1).sum()
        print(f"    {len(subjects)} subjects, N={adj_matrices.shape[1]}, "
              f"group0={n0} group1={n1}")

        fold_results = run_cv(adj_matrices, edge_std, domain_ids, labels, args)

        accs = [r["accuracy"] for r in fold_results]
        aucs = [r["auc"] for r in fold_results]
        row = {
            "config": cfg,
            "classifier": "BrainNetTransformer",
            "method": parts[2] if len(parts) >= 3 else cfg,
            "n_components": n_comp,
            "scc_strategy": parts[1] if len(parts) >= 2 else "",
            "n_subjects": len(subjects),
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
            "fold_details": fold_results,
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

    csv_path = os.path.join(out_dir, "tier2_results.csv")
    df.drop(columns=["fold_details"], errors="ignore").to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
