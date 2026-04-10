"""
Tier 3: Solution-Set Transformer — HC vs SZ classification leveraging
RASL's multi-solution output per subject.

This architecture is unique to RASL: rather than collapsing the k
candidate causal graphs into a single mean, it learns from the *set*
of solutions.  A shared graph encoder maps each solution to an
embedding, then a Set Transformer aggregation (Induced Set Attention
Block) pools the k embeddings into a fixed-size subject representation
that is fed to a classifier.

The hypothesis is that the *pattern of variation* across RASL solutions
is itself a diagnostic biomarker — e.g. SZ subjects may have more
dispersed solution sets or different high-confidence edges.

For PCMCI / GCM baselines (single solution per subject) the set
attention degenerates to a simple passthrough, providing a fair
comparison.

Usage:
    python solution_set_transformer.py --timestamp 03232026175230
    python solution_set_transformer.py --timestamp 03232026175230 \\
        --configs N10_domain_RASL N10_none_PCMCI --epochs 80
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from gunfolds.utils import zickle as zkl
from gunfolds import conversions as cv

_REAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "real_data"
)
if _REAL_DATA_DIR not in sys.path:
    sys.path.insert(0, _REAL_DATA_DIR)
from component_config import get_comp_indices

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tier-3 Solution-Set Transformer.")
    p.add_argument("--timestamp", required=True)
    p.add_argument("--results_root", default="fbirn_results")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--max_solutions", type=int, default=10,
                   help="Max solutions per subject (pad/truncate)")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_inducing", type=int, default=4,
                   help="Inducing points for ISAB pooling")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
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


def build_solution_set_data(subject_results, max_solutions):
    """
    Build padded solution-set tensors.

    Returns
    -------
    sol_adjs  : ndarray [n_subj, max_solutions, N, N]
    sol_costs : ndarray [n_subj, max_solutions]
    sol_urate : ndarray [n_subj, max_solutions]
    sol_mask  : ndarray [n_subj, max_solutions]  (1 = real, 0 = pad)
    labels    : ndarray [n_subj]
    n_nodes   : int
    """
    n_subj = len(subject_results)
    first_adj = _adj_from_sol(subject_results[0]["solutions"][0])
    N = first_adj.shape[0]

    sol_adjs = np.zeros((n_subj, max_solutions, N, N), dtype=np.float32)
    sol_costs = np.zeros((n_subj, max_solutions), dtype=np.float32)
    sol_urate = np.zeros((n_subj, max_solutions), dtype=np.float32)
    sol_mask = np.zeros((n_subj, max_solutions), dtype=np.float32)
    labels = np.zeros(n_subj, dtype=np.int64)

    for i, info in enumerate(subject_results):
        labels[i] = info["group"]
        sols = info["solutions"][:max_solutions]
        for j, sol in enumerate(sols):
            adj = _adj_from_sol(sol)
            np.fill_diagonal(adj, 0)
            sol_adjs[i, j] = adj
            sol_costs[i, j] = sol.get("cost", 0.0)
            u = sol.get("undersampling")
            if u is not None:
                sol_urate[i, j] = u[0] if isinstance(u, (list, tuple)) else float(u)
            sol_mask[i, j] = 1.0

    return sol_adjs, sol_costs, sol_urate, sol_mask, labels, N


class SolutionSetDataset(Dataset):
    def __init__(self, sol_adjs, sol_costs, sol_urate, sol_mask, labels):
        self.adjs = torch.from_numpy(sol_adjs)
        self.costs = torch.from_numpy(sol_costs)
        self.urate = torch.from_numpy(sol_urate)
        self.mask = torch.from_numpy(sol_mask)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.adjs[idx], self.costs[idx], self.urate[idx],
                self.mask[idx], self.labels[idx])


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class MultiheadAttentionBlock(nn.Module):
    """Standard multi-head attention with residual + layernorm."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        out, _ = self.attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(query + self.dropout(out))


class ISAB(nn.Module):
    """
    Induced Set Attention Block (Lee et al., ICML 2019).

    Uses m inducing points to reduce O(n^2) attention to O(n*m).
    """
    def __init__(self, d_model, n_heads, n_inducing, dropout=0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, d_model))
        self.attn1 = MultiheadAttentionBlock(d_model, n_heads, dropout)
        self.attn2 = MultiheadAttentionBlock(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B = x.size(0)
        ind = self.inducing.expand(B, -1, -1)
        h = self.attn1(ind, x, key_padding_mask=mask)
        out = self.attn2(x, h)
        return self.norm(out + self.ff(out))


class PMA(nn.Module):
    """
    Pooling by Multihead Attention — maps a set to k seed vectors.
    """
    def __init__(self, d_model, n_heads, n_seeds, dropout=0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model))
        self.attn = MultiheadAttentionBlock(d_model, n_heads, dropout)

    def forward(self, x, mask=None):
        B = x.size(0)
        s = self.seeds.expand(B, -1, -1)
        return self.attn(s, x, key_padding_mask=mask)


class GraphEncoder(nn.Module):
    """
    Encodes a single N x N adjacency + scalar metadata into a
    fixed-size vector.
    """
    def __init__(self, n_nodes, d_model, dropout=0.1):
        super().__init__()
        n_edges = n_nodes * (n_nodes - 1)
        self.proj = nn.Sequential(
            nn.Linear(n_edges + 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, adj, cost, urate):
        """
        Parameters
        ----------
        adj   : [B, N, N]
        cost  : [B]
        urate : [B]
        """
        B, N, _ = adj.shape
        mask = ~torch.eye(N, dtype=torch.bool, device=adj.device)
        flat = adj[:, mask].reshape(B, -1)
        meta = torch.stack([cost, urate], dim=-1)  # [B, 2]
        return self.proj(torch.cat([flat, meta], dim=-1))  # [B, d_model]


class SolutionSetTransformer(nn.Module):
    """
    Two-level architecture:
      1. GraphEncoder (shared) encodes each of k solutions
      2. ISAB + PMA aggregates the set of solution embeddings
      3. MLP classifier on pooled representation
    """
    def __init__(self, n_nodes, d_model=64, n_heads=4, n_inducing=4,
                 dropout=0.3):
        super().__init__()
        self.graph_enc = GraphEncoder(n_nodes, d_model, dropout)
        self.isab = ISAB(d_model, n_heads, n_inducing, dropout)
        self.pma = PMA(d_model, n_heads, n_seeds=1, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, sol_adjs, sol_costs, sol_urate, sol_mask):
        """
        Parameters
        ----------
        sol_adjs  : [B, K, N, N]
        sol_costs : [B, K]
        sol_urate : [B, K]
        sol_mask  : [B, K]  (1=real, 0=pad)
        """
        B, K, N, _ = sol_adjs.shape

        adjs_flat = sol_adjs.reshape(B * K, N, N)
        costs_flat = sol_costs.reshape(B * K)
        urate_flat = sol_urate.reshape(B * K)

        emb = self.graph_enc(adjs_flat, costs_flat, urate_flat)  # [B*K, d]
        emb = emb.reshape(B, K, -1)  # [B, K, d]

        pad_mask = (sol_mask == 0)  # True where padded
        emb = self.isab(emb, mask=pad_mask)  # [B, K, d]
        pooled = self.pma(emb, mask=pad_mask)  # [B, 1, d]
        pooled = pooled.squeeze(1)  # [B, d]

        return self.classifier(pooled)  # [B, 2]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for adjs, costs, urate, mask, labels in loader:
        adjs = adjs.to(DEVICE)
        costs = costs.to(DEVICE)
        urate = urate.to(DEVICE)
        mask = mask.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(adjs, costs, urate, mask)
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
    for adjs, costs, urate, mask, labels in loader:
        adjs = adjs.to(DEVICE)
        costs = costs.to(DEVICE)
        urate = urate.to(DEVICE)
        mask = mask.to(DEVICE)
        logits = model(adjs, costs, urate, mask)
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


def run_cv(sol_adjs, sol_costs, sol_urate, sol_mask, labels, n_nodes, args):
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(sol_adjs, labels)):
        print(f"    Fold {fold + 1}/{args.n_folds} ...", end=" ", flush=True)
        torch.manual_seed(args.seed + fold)

        train_ds = SolutionSetDataset(
            sol_adjs[train_idx], sol_costs[train_idx],
            sol_urate[train_idx], sol_mask[train_idx], labels[train_idx])
        val_ds = SolutionSetDataset(
            sol_adjs[val_idx], sol_costs[val_idx],
            sol_urate[val_idx], sol_mask[val_idx], labels[val_idx])

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

        model = SolutionSetTransformer(
            n_nodes=n_nodes, d_model=args.d_model, n_heads=args.n_heads,
            n_inducing=args.n_inducing, dropout=args.dropout,
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
    root_dir = os.path.join(args.results_root, args.timestamp)
    if not os.path.isdir(root_dir):
        alt = os.path.join("..", "real_data", args.results_root, args.timestamp)
        if os.path.isdir(alt):
            root_dir = alt
        else:
            print(f"Error: {root_dir} not found")
            sys.exit(1)

    configs = args.configs or discover_configs(root_dir)
    out_dir = args.output_dir or os.path.join(root_dir, "ml_classification")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("TIER 3: SOLUTION-SET TRANSFORMER — HC vs SZ CLASSIFICATION")
    print("=" * 80)
    print(f"Timestamp:     {args.timestamp}")
    print(f"Configs:       {configs}")
    print(f"Max solutions: {args.max_solutions}")
    print(f"Device:        {DEVICE}")
    print(f"Model:         d={args.d_model} heads={args.n_heads} "
          f"inducing={args.n_inducing}")
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

        sol_adjs, sol_costs, sol_urate, sol_mask, labels, n_nodes = \
            build_solution_set_data(subjects, args.max_solutions)

        n0 = (labels == 0).sum()
        n1 = (labels == 1).sum()
        avg_sols = sol_mask.sum(axis=1).mean()
        print(f"    {len(subjects)} subjects, N={n_nodes}, "
              f"avg_solutions={avg_sols:.1f}, group0={n0} group1={n1}")

        fold_results = run_cv(
            sol_adjs, sol_costs, sol_urate, sol_mask, labels, n_nodes, args)

        accs = [r["accuracy"] for r in fold_results]
        aucs = [r["auc"] for r in fold_results]
        parts = cfg.split("_")
        row = {
            "config": cfg,
            "classifier": "SolutionSetTransformer",
            "method": parts[2] if len(parts) >= 3 else cfg,
            "n_components": int(parts[0][1:]) if parts[0].startswith("N") else 0,
            "scc_strategy": parts[1] if len(parts) >= 2 else "",
            "n_subjects": len(subjects),
            "avg_solutions": avg_sols,
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
    show_cols = ["config", "classifier", "avg_solutions",
                 "accuracy_mean", "accuracy_std", "auc_mean", "auc_std"]
    show = df[[c for c in show_cols if c in df.columns]].copy()
    for c in ["accuracy_mean", "accuracy_std", "auc_mean", "auc_std"]:
        if c in show.columns:
            show[c] = show[c].round(4)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(show.to_string(index=False))

    csv_path = os.path.join(out_dir, "tier3_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
