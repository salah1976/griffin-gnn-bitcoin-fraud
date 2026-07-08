"""
GriffinGNN — Entraînement full-batch (HeteroConv / SAGEConv)
Split chronologique, scaling sans data leakage, focal loss pondérée.
"""

import os
import random
import sqlite3

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


# ============================================================
# Config
# ============================================================

SEED = 42
DB_PATH = "/bitcoin_fraud.db"
CSV_PATH = "/bitcoin_processed_43features.csv"
OUT_PATH = "/gnn_v2_preds.pt"

TRAIN_FRAC = 0.64
VAL_FRAC = 0.80  # cumulatif -> val = [0.64, 0.80]

HIDDEN_DIM = 128
N_LAYERS = 3
DROPOUT = 0.10
LR = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 2.0
NUM_EPOCHS = 150
PATIENCE = 20
THRESHOLD_GRID = np.linspace(0.30, 0.99, 350)

EXCLUDE_NUMERIC = [
    "tx_id", "block_id", "label_any",
    "label_verified", "label_not_verified", "heuristic_score",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Chargement des données
# ============================================================

def load_raw_tables(db_path: str, csv_path: str):
    conn = sqlite3.connect(db_path)
    transactions = pd.read_sql_query(
        "SELECT id AS tx_id, tx_hash, block_id, coinbase_flag FROM transactions", conn
    )
    inputs_df = pd.read_sql_query(
        "SELECT tx_id, address_id FROM transaction_inputs", conn
    )
    outputs_df = pd.read_sql_query(
        "SELECT tx_id, address_id FROM transaction_outputs", conn
    )
    addresses = pd.read_sql_query(
        "SELECT id AS address_id, address FROM addresses", conn
    )
    conn.close()

    features_df = pd.read_csv(csv_path)
    for col in ("tx_hash", "label_any"):
        assert col in features_df.columns, f"Missing column: {col}"

    return transactions, inputs_df, outputs_df, addresses, features_df


def merge_and_clean(transactions, features_df):
    tx = transactions.merge(features_df, on="tx_hash", how="inner")
    tx = tx.dropna(subset=["label_any"]).copy()
    tx["label_any"] = tx["label_any"].astype(int)
    tx = tx.drop(columns=["timestamp"], errors="ignore")
    return tx


def get_feature_cols(tx_merged: pd.DataFrame):
    numeric_cols = tx_merged.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in EXCLUDE_NUMERIC]


# ============================================================
# Split chronologique + scaling (fit sur train uniquement)
# ============================================================

def chronological_split(tx_merged: pd.DataFrame):
    tx_merged = tx_merged.sort_values("block_id").reset_index(drop=True)
    n = len(tx_merged)
    train_end = int(TRAIN_FRAC * n)
    val_end = int(VAL_FRAC * n)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        sub = tx_merged.iloc[idx]
        print(
            f"{name}: blocks {sub['block_id'].min()}->{sub['block_id'].max()} | "
            f"{len(sub)} tx | fraud={sub['label_any'].sum()} "
            f"({100 * sub['label_any'].mean():.2f}%)"
        )

    tx_merged["tx_idx"] = np.arange(n)
    return tx_merged, train_idx, val_idx, test_idx


def build_masks(n_nodes, train_idx, val_idx, test_idx):
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def scale_tx_features(X_raw, train_idx, val_idx, test_idx):
    scaler = StandardScaler()
    X_scaled = np.zeros_like(X_raw, dtype=np.float32)
    X_scaled[train_idx] = scaler.fit_transform(X_raw[train_idx])
    X_scaled[val_idx] = scaler.transform(X_raw[val_idx])
    X_scaled[test_idx] = scaler.transform(X_raw[test_idx])
    return X_scaled.astype(np.float32), scaler


# ============================================================
# Construction du graphe hétérogène
# ============================================================

def build_edges(inputs_df, outputs_df, tx_id_to_idx, addr_id_to_idx):
    unique_tx_ids = set(tx_id_to_idx.keys())
    unique_addr_ids = set(addr_id_to_idx.keys())

    inputs_mapped = inputs_df[
        inputs_df["tx_id"].isin(unique_tx_ids)
        & inputs_df["address_id"].isin(unique_addr_ids)
    ].copy()
    outputs_mapped = outputs_df[
        outputs_df["tx_id"].isin(unique_tx_ids)
        & outputs_df["address_id"].isin(unique_addr_ids)
    ].copy()

    inputs_mapped["tx_idx"] = inputs_mapped["tx_id"].map(tx_id_to_idx)
    inputs_mapped["addr_idx"] = inputs_mapped["address_id"].map(addr_id_to_idx)
    outputs_mapped["tx_idx"] = outputs_mapped["tx_id"].map(tx_id_to_idx)
    outputs_mapped["addr_idx"] = outputs_mapped["address_id"].map(addr_id_to_idx)

    inputs_mapped = inputs_mapped.dropna(subset=["tx_idx", "addr_idx"]).copy()
    outputs_mapped = outputs_mapped.dropna(subset=["tx_idx", "addr_idx"]).copy()
    inputs_mapped[["tx_idx", "addr_idx"]] = inputs_mapped[["tx_idx", "addr_idx"]].astype(np.int64)
    outputs_mapped[["tx_idx", "addr_idx"]] = outputs_mapped[["tx_idx", "addr_idx"]].astype(np.int64)

    edge_addr_to_tx = torch.from_numpy(
        np.vstack([inputs_mapped["addr_idx"].values, inputs_mapped["tx_idx"].values]).astype(np.int64)
    )
    edge_tx_to_addr = torch.from_numpy(
        np.vstack([outputs_mapped["tx_idx"].values, outputs_mapped["addr_idx"].values]).astype(np.int64)
    )
    return inputs_mapped, outputs_mapped, edge_addr_to_tx, edge_tx_to_addr


def build_address_features(inputs_mapped, outputs_mapped, train_idx, num_addr_nodes):
    train_tx_set = set(train_idx.tolist())

    inputs_train = inputs_mapped[inputs_mapped["tx_idx"].isin(train_tx_set)].copy()
    outputs_train = outputs_mapped[outputs_mapped["tx_idx"].isin(train_tx_set)].copy()

    addr_in_train = inputs_train["addr_idx"].values
    addr_out_train = outputs_train["addr_idx"].values

    in_deg = np.bincount(addr_in_train, minlength=num_addr_nodes).astype(np.float32)
    out_deg = np.bincount(addr_out_train, minlength=num_addr_nodes).astype(np.float32)
    tot_deg = in_deg + out_deg

    in_out_ratio = (in_deg + 1.0) / (out_deg + 1.0)
    active_both = ((in_deg > 0) & (out_deg > 0)).astype(np.float32)

    addr_tx_train = pd.concat([
        inputs_train[["addr_idx", "tx_idx"]],
        outputs_train[["addr_idx", "tx_idx"]],
    ])

    addr_unique_tx = (
        addr_tx_train.groupby("addr_idx")["tx_idx"].nunique()
        .reindex(range(num_addr_nodes), fill_value=0).values.astype(np.float32)
    )
    addr_unique_nb = (
        addr_tx_train.groupby("addr_idx")["tx_idx"].count()
        .reindex(range(num_addr_nodes), fill_value=0).values.astype(np.float32)
    )
    activity_ratio = (tot_deg + 1) / (addr_unique_tx + 1)

    addr_x_raw = np.vstack([
        np.log1p(in_deg),
        np.log1p(out_deg),
        np.log1p(tot_deg),
        np.log1p(in_out_ratio),
        active_both,
        np.log1p(addr_unique_tx),
        np.log1p(addr_unique_nb),
        np.log1p(activity_ratio),
    ]).T.astype(np.float32)

    train_addr_idx = np.unique(np.concatenate([addr_in_train, addr_out_train])).astype(np.int64)

    addr_scaler = StandardScaler()
    addr_scaler.fit(addr_x_raw[train_addr_idx])
    addr_x = addr_scaler.transform(addr_x_raw).astype(np.float32)

    print("Address feature dim:", addr_x.shape[1])
    print("Train addr count:", len(train_addr_idx))
    return addr_x


# ============================================================
# Modèle
# ============================================================

class GriffinGNNv2(nn.Module):
    def __init__(self, tx_in_dim, addr_in_dim, hidden_dim=128, out_dim=2, dropout=0.10, n_layers=3):
        super().__init__()

        self.tx_proj = nn.Sequential(
            nn.Linear(tx_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.addr_proj = nn.Sequential(
            nn.Linear(addr_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList([
            HeteroConv({
                ("address", "to_tx", "transaction"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("transaction", "to_address", "address"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            }, aggr="sum")
            for _ in range(n_layers)
        ])

        self.norms_tx = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.norms_addr = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x_dict = data.x_dict
        edge_dict = data.edge_index_dict

        h_tx = self.tx_proj(x_dict["transaction"])
        h_addr = self.addr_proj(x_dict["address"])

        for i, conv in enumerate(self.convs):
            x_in = {"transaction": h_tx, "address": h_addr}
            x_new = conv(x_in, edge_dict)

            h_tx_new = x_new.get("transaction", h_tx)
            h_addr_new = x_new.get("address", h_addr)

            h_tx = F.dropout(
                F.relu(self.norms_tx[i](h_tx_new + h_tx)), p=self.dropout, training=self.training
            )
            h_addr = F.dropout(
                F.relu(self.norms_addr[i](h_addr_new + h_addr)), p=self.dropout, training=self.training
            )

        return self.classifier(h_tx)


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ============================================================
# Entraînement / évaluation
# ============================================================

def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    mask = data["transaction"].train_mask
    loss = criterion(out[mask], data["transaction"].y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split="val", threshold=None):
    model.eval()
    out = model(data)
    y = data["transaction"].y

    mask = {
        "val": data["transaction"].val_mask,
        "test": data["transaction"].test_mask,
    }.get(split, data["transaction"].train_mask)

    logits = out[mask]
    probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    true = y[mask].cpu().numpy()
    preds = (probs >= (threshold if threshold else 0.5)).astype(int)

    auc = roc_auc_score(true, probs) if len(np.unique(true)) > 1 else float("nan")
    ap = average_precision_score(true, probs) if len(np.unique(true)) > 1 else float("nan")
    f1_fraud = f1_score(true, preds, pos_label=1, zero_division=0)
    p_fraud = precision_score(true, preds, pos_label=1, zero_division=0)
    r_fraud = recall_score(true, preds, pos_label=1, zero_division=0)

    return auc, ap, f1_fraud, p_fraud, r_fraud, probs, true


@torch.no_grad()
def find_best_threshold_val(model, data):
    """Cherche le seuil (grille THRESHOLD_GRID) qui maximise le F1-fraud sur validation."""
    _, _, _, _, _, probs, true = evaluate(model, data, "val")
    best_th, best_f1 = 0.50, 0.0
    for th in THRESHOLD_GRID:
        preds = (probs >= th).astype(int)
        f1 = f1_score(true, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1


def fit(model, data, optimizer, criterion, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    best_state = None
    best_val_ap = -1.0
    left = patience

    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(model, data, optimizer, criterion)
        val_auc, val_ap, val_f1, vp, vr, _, _ = evaluate(model, data, "val")

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[Epoch {epoch:03d}] Loss={loss:.4f} | Val AUC={val_auc:.4f} | "
                f"Val AP={val_ap:.4f} | Val F1={val_f1:.4f} (P={vp:.3f} R={vr:.3f})"
            )

        if val_ap > best_val_ap + 1e-5:
            best_val_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            left = patience
        else:
            left -= 1
            if left <= 0:
                print(f"Early stopping epoch {epoch} (best Val AP={best_val_ap:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    return model


# ============================================================
# Diagnostics de fuite structurelle (exposed vs clean)
# ============================================================

def exposed_vs_clean_report(model, data, inputs_mapped, outputs_mapped, train_idx, test_idx, y_tx, best_th):
    train_fraud_set = set(train_idx[y_tx[train_idx] == 1].tolist())

    fraud_addr_set = set(
        inputs_mapped[inputs_mapped["tx_idx"].isin(train_fraud_set)]["addr_idx"].tolist()
        + outputs_mapped[outputs_mapped["tx_idx"].isin(train_fraud_set)]["addr_idx"].tolist()
    )

    test_idx_set = set(test_idx.tolist())

    exposed_test = set(
        inputs_mapped[
            inputs_mapped["tx_idx"].isin(test_idx_set) & inputs_mapped["addr_idx"].isin(fraud_addr_set)
        ]["tx_idx"].tolist()
        + outputs_mapped[
            outputs_mapped["tx_idx"].isin(test_idx_set) & outputs_mapped["addr_idx"].isin(fraud_addr_set)
        ]["tx_idx"].tolist()
    )
    clean_test = [i for i in test_idx if i not in exposed_test]
    exposed_test = list(exposed_test)

    model.eval()
    with torch.no_grad():
        out = model(data)
        probs_all = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

    if exposed_test:
        ep, et = probs_all[exposed_test], y_tx[exposed_test]
        print("\n=== EXPOSED (connecté aux fraudes train) ===")
        print(f"N={len(exposed_test)} | fraud={et.sum()} ({100 * et.mean():.2f}%)")
        print(f"AUC-PR = {average_precision_score(et, ep):.4f}")
        print(f"F1     = {f1_score(et, (ep >= best_th).astype(int), zero_division=0):.4f}")

    if clean_test:
        cp, ct = probs_all[clean_test], y_tx[clean_test]
        print("\n=== CLEAN (isolé des fraudes train) ===")
        print(f"N={len(clean_test)} | fraud={ct.sum()} ({100 * ct.mean():.2f}%)")
        if ct.sum() > 0:
            print(f"AUC-PR = {average_precision_score(ct, cp):.4f}")
            print(f"F1     = {f1_score(ct, (cp >= best_th).astype(int), zero_division=0):.4f}")
        else:
            print("Aucune fraude dans le subset clean")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    transactions, inputs_df, outputs_df, addresses, features_df = load_raw_tables(DB_PATH, CSV_PATH)
    tx_merged = merge_and_clean(transactions, features_df)
    feature_cols = get_feature_cols(tx_merged)

    tx_merged, train_idx, val_idx, test_idx = chronological_split(tx_merged)
    y_tx = tx_merged["label_any"].values.astype(np.int64)
    num_tx_nodes = len(tx_merged)

    tx_id_to_idx = {tid: i for i, tid in enumerate(tx_merged["tx_id"].values)}
    unique_addr_ids = addresses["address_id"].unique()
    addr_id_to_idx = {aid: i for i, aid in enumerate(unique_addr_ids)}
    num_addr_nodes = len(unique_addr_ids)

    X_tx_raw = tx_merged[feature_cols].fillna(0.0).values.astype(np.float32)
    X_tx, _ = scale_tx_features(X_tx_raw, train_idx, val_idx, test_idx)

    train_mask, val_mask, test_mask = build_masks(num_tx_nodes, train_idx, val_idx, test_idx)

    inputs_mapped, outputs_mapped, edge_addr_to_tx, edge_tx_to_addr = build_edges(
        inputs_df, outputs_df, tx_id_to_idx, addr_id_to_idx
    )
    addr_x = build_address_features(inputs_mapped, outputs_mapped, train_idx, num_addr_nodes)

    data = HeteroData()
    data["transaction"].x = torch.from_numpy(X_tx)
    data["transaction"].y = torch.from_numpy(y_tx)
    data["address"].x = torch.from_numpy(addr_x)
    data["address", "to_tx", "transaction"].edge_index = edge_addr_to_tx
    data["transaction", "to_address", "address"].edge_index = edge_tx_to_addr
    data["transaction"].train_mask = train_mask
    data["transaction"].val_mask = val_mask
    data["transaction"].test_mask = test_mask
    data = data.to(DEVICE)

    model = GriffinGNNv2(
        tx_in_dim=data["transaction"].x.shape[1],
        addr_in_dim=data["address"].x.shape[1],
        hidden_dim=HIDDEN_DIM,
        out_dim=2,
        dropout=DROPOUT,
        n_layers=N_LAYERS,
    ).to(DEVICE)

    y_train_np = data["transaction"].y[data["transaction"].train_mask].cpu().numpy()
    class_counts = np.bincount(y_train_np)
    class_weights = class_counts.sum() / (2.0 * class_counts + 1e-8)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    print("Class weights:", class_weights)

    criterion = WeightedFocalLoss(class_weights=class_weights, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model = fit(model, data, optimizer, criterion)

    test_auc, test_ap, test_f1, tp, tr, test_probs, test_true = evaluate(model, data, "test")
    print("\n=== TEST (threshold=0.5) ===")
    print(f"AUC={test_auc:.4f} | AP={test_ap:.4f} | F1_fraud={test_f1:.4f} (P={tp:.3f}, R={tr:.3f})")

    best_th, best_val_f1_th = find_best_threshold_val(model, data)
    print(f"\nBest threshold on VAL : {best_th:.3f} (Val F1={best_val_f1_th:.4f})")

    test_auc2, test_ap2, test_f12, tp2, tr2, _, _ = evaluate(model, data, "test", threshold=best_th)
    print("\n=== TEST (threshold optimized) ===")
    print(f"AUC={test_auc2:.4f} | AP={test_ap2:.4f} | F1_fraud={test_f12:.4f} (P={tp2:.3f}, R={tr2:.3f})")

    preds_opt = (test_probs >= best_th).astype(int)
    print(f"\nClassification report (TEST, p >= {best_th:.3f}):")
    print(classification_report(test_true, preds_opt, digits=4, zero_division=0))

    if "label_verified" in tx_merged.columns:
        verified_mask = tx_merged["label_verified"].fillna(0).astype(int).values
        test_verified_idx = [i for i in test_idx if verified_mask[i] == 1]
        test_legit_sample = [i for i in test_idx if y_tx[i] == 0][:5000]
        eval_idx = test_verified_idx + test_legit_sample

        if test_verified_idx:
            model.eval()
            with torch.no_grad():
                out = model(data)
                probs_all = F.softmax(out, dim=-1)[:, 1].cpu().numpy()
            vp, vt = probs_all[eval_idx], y_tx[eval_idx]
            print("\n=== label_verified uniquement ===")
            print(f"N verified fraud = {len(test_verified_idx)}")
            print(f"AUC-PR = {average_precision_score(vt, vp):.4f}")
            print(f"F1     = {f1_score(vt, (vp >= best_th).astype(int), zero_division=0):.4f}")

    exposed_vs_clean_report(model, data, inputs_mapped, outputs_mapped, train_idx, test_idx, y_tx, best_th)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save({"probs": test_probs, "y_true": test_true, "threshold": best_th}, OUT_PATH)
    print(f"\nPrédictions sauvegardées -> {OUT_PATH}")


if __name__ == "__main__":
    main()
