#!/usr/bin/env python3
"""
train_edges.py
Train an XGBoost binary classifier on preprocessed edge features
using spatial K-fold cross-validation.

Spatial folds: edges are divided into N latitude strips using quantile
bins on centroid_lat (from meta_all).  Each fold's validation set is one
contiguous geographic strip; training uses the remaining K-1 strips.
This avoids leakage between spatially autocorrelated neighbouring edges.

Reads from {project}/edges/preprocessed/:
    X_all.parquet, y_all.parquet, meta_all.parquet, feature_config.json

Writes to {project}/models_edges/:
    edges_model_{N}.ubj                  — full model (all training data)
    edges_model_{N}_meta.json            — training config, OOF metrics, importances
    edges_model_{N}_oof_preds.parquet    — edge_id + fold + y_true + y_proba (OOF)
    edges_model_{N}_fold_{k}.ubj         — per-fold models

Usage:
    python train_edges.py --project project-ma
    python train_edges.py --project project-ma --n-folds 5 --max-depth 4
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from generate.generate_utils import BASE_DIR

warnings.filterwarnings("ignore")


# ==========================================
#            CONFIGURATION
# ==========================================

PROJECT      = "project-ma"
RANDOM_STATE = 42
N_FOLDS      = 8

XGB_DEFAULTS = dict(
    n_estimators          = 5000,
    learning_rate         = 0.02,
    max_depth             = 3,
    min_child_weight      = 7,
    subsample             = 0.7,
    colsample_bytree      = 0.5,
    reg_lambda            = 5.0,
    gamma                 = 0.5,
    objective             = "binary:logistic",
    eval_metric           = "aucpr",
    tree_method           = "hist",
    early_stopping_rounds = 150,
    n_jobs                = -1,
)


# ==========================================
#               HELPERS
# ==========================================

def _next_model_index(models_dir: Path) -> int:
    """Return the lowest non-negative integer N for which edges_model_N.ubj doesn't exist."""
    existing = {
        int(p.stem.split("_")[-1])
        for p in models_dir.glob("edges_model_*.ubj")
        if p.stem.split("_")[-1].isdigit()
    }
    n = 0
    while n in existing:
        n += 1
    return n


def _load_parquet(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [not found] {path.name} — {label}")
        return None
    df = pd.read_parquet(path)
    print(f"  {path.name}: {len(df):,} rows, {df.shape[1]} cols")
    return df


def assign_spatial_folds(meta: pd.DataFrame, n_folds: int) -> np.ndarray:
    """
    Assign each edge to a spatial fold index in [0, n_folds).

    Strategy: quantile-bin centroid_lat into n_folds equal-population strips.
    Fold k's validation set = edges whose centroid falls in the k-th lat band.
    Quantile binning ensures roughly balanced fold sizes even when edge density
    varies by latitude.

    Requires meta to have a 'centroid_lat' column, written by preprocess_edges.py.
    """
    if "centroid_lat" not in meta.columns:
        raise ValueError(
            "meta_all must contain 'centroid_lat' for spatial fold assignment.\n"
            "  → Re-run preprocess_edges.py to regenerate meta_all.parquet."
        )
    lat = meta["centroid_lat"].values
    bins = np.quantile(lat, np.linspace(0, 1, n_folds + 1))
    bins[0]  -= 1e-9
    bins[-1] += 1e-9
    return np.clip(np.digitize(lat, bins) - 1, 0, n_folds - 1)


# ==========================================
#                  MAIN
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost on edge features using spatial K-fold CV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_edges.py --project project-ma\n"
            "  python train_edges.py --project project-ma --n-folds 5 --max-depth 4\n"
        ),
    )
    parser.add_argument("--project",          default=PROJECT,
                        help="Project folder name (default: %(default)s)")
    parser.add_argument("--n-estimators",     type=int,   default=XGB_DEFAULTS["n_estimators"])
    parser.add_argument("--learning-rate",    type=float, default=XGB_DEFAULTS["learning_rate"])
    parser.add_argument("--max-depth",        type=int,   default=XGB_DEFAULTS["max_depth"])
    parser.add_argument("--min-child-weight", type=int,   default=XGB_DEFAULTS["min_child_weight"])
    parser.add_argument("--subsample",        type=float, default=XGB_DEFAULTS["subsample"])
    parser.add_argument("--colsample-bytree", type=float, default=XGB_DEFAULTS["colsample_bytree"])
    parser.add_argument("--reg-lambda",       type=float, default=XGB_DEFAULTS["reg_lambda"])
    parser.add_argument("--gamma",            type=float, default=XGB_DEFAULTS["gamma"])
    parser.add_argument("--n-folds",          type=int,   default=N_FOLDS,
                        help="Number of spatial CV folds (default: %(default)s)")
    args = parser.parse_args()

    project_dir = BASE_DIR / args.project
    if not project_dir.exists():
        print(f"[ERROR] Project folder not found: {project_dir}")
        return

    prep_dir   = project_dir / "edges" / "preprocessed"
    models_dir = project_dir / "models_edges"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_idx  = _next_model_index(models_dir)
    model_stem = f"edges_model_{model_idx}"

    print(f"\n{'='*60}")
    print(f" Train Edges — {args.project}  →  {model_stem}")
    print(f"{'='*60}")

    # ---- Load data ----
    print(f"\nLoading from {prep_dir} …")
    X_all_df = _load_parquet(prep_dir / "X_all.parquet",    "all features")
    y_all_df = _load_parquet(prep_dir / "y_all.parquet",    "all labels")
    meta_all = _load_parquet(prep_dir / "meta_all.parquet", "all meta")

    if any(x is None for x in [X_all_df, y_all_df, meta_all]):
        print("[ERROR] Missing preprocessed files — run preprocess_edges.py first.")
        return

    X_all   = X_all_df
    y_all_s = y_all_df.iloc[:, 0].astype(int)

    cfg_path = prep_dir / "feature_config.json"
    feature_config: dict = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    # ---- Class balance ----
    n_pos = int((y_all_s == 1).sum())
    n_neg = int((y_all_s == 0).sum())
    scale_pos_weight = round(n_neg / max(n_pos, 1), 4)
    print(f"\n  Positives: {n_pos:,}  negatives: {n_neg:,}  scale_pos_weight: {scale_pos_weight}")

    # ---- Assign spatial folds ----
    fold_ids = assign_spatial_folds(meta_all, args.n_folds)
    unique, counts = np.unique(fold_ids, return_counts=True)
    print(f"\n  Spatial folds ({args.n_folds} lat strips):")
    for fid, cnt in zip(unique, counts):
        mask = fold_ids == fid
        lat_lo = float(meta_all["centroid_lat"].values[mask].min())
        lat_hi = float(meta_all["centroid_lat"].values[mask].max())
        n_pos_f = int((y_all_s.values[mask] == 1).sum())
        print(f"    fold {fid}: {cnt:>6,} edges  lat [{lat_lo:.3f}, {lat_hi:.3f}]  pos={n_pos_f:,}")

    # ---- Shared XGBoost params ----
    params = dict(
        n_estimators          = args.n_estimators,
        learning_rate         = args.learning_rate,
        max_depth             = args.max_depth,
        min_child_weight      = args.min_child_weight,
        subsample             = args.subsample,
        colsample_bytree      = args.colsample_bytree,
        reg_lambda            = args.reg_lambda,
        gamma                 = args.gamma,
        objective             = XGB_DEFAULTS["objective"],
        eval_metric           = XGB_DEFAULTS["eval_metric"],
        tree_method           = XGB_DEFAULTS["tree_method"],
        early_stopping_rounds = XGB_DEFAULTS["early_stopping_rounds"],
        n_jobs                = XGB_DEFAULTS["n_jobs"],
        scale_pos_weight      = scale_pos_weight,
        random_state          = RANDOM_STATE,
    )

    # ==================================================================
    # SPATIAL K-FOLD — train on K-1 lat strips, validate on 1 lat strip
    # ==================================================================
    print(f"\n{'─'*60}")
    print(f" Spatial K-Fold ({args.n_folds} folds) — OOF predictions")
    print(f"{'─'*60}")

    oof_records: list[pd.DataFrame] = []
    fold_model_files: list[str] = []
    fold_best_iters: list[int] = []

    for fold_k in range(args.n_folds):
        tr_idx  = np.where(fold_ids != fold_k)[0]
        val_idx = np.where(fold_ids == fold_k)[0]

        print(f"\n  [fold {fold_k + 1}/{args.n_folds}]  "
              f"train={len(tr_idx):,}  val={len(val_idx):,}")

        Xf_tr  = X_all.iloc[tr_idx]
        yf_tr  = y_all_s.iloc[tr_idx]
        Xf_val = X_all.iloc[val_idx]
        yf_val = y_all_s.iloc[val_idx]

        fold_model = XGBClassifier(**params)
        fold_model.fit(
            Xf_tr, yf_tr,
            eval_set=[(Xf_val, yf_val)],
            verbose=100,
        )

        fold_proba = fold_model.predict_proba(Xf_val)[:, 1].astype(np.float32)
        fold_roc   = float(roc_auc_score(yf_val, fold_proba))
        fold_pr    = float(average_precision_score(yf_val, fold_proba))
        print(f"    ROC-AUC={fold_roc:.4f}  PR-AUC={fold_pr:.4f}  "
              f"best_iter={getattr(fold_model, 'best_iteration', '?')}")

        rec = pd.DataFrame({
            "fold":    fold_k,
            "y_true":  yf_val.values,
            "y_proba": fold_proba,
        }, index=Xf_val.index)
        if "edge_id" in meta_all.columns:
            rec.insert(0, "edge_id", meta_all["edge_id"].iloc[val_idx].values)

        oof_records.append(rec)
        fold_best_iters.append(int(getattr(fold_model, "best_iteration", args.n_estimators)))

        fold_fname = f"{model_stem}_fold_{fold_k}.ubj"
        fold_model.save_model(str(models_dir / fold_fname))
        fold_model_files.append(fold_fname)
        print(f"    Saved → {fold_fname}")

    oof_df  = pd.concat(oof_records).sort_index().reset_index(drop=True)
    oof_roc = float(roc_auc_score(oof_df["y_true"], oof_df["y_proba"]))
    oof_pr  = float(average_precision_score(oof_df["y_true"], oof_df["y_proba"]))
    print(f"\n  OOF ROC-AUC : {oof_roc:.4f}")
    print(f"  OOF PR-AUC  : {oof_pr:.4f}")

    oof_path = models_dir / f"{model_stem}_oof_preds.parquet"
    oof_df.to_parquet(oof_path, index=False)
    print(f"  Saved OOF predictions → {oof_path}")

    # ==================================================================
    # FULL MODEL — all data, n_estimators = mean of fold best_iters
    # ==================================================================
    mean_best_iter = int(round(sum(fold_best_iters) / len(fold_best_iters)))
    print(f"\n{'─'*60}")
    print(f" Full model — all {len(X_all):,} edges  "
          f"(n_estimators={mean_best_iter} from fold mean)")
    print(f"{'─'*60}")

    full_params = {**params, "n_estimators": mean_best_iter, "early_stopping_rounds": None}
    full_model  = XGBClassifier(**full_params)
    full_model.fit(X_all, y_all_s, verbose=100)

    model_path = models_dir / f"{model_stem}.ubj"
    full_model.save_model(str(model_path))
    print(f"\n  Saved full model → {model_path}")

    importances = full_model.get_booster().get_score(importance_type="gain")
    top_feats   = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:30]

    # ---- Save meta ----
    meta_out: dict = {
        "model_index":        model_idx,
        "project":            args.project,
        "state":              feature_config.get("state"),
        "model_file":         model_path.name,
        "n_estimators_full":  mean_best_iter,
        "fold_best_iters":    fold_best_iters,
        "oof_roc_auc":        oof_roc,
        "oof_pr_auc":         oof_pr,
        "n_folds":            args.n_folds,
        "fold_type":          "spatial_lat_strips",
        "fold_model_files":   fold_model_files,
        "n_all":              len(X_all),
        "n_features":         X_all.shape[1],
        "positive_rate_all":  round(n_pos / (n_pos + n_neg), 4),
        "scale_pos_weight":   scale_pos_weight,
        "xgb_params":         {k: v for k, v in params.items()
                               if k not in ("scale_pos_weight", "early_stopping_rounds")},
        "feature_cols":       feature_config.get("feature_cols", list(X_all.columns)),
        "top30_feature_importance_gain": [
            {"feature": f, "gain": round(g, 4)} for f, g in top_feats
        ],
        "preprocess_config":  {
            k: feature_config.get(k)
            for k in ("apply_log", "apply_scale", "clip_quantiles", "random_state")
        },
    }
    meta_path = models_dir / f"{model_stem}_meta.json"
    meta_path.write_text(json.dumps(meta_out, indent=2))
    print(f"  Saved meta → {meta_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f" {model_stem}  OOF ROC-AUC={oof_roc:.4f}  OOF PR-AUC={oof_pr:.4f}")
    print(f"{'='*60}")
    print(f"\n  Top 30 features by gain:")
    for feat, gain in top_feats[:30]:
        bar = "█" * int(min(gain / max(importances.values()), 1.0) * 30)
        print(f"    {feat:<45s}  {gain:>10.1f}  {bar}")


if __name__ == "__main__":
    main()
