#!/usr/bin/env python3
"""
train_roads.py
Train an XGBoost binary classifier on preprocessed road features.

Reads from {project}/roads/preprocessed/:
    X_train.parquet, y_train.parquet
    X_val.parquet,   y_val.parquet
    feature_config.json

Writes to {project}/models_roads/:
    roads_model_{N}.ubj          — XGBoost model (binary format)
    roads_model_{N}_meta.json    — training config, val metrics, feature importances
    roads_model_{N}_val_preds.parquet — val set road_id + y_true + y_proba

Models are auto-numbered starting at 0 (next available index).

Usage:
    python train_roads.py --project project-ma
    python train_roads.py --project project-ma --n-estimators 3000 --max-depth 6
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

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

# XGBoost defaults — all overridable via CLI
XGB_DEFAULTS = dict(
    n_estimators        = 5000,
    learning_rate       = 0.03,
    max_depth           = 5,
    min_child_weight    = 5,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    reg_lambda          = 2.0,
    gamma               = 0.1,
    objective           = "binary:logistic",
    eval_metric         = "aucpr",
    tree_method         = "hist",
    early_stopping_rounds = 200,
    n_jobs              = -1,
)


# ==========================================
#               HELPERS
# ==========================================

def _next_model_index(models_dir: Path) -> int:
    """Return the lowest non-negative integer N for which roads_model_N.ubj doesn't exist."""
    existing = {
        int(p.stem.split("_")[-1])
        for p in models_dir.glob("roads_model_*.ubj")
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


# ==========================================
#                  MAIN
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost on preprocessed road features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_roads.py --project project-ma\n"
            "  python train_roads.py --project project-ma --n-estimators 3000 --max-depth 6\n"
        ),
    )
    parser.add_argument("--project", default=PROJECT,
                        help="Project folder name (default: %(default)s)")
    parser.add_argument("--n-estimators",     type=int,   default=XGB_DEFAULTS["n_estimators"])
    parser.add_argument("--learning-rate",    type=float, default=XGB_DEFAULTS["learning_rate"])
    parser.add_argument("--max-depth",        type=int,   default=XGB_DEFAULTS["max_depth"])
    parser.add_argument("--min-child-weight", type=int,   default=XGB_DEFAULTS["min_child_weight"])
    parser.add_argument("--subsample",        type=float, default=XGB_DEFAULTS["subsample"])
    parser.add_argument("--colsample-bytree", type=float, default=XGB_DEFAULTS["colsample_bytree"])
    parser.add_argument("--reg-lambda",       type=float, default=XGB_DEFAULTS["reg_lambda"])
    parser.add_argument("--gamma",            type=float, default=XGB_DEFAULTS["gamma"])
    args = parser.parse_args()

    project_dir = BASE_DIR / args.project
    if not project_dir.exists():
        print(f"[ERROR] Project folder not found: {project_dir}")
        return

    prep_dir   = project_dir / "roads" / "preprocessed"
    models_dir = project_dir / "models_roads"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_idx  = _next_model_index(models_dir)
    model_stem = f"roads_model_{model_idx}"

    print(f"\n{'='*60}")
    print(f" Train Roads — {args.project}  →  {model_stem}")
    print(f"{'='*60}")

    # ---- Load preprocessed splits ----
    print(f"\nLoading from {prep_dir} …")
    X_train = _load_parquet(prep_dir / "X_train.parquet",    "train features")
    y_train = _load_parquet(prep_dir / "y_train.parquet",    "train labels")
    X_val   = _load_parquet(prep_dir / "X_val.parquet",      "val features")
    y_val   = _load_parquet(prep_dir / "y_val.parquet",      "val labels")

    if any(x is None for x in [X_train, y_train, X_val, y_val]):
        print("[ERROR] Missing preprocessed splits — run preprocess-roads.py first.")
        return

    # Squeeze label DataFrames to Series
    y_train_s = y_train.iloc[:, 0].astype(int)
    y_val_s   = y_val.iloc[:, 0].astype(int)

    # Load meta for val road_ids (optional — for output parquet)
    meta_val = _load_parquet(prep_dir / "meta_val.parquet", "val meta")

    # Load feature config for provenance
    cfg_path = prep_dir / "feature_config.json"
    feature_config: dict = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    # ---- Class balance ----
    n_pos = int((y_train_s == 1).sum())
    n_neg = int((y_train_s == 0).sum())
    scale_pos_weight = round(n_neg / max(n_pos, 1), 4)
    print(f"\n  Train positives: {n_pos:,}  negatives: {n_neg:,}  scale_pos_weight: {scale_pos_weight}")
    print(f"  Val   positives: {int((y_val_s==1).sum()):,}  negatives: {int((y_val_s==0).sum()):,}")

    # ---- Build model ----
    params = dict(
        n_estimators        = args.n_estimators,
        learning_rate       = args.learning_rate,
        max_depth           = args.max_depth,
        min_child_weight    = args.min_child_weight,
        subsample           = args.subsample,
        colsample_bytree    = args.colsample_bytree,
        reg_lambda          = args.reg_lambda,
        gamma               = args.gamma,
        objective           = XGB_DEFAULTS["objective"],
        eval_metric         = XGB_DEFAULTS["eval_metric"],
        tree_method         = XGB_DEFAULTS["tree_method"],
        early_stopping_rounds = XGB_DEFAULTS["early_stopping_rounds"],
        n_jobs              = XGB_DEFAULTS["n_jobs"],
        scale_pos_weight    = scale_pos_weight,
        random_state        = RANDOM_STATE,
    )

    model = XGBClassifier(**params)

    print(f"\nTraining XGBoost (max {args.n_estimators} trees, early stop {XGB_DEFAULTS['early_stopping_rounds']}) …")
    model.fit(
        X_train, y_train_s,
        eval_set=[(X_val, y_val_s)],
        verbose=100,
    )

    best_iter = int(getattr(model, "best_iteration", args.n_estimators))
    print(f"\n  Best iteration: {best_iter}")

    # ---- Val metrics ----
    val_proba = model.predict_proba(X_val)[:, 1].astype(np.float32)
    roc_auc   = float(roc_auc_score(y_val_s, val_proba))
    pr_auc    = float(average_precision_score(y_val_s, val_proba))

    print(f"\n  Val ROC-AUC : {roc_auc:.4f}")
    print(f"  Val PR-AUC  : {pr_auc:.4f}")

    # ---- Save model ----
    model_path = models_dir / f"{model_stem}.ubj"
    model.save_model(str(model_path))
    print(f"\n  Saved model → {model_path}")

    # ---- Save val predictions ----
    val_preds = pd.DataFrame({
        "y_true":  y_val_s.values,
        "y_proba": val_proba,
    })
    if meta_val is not None and "road_id" in meta_val.columns:
        val_preds.insert(0, "road_id", meta_val["road_id"].values)
    val_preds_path = models_dir / f"{model_stem}_val_preds.parquet"
    val_preds.to_parquet(val_preds_path, index=False)
    print(f"  Saved val predictions → {val_preds_path}")

    # ---- Feature importances ----
    importances = model.get_booster().get_score(importance_type="gain")
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:30]

    # ---- Save meta ----
    meta: dict = {
        "model_index":        model_idx,
        "project":            args.project,
        "state":              feature_config.get("state"),
        "model_file":         model_path.name,
        "best_iteration":     best_iter,
        "val_roc_auc":        roc_auc,
        "val_pr_auc":         pr_auc,
        "n_train":            len(X_train),
        "n_val":              len(X_val),
        "n_features":         X_train.shape[1],
        "positive_rate_train": round(n_pos / (n_pos + n_neg), 4),
        "positive_rate_val":  round(float(y_val_s.mean()), 4),
        "scale_pos_weight":   scale_pos_weight,
        "xgb_params":         {k: v for k, v in params.items() if k != "scale_pos_weight"},
        "feature_cols":       feature_config.get("feature_cols", list(X_train.columns)),
        "top30_feature_importance_gain": [{"feature": f, "gain": round(g, 4)} for f, g in top_feats],
        "preprocess_config":  {
            k: feature_config.get(k)
            for k in ("apply_log", "apply_scale", "clip_quantiles", "val_fraction", "random_state")
        },
    }
    meta_path = models_dir / f"{model_stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved meta       → {meta_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f" {model_stem}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"{'='*60}")
    print(f"\n  Top 10 features by gain:")
    for feat, gain in top_feats[:10]:
        bar = "█" * int(min(gain / max(importances.values()), 1.0) * 30)
        print(f"    {feat:<45s}  {gain:>10.1f}  {bar}")


if __name__ == "__main__":
    main()
