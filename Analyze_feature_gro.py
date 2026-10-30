# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_FEATURES: List[str] = [
    "delta_S", "delta_H", "T_melting", "omiga", "VEC", "e/a", "Ec", "wf_sixsq",
    "r_ave", "delta_r", "chi_ave", "delta_chi", "nabla", "D.chi", "D.r",
    "E_ave_Youngs", "miu_lattice distortion", "Poisson_ave", "G_ave_shear",
    "F_P_N factor", "A_energy term", "delta_G", "D.G", "gamma_gamma para",
    "eta_modulus mismatch", "EN_electronegativity", "Bo", "Md", "PR", "IED",
]

DEFAULT_WEIGHTS: Dict[str, float] = {
    "RF": 0.25, "XGBoost": 0.25, "SVR": 0.25, "ANN": 0.25,
}


@dataclass
class Config:
    input_path: str = "data.csv"
    output_dir: str = "results"
    n_iter: int = 10
    test_size: float = 0.2
    random_seed_base: int = 42
    top_n: int = 20
    background_size: int = 100
    kernel_nsamples: int = 100
    show_plots: bool = False
    features: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Bootstrap SHAP feature importance ranking across 4 models."
    )
    p.add_argument("--input", default="data.csv", help="Input CSV (last column = target)")
    p.add_argument("--output-dir", default="results", help="Output directory")
    p.add_argument("--n-iter", type=int, default=10, help="Bootstrap iterations")
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument("--top-n", type=int, default=20, help="Top-N features in plots")
    p.add_argument("--background-size", type=int, default=100,
                   help="Background samples for KernelExplainer")
    p.add_argument("--kernel-nsamples", type=int, default=100,
                   help="KernelExplainer evaluation samples")
    p.add_argument("--features", nargs="+", default=None,
                   help="Override default feature list (space-separated)")
    p.add_argument("--show-plots", action="store_true", help="Show plots interactively")
    args = p.parse_args()

    cfg = Config(
        input_path=args.input,
        output_dir=args.output_dir,
        n_iter=args.n_iter,
        test_size=args.test_size,
        random_seed_base=args.seed,
        top_n=args.top_n,
        background_size=args.background_size,
        kernel_nsamples=args.kernel_nsamples,
        show_plots=args.show_plots,
    )
    if args.features:
        cfg.features = args.features
    return cfg


# =============================================================================
# Utilities
# =============================================================================

def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def load_and_scale(path: str, features: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    data = pd.read_csv(path)
    existing = [f for f in features if f in data.columns]
    missing = set(features) - set(existing)
    if missing:
        print(f"[WARN] Features not found, ignored: {sorted(missing)}")
    print(f"Using {len(existing)} features")
    X = data[existing]
    y = data.iloc[:, -1]
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=existing)
    return X_scaled, y, existing


def bootstrap_split(X, y, cfg: Config, i: int):
    return train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_seed_base + i
    )


def aggregate(imp_matrix: np.ndarray, features: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": features,
        "mean_shap": imp_matrix.mean(axis=0),
        "std_shap": imp_matrix.std(axis=0),
    })
    return df.sort_values("mean_shap", ascending=False).reset_index(drop=True)


def save_bar(df: pd.DataFrame, xlabel: str, title: str,
             path: Path, top_n: int, show: bool = False) -> None:
    top = df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["mean_shap"], xerr=top["std_shap"],
            capsize=3, color="steelblue", edgecolor="black")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def save_box(imp_matrix: np.ndarray, all_features: List[str],
             df_sorted: pd.DataFrame, title: str, path: Path,
             top_n: int = 10, show: bool = False) -> None:
    top_feats = df_sorted["feature"].head(top_n).tolist()
    idx = [all_features.index(f) for f in top_feats]
    df_box = pd.DataFrame(imp_matrix[:, idx], columns=top_feats)
    fig, ax = plt.subplots(figsize=(12, 6))
    df_box.boxplot(rot=45, patch_artist=True, ax=ax)
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Model-specific bootstrap routines
# =============================================================================

def run_rf(X, y, cfg: Config) -> np.ndarray:
    imp_list = []
    for i in tqdm(range(cfg.n_iter), desc="RF bootstrap"):
        X_tr, X_te, y_tr, _ = bootstrap_split(X, y, cfg, i)
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_split=5, min_samples_leaf=2,
            random_state=cfg.random_seed_base + i, n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        sv = shap.TreeExplainer(model).shap_values(X_te.values)
        imp_list.append(np.mean(np.abs(sv), axis=0))
    return np.array(imp_list)


def run_xgb(X, y, cfg: Config) -> np.ndarray:
    params = {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "verbosity": 0,
    }
    imp_list = []
    for i in tqdm(range(cfg.n_iter), desc="XGBoost bootstrap"):
        X_tr, X_te, y_tr, _ = bootstrap_split(X, y, cfg, i)
        model = xgb.XGBRegressor(**params, random_state=cfg.random_seed_base + i)
        model.fit(X_tr, y_tr)
        sv = shap.TreeExplainer(model).shap_values(X_te.values)
        imp_list.append(np.mean(np.abs(sv), axis=0))
    return np.array(imp_list)


def run_svr(X, y, cfg: Config) -> np.ndarray:
    imp_list = []
    for i in tqdm(range(cfg.n_iter), desc="SVR bootstrap"):
        X_tr, X_te, y_tr, _ = bootstrap_split(X, y, cfg, i)
        model = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
        model.fit(X_tr, y_tr)
        bg = X_tr.sample(n=min(cfg.background_size, len(X_tr)),
                         random_state=cfg.random_seed_base + i)
        explainer = shap.KernelExplainer(model.predict, bg)
        sv = explainer.shap_values(X_te.values, nsamples=cfg.kernel_nsamples)
        imp_list.append(np.mean(np.abs(sv), axis=0))
    return np.array(imp_list)


def run_ann(X, y, cfg: Config) -> np.ndarray:
    params = {
        "hidden_layer_sizes": (100, 100), "activation": "relu", "solver": "adam",
        "learning_rate_init": 0.001, "max_iter": 500, "early_stopping": True,
        "validation_fraction": 0.1, "n_iter_no_change": 20, "alpha": 0.0001,
        "batch_size": "auto", "verbose": False,
    }
    imp_list = []
    for i in tqdm(range(cfg.n_iter), desc="ANN bootstrap"):
        X_tr, X_te, y_tr, _ = bootstrap_split(X, y, cfg, i)
        model = MLPRegressor(**params, random_state=cfg.random_seed_base + i)
        model.fit(X_tr, y_tr)
        bg = X_tr.sample(n=min(cfg.background_size, len(X_tr)),
                         random_state=cfg.random_seed_base + i)
        explainer = shap.KernelExplainer(model.predict, bg)
        sv = explainer.shap_values(X_te.values, nsamples=cfg.kernel_nsamples)
        imp_list.append(np.mean(np.abs(sv), axis=0))
    return np.array(imp_list)


MODEL_RUNNERS = {
    "RF": run_rf,
    "XGBoost": run_xgb,
    "SVR": run_svr,
    "ANN": run_ann,
}


# =============================================================================
# Pipeline
# =============================================================================

def compute_all_importances(X, y, features, cfg, out_dir: Path) -> Dict[str, pd.DataFrame]:
    results = {}
    for name, runner in MODEL_RUNNERS.items():
        print(f"\n========== {name} Bootstrap SHAP ==========")
        imp_matrix = runner(X, y, cfg)
        df = aggregate(imp_matrix, features)
        df.to_csv(out_dir / f"{name}_bootstrap_SHAP_importance.csv", index=False)
        save_bar(
            df, xlabel="Mean |SHAP|",
            title=f"{name} feature importance "
                  f"({cfg.n_iter} bootstrap runs, mean ± std)",
            path=out_dir / f"{name}_bootstrap_SHAP_bar.png",
            top_n=cfg.top_n, show=cfg.show_plots,
        )
        save_box(
            imp_matrix, features, df,
            title=f"Top-{min(10, len(df))} features across {cfg.n_iter} bootstrap runs",
            path=out_dir / f"{name}_bootstrap_SHAP_boxplot.png",
            show=cfg.show_plots,
        )
        results[name] = df
    return results


def integrated_ranking(results: Dict[str, pd.DataFrame], cfg: Config,
                       out_dir: Path) -> pd.DataFrame:
    print("\n========== Integrated Normalized Ranking ==========")
    norm = {}
    for name, df in results.items():
        m = df["mean_shap"].max()
        df["norm_shap"] = df["mean_shap"] / m if m > 0 else 0.0
        norm[name] = df.set_index("feature")["norm_shap"]

    first_key = next(iter(norm))
    combined = pd.DataFrame(index=norm[first_key].index)
    for name, series in norm.items():
        combined[name] = series

    model_cols = list(cfg.weights.keys())
    combined["weighted_avg"] = sum(combined[m] * cfg.weights[m] for m in model_cols)
    combined["error_sem"] = combined[model_cols].std(axis=1) / np.sqrt(len(model_cols))
    combined = combined.sort_values("weighted_avg", ascending=False)

    print(combined[["weighted_avg", "error_sem"]].head(20))
    combined.to_csv(out_dir / "Integrated_normalized_SHAP_ranking.csv")

    top = combined.head(cfg.top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.index, top["weighted_avg"], xerr=top["error_sem"],
            capsize=3, color="darkorange", edgecolor="black")
    ax.invert_yaxis()
    ax.set_xlabel("Normalized weighted importance (mean ± SEM)")
    ax.set_title(f"Integrated SHAP importance ranking (Top {cfg.top_n})")
    fig.tight_layout()
    fig.savefig(out_dir / "Integrated_importance_ranking.png", dpi=300)
    if cfg.show_plots:
        plt.show()
    plt.close(fig)
    return combined


def main() -> None:
    cfg = parse_args()
    setup_matplotlib()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, features = load_and_scale(cfg.input_path, cfg.features)
    results = compute_all_importances(X, y, features, cfg, out_dir)
    integrated_ranking(results, cfg, out_dir)

    print(f"\nAll results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()