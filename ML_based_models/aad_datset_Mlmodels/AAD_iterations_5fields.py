# AAD_iterations_5fields.py
# Iterative evaluation on REAL AAD dataset (SQLite) using 5 fields aligned to paper.
# Features: Attack Type, Attack Class, Violated Security Property, Affected Asset
# Label:    Attack Level
#
# Models: KNN, Naive Bayes, Random Forest
# Outputs: CSV of all runs + mean/std summary + plots

from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# CONFIG
# -----------------------------
DB_PATH = r"C:\D-Drive\Neeraja\MastersNeeraja\autonomousCooperative\TaraPaperSummary\New folder\Automotive_Attack_Database_(AAD)_V3.0.db"
TABLE_NAME = "Automotive Security Attacks"

OUT_DIR = Path("aad_results")
OUT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "Attack Type",
    "Attack Class",
    "Violated Security Property",
    "Affected Asset",
]
LABEL_COL = "Attack Level"

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  # 10 iterations
TEST_SIZE = 0.2

sns.set(style="whitegrid")


# -----------------------------
# Helpers
# -----------------------------
def load_aad_dataset(db_path: str, table_name: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df = pd.read_sql(f'SELECT * FROM "{table_name}";', con)
    con.close()
    return df


def clean_aad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean label (Attack Level)
    df[LABEL_COL] = (
        df[LABEL_COL]
        .astype(str)
        .str.replace("\n-", "", regex=False)
        .str.strip()
    )

    # Drop missing
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

    # Optional: remove empty strings
    for c in FEATURE_COLS + [LABEL_COL]:
        df = df[df[c].astype(str).str.strip() != ""]

    return df


def compute_rates_from_cm(cm: np.ndarray) -> dict:
    """
    Macro-averaged TPR, TNR, FPR, FNR for multi-class CM.
    cm: rows=true, cols=predicted
    """
    n = cm.shape[0]
    total = cm.sum()

    tpr_list, tnr_list, fpr_list, fnr_list = [], [], [], []
    for i in range(n):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FN - FP

        if TP + FN > 0:
            tpr_list.append(TP / (TP + FN))
        if TN + FP > 0:
            tnr_list.append(TN / (TN + FP))
        if FP + TN > 0:
            fpr_list.append(FP / (FP + TN))
        if FN + TP > 0:
            fnr_list.append(FN / (FN + TP))

    mean = lambda x: float(np.mean(x)) if x else 0.0
    return {
        "tpr": mean(tpr_list),
        "tnr": mean(tnr_list),
        "fpr": mean(fpr_list),
        "fnr": mean(fnr_list),
    }


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS)]
    )


def get_models(seed: int):
    return {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": MultinomialNB(),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed),
    }


# -----------------------------
# One run (one seed)
# -----------------------------
def evaluate_one_seed(df: pd.DataFrame, seed: int) -> dict:
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    # Stratify requires each class >= 2 in whole data (AAD usually OK)
    strat = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=strat
    )

    labels = sorted(y.unique().tolist())
    preprocessor = build_preprocessor()

    results = {}
    for name, clf in get_models(seed).items():
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, preds, labels=labels)
        rates = compute_rates_from_cm(cm)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tpr": rates["tpr"],
            "tnr": rates["tnr"],
            "fpr": rates["fpr"],
            "fnr": rates["fnr"],
            "cm": cm,
            "labels": labels,
        }

    return results


# -----------------------------
# Multi-run (iterations)
# -----------------------------



def summarize_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df_runs.groupby("model")[["accuracy", "precision", "recall", "f1", "tpr", "tnr", "fpr", "fnr"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "model" if col[0] == "model" else f"{col[0]}_{col[1]}"
        for col in summary.columns.to_flat_index()
    ]
    return summary


# -----------------------------
# Plots
# -----------------------------
def plot_grouped_metrics(summary_df: pd.DataFrame) -> None:
    plot_df = pd.DataFrame({
        "Model": summary_df["model"].str.upper().str.replace("_", " "),
        "Accuracy": summary_df["accuracy_mean"],
        "Precision": summary_df["precision_mean"],
        "Recall": summary_df["recall_mean"],
        "F1": summary_df["f1_mean"],
    })

    melted = plot_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(9, 4))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1.0)
    plt.title("AAD (5 fields): Mean performance across iterations")
    plt.tight_layout()
    out = OUT_DIR / "aad_5fields_grouped_metrics.png"
    plt.savefig(out, dpi=300)
    plt.show()
    print(f"[SAVED] {out}")


def plot_box(df_runs: pd.DataFrame, metric: str) -> None:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df_runs, x="model", y=metric)
    sns.stripplot(data=df_runs, x="model", y=metric, color="black", alpha=0.30)
    plt.ylim(0, 1.0)
    plt.title(f"AAD (5 fields): {metric} across iterations")
    plt.tight_layout()
    out = OUT_DIR / f"aad_5fields_box_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.show()
    print(f"[SAVED] {out}")


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, filename: str) -> None:
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out = OUT_DIR / filename
    plt.savefig(out, dpi=300)
    plt.show()
    print(f"[SAVED] {out}")

def plot_paper_style_bar(summary_df: pd.DataFrame) -> None:
    """
    Create paper-style grouped bar chart:
    KNN vs Naive Bayes vs Random Forest (AAD dataset)
    metrics: accuracy, precision, recall, f1
    values shown as percentages.
    """

    # Prepare data (mean values across iterations)
    plot_df = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1"],
        "KNN": [
            summary_df.loc[summary_df["model"] == "knn", "accuracy_mean"].values[0],
            summary_df.loc[summary_df["model"] == "knn", "precision_mean"].values[0],
            summary_df.loc[summary_df["model"] == "knn", "recall_mean"].values[0],
            summary_df.loc[summary_df["model"] == "knn", "f1_mean"].values[0],
        ],
        "NAIVE_BAYES": [
            summary_df.loc[summary_df["model"] == "naive_bayes", "accuracy_mean"].values[0],
            summary_df.loc[summary_df["model"] == "naive_bayes", "precision_mean"].values[0],
            summary_df.loc[summary_df["model"] == "naive_bayes", "recall_mean"].values[0],
            summary_df.loc[summary_df["model"] == "naive_bayes", "f1_mean"].values[0],
        ],
        "RANDOM_FOREST": [
            summary_df.loc[summary_df["model"] == "random_forest", "accuracy_mean"].values[0],
            summary_df.loc[summary_df["model"] == "random_forest", "precision_mean"].values[0],
            summary_df.loc[summary_df["model"] == "random_forest", "recall_mean"].values[0],
            summary_df.loc[summary_df["model"] == "random_forest", "f1_mean"].values[0],
        ],
    })

    # Convert to long format for seaborn
    long_df = plot_df.melt(id_vars="metric", var_name="model", value_name="score")

    # Convert to percentage
    long_df["score"] = long_df["score"] * 100

    plt.figure(figsize=(9, 4.5))
    ax = sns.barplot(data=long_df, x="metric", y="score", hue="model")

    ax.set_title("KNN vs Naïve Bayes vs Random Forest (AAD dataset)")
    ax.set_xlabel("metric")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.legend(title="model", loc="upper right")

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure_aad_models_comparison_10iter.png", dpi=300)
    plt.show()

    print("[SAVED]", OUT_DIR / "figure_aad_models_comparison_10iter.png")

def evaluate_multi(df: pd.DataFrame, target_runs: int = 10, start_seed: int = 1) -> pd.DataFrame:
    rows = []
    seed = start_seed
    successful_runs = 0

    while successful_runs < target_runs:
        try:
            res = evaluate_one_seed(df, seed)

            for model, m in res.items():
                rows.append({
                    "iteration": successful_runs + 1,
                    "seed": seed,
                    "model": model,            # IMPORTANT: lowercase "model"
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "tpr": m["tpr"],
                    "tnr": m["tnr"],
                    "fpr": m["fpr"],
                    "fnr": m["fnr"],
                })

            successful_runs += 1

        except ValueError as e:
            print(f"[SKIPPED] Seed {seed} failed: {e}")

        seed += 1

    return pd.DataFrame(rows)



# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Loading AAD DB...")
    df_raw = load_aad_dataset(DB_PATH, TABLE_NAME)
    print(f"Loaded rows: {len(df_raw)}")

    df = clean_aad(df_raw)
    print(f"After cleaning rows: {len(df)}\n")

    print("Attack Level distribution:")
    print(df[LABEL_COL].value_counts())
    print()

    target_runs = 10
    print(f"Running {target_runs} iterations (starting seed=1, retry on failure) ...")
    runs_df = evaluate_multi(df, target_runs=target_runs, start_seed=1)

    # Save all runs
    runs_csv = OUT_DIR / "aad_5fields_iterations_all_runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"[SAVED] {runs_csv}")

    print("Columns in runs_df:", list(runs_df.columns))
    print("Unique iterations:", runs_df["iteration"].nunique())
    print("Models:", runs_df["model"].unique())

    # Summary
    summary_df = summarize_runs(runs_df)

    # Print summary as % for readability
    pretty = summary_df.copy()
    for col in pretty.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            pretty[col] = pretty[col] * 100

    print("\n===== SUMMARY (mean ± std) across iterations =====")
    print(pretty.round(2).to_string(index=False))

    # Plots
    plot_paper_style_bar(summary_df)
    plot_grouped_metrics(summary_df)
    plot_box(runs_df, "accuracy")
    plot_box(runs_df, "f1")

    # Confusion matrix of BEST MODEL (highest mean F1) for a representative seed
    best_model = summary_df.sort_values("f1_mean", ascending=False).iloc[0]["model"]
    rep_seed = int(runs_df.loc[runs_df["Iteration"] == 1, "Seed"].iloc[0])
    print(f"\nBest model by mean F1: {best_model.upper()} | Representative seed: {rep_seed}")

    rep_res = evaluate_one_seed(df, rep_seed)
    cm = rep_res[best_model]["cm"]
    labels = rep_res[best_model]["labels"]
    plot_confusion_matrix(
        cm, labels,
        title=f"{best_model.upper()} Confusion Matrix (AAD, seed={rep_seed})",
        filename=f"aad_5fields_cm_{best_model}_seed{rep_seed}.png"
    )


if __name__ == "__main__":
    main()
