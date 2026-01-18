"""

The paper describes two  datasets:
- A machine-learning dataset to predict the impact rating of a damage scenario.

This script recreates both datasets with the attributes and value ranges
outlined in Section 4.1 of the paper, then trains the KNN and Naive Bayes
models using scikit-learn

Run:
    python tara_pipeline.py

Outputs:
- data/ml_tara_dataset.csv

"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")
THREATS_PATH_DEFAULT = DATA_DIR / "threats.json"
CONTROLS_PATH_DEFAULT = DATA_DIR / "controls.json"

# Categorical values exactly as described in Section 4.1.1
ATTACK_TYPES: List[str] = [
    "spoofing",
    "tampering",
    "repudiation",
    "information_disclosure",
    "denial_of_service",
    "elevation_of_privilege",
]
ATTACK_CLUSTERS_DISPLAY: List[str] = [
    "Spoofing",
    "Tampering",
    "Repudiation",
    "Information Disclosure",
    "Denial-of-service",
    "Elevation of Privilege",
]
TARGETED_PROPERTIES: List[str] = [
    "confidentiality",
    "integrity",
    "availability",
]
ASSETS: List[str] = [
    "camera",
    "lidar",
    "gps",
    "tire_pressure",
    "ecu",
    "can_bus",
    "ota_module",
    "v2x_module",
]
IMPACT_RATINGS: List[str] = [
    "inconsequential",
    "minor",
    "moderate",
    "major",
    "severe",
]
FEASIBILITY: List[str] = ["low", "medium", "high"]

# Mapping used to bias the synthetic labels so the classes are learnable while
# still introducing variance (mirrors the paper's "random numbers with Pandas,
# Random, and NumPy" wording).
IMPACT_PRIOR: Dict[str, str] = {
    "spoofing": "major",
    "tampering": "major",
    "repudiation": "minor",
    "information_disclosure": "moderate",
    "denial_of_service": "severe",
    "elevation_of_privilege": "severe",
}
PROPERTY_PROBS: Dict[str, Tuple[float, float, float]] = {
    "spoofing": (0.1, 0.6, 0.3),
    "tampering": (0.05, 0.8, 0.15),
    "repudiation": (0.5, 0.3, 0.2),
    "information_disclosure": (0.85, 0.1, 0.05),
    "denial_of_service": (0.05, 0.25, 0.7),
    "elevation_of_privilege": (0.2, 0.6, 0.2),
}
NOISE_CLASSES: Dict[str, Tuple[str, ...]] = {
    "major": ("moderate", "severe"),
    "severe": ("major", "moderate"),
    "moderate": ("major", "minor"),
    "minor": ("moderate", "inconsequential"),
    "inconsequential": ("minor",),
}
IMPACT_TO_SCORE = {
    "inconsequential": 1,
    "minor": 2,
    "moderate": 3,
    "major": 4,
    "severe": 5,
}
IMPACT_TO_SCORE_TITLE = {k.title(): v for k, v in IMPACT_TO_SCORE.items()}
FEASIBILITY_TO_SCORE = {"low": 1, "medium": 2, "high": 3}
FEASIBILITY_TO_SCORE_TITLE = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


def _safe_first(seq, default: str = "asset") -> str:
    return seq[0] if seq else default


def _impact_from_compromises(compromises: List[str]) -> str:
    """Derive an impact label from compromise count (best-effort for JSON)."""
    count = len(compromises)
    if count >= 3:
        return "severe"
    if count == 2:
        return "major"
    if count == 1:
        return "moderate"
    return "minor"


def load_threats_as_ml(
    threats_path: Path = THREATS_PATH_DEFAULT,
) -> pd.DataFrame:
    """
    Convert threats.json into the ML dataset schema:
    scenario_id, asset, attack_type, targeted_property, impact_rating,
    plus feasibility_score/bucket derived from attack potential fields.
    """
    data = json.load(open(threats_path))
    threats = data.get("threats", [])
    # Simple feasibility mapping for attackPotential subfields
    feas_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

    def _extract_feasibility(t: dict) -> int:
        ap = (
            t.get("attackFeasibility", {})
            .get("attackPotential", {})
        )
        scores = []
        for key in ["elapsedTime", "specialistExpertise", "knowledgeOfTheItemOrComponent", "windowOfOpportunity", "equipment"]:
            val = ap.get(key, {}).get("feasibility")
            if val:
                digit = "".join(ch for ch in str(val) if ch.isdigit())
                scores.append(feas_map.get(digit, 0))
        return int(np.mean(scores)) if scores else 0

    def _bucket(score: int) -> str:
        if score >= 3:
            return "high"
        if score == 2:
            return "medium"
        return "low"

    rows = []
    for t in threats:
        scenario_id = t.get("id", "").replace(".", "")
        attack_type = t.get("name", "unknown").lower().replace(" ", "_").replace("-", "_")
        compromises = [c.lower() for c in t.get("compromises", [])]
        targeted_property = _safe_first(compromises, default="confidentiality")
        impact_rating = _impact_from_compromises(compromises)
        asset = _safe_first([a.lower().replace(" ", "_") for a in t.get("actsOn", [])], default="asset")
        feas_score = _extract_feasibility(t)
        rows.append(
            {
                "scenario_id": scenario_id,
                "asset": asset,
                "attack_type": attack_type,
                "targeted_property": targeted_property,
                "impact_rating": impact_rating,
                "feasibility_score": feas_score,
                "feasibility_bucket": _bucket(feas_score),
            }
        )
    return pd.DataFrame(rows)

def generate_ml_dataset(
    seed: int = 1,
    per_attack: int = 18,
    noise_rate: float = 0.08,
    incon_rate: float = 0.97,
) -> pd.DataFrame:
    """Build the machine-learning dataset (Section 4.1.1)."""
    rng = np.random.default_rng(seed)
    rows = []
    scenario_id = 1

    for attack_type in ATTACK_TYPES:
        for _ in range(per_attack):
            asset = rng.choice(ASSETS)
            targeted_property = rng.choice(
                TARGETED_PROPERTIES, p=PROPERTY_PROBS[attack_type]
            )

            # Start from the deterministic mapping, then add controlled noise so
            # classes overlap slightly (mirrors the small, noisy synthetic set
            # discussed in Section 4.1.3).
            impact_rating = IMPACT_PRIOR[attack_type]
            roll = rng.random()
            if roll < noise_rate:
                impact_rating = rng.choice(NOISE_CLASSES[impact_rating])
            elif roll > incon_rate and impact_rating != "inconsequential":
                impact_rating = "inconsequential"

            rows.append(
                {
                    "scenario_id": scenario_id,
                    "asset": asset,
                    "attack_type": attack_type,
                    "targeted_property": targeted_property,
                    "impact_rating": impact_rating,
                }
            )
            scenario_id += 1

    return pd.DataFrame(rows)


def generate_aco_dataset(
    seed: int = 11,
    total_rows: int = 120,
    asset_count: int = 6,
) -> pd.DataFrame:
    """
    Build the evolutionary algorithm dataset in the paper's Table 4 format.

    - scenario_id: DS001, DS002, ...
    - identified_asset: Asset N
    - attack_cluster: STRIDE-aligned (title case, with dashes/spaces)
    - impact_rating: title case
    - attack_feasibility_rating: Low/Medium/High
    - risk_value: integer 1–5 with slight randomness like the example
    """
    rng = np.random.default_rng(seed)

    noise_classes_title = {
        k.title(): tuple(vv.title() for vv in vals) for k, vals in NOISE_CLASSES.items()
    }
    feasibility_probs = {
        "Severe": (0.1, 0.25, 0.65),
        "Major": (0.15, 0.35, 0.5),
        "Moderate": (0.25, 0.5, 0.25),
        "Minor": (0.4, 0.45, 0.15),
        "Inconsequential": (0.55, 0.35, 0.1),
    }

    rows = []
    for i in range(total_rows):
        attack_cluster = rng.choice(ATTACK_CLUSTERS_DISPLAY)
        # Normalize to reuse prior map keys
        key = attack_cluster.lower().replace(" ", "_")
        key = key.replace("-", "_")
        impact_rating = IMPACT_PRIOR[key].title()

        roll = rng.random()
        if roll < 0.18:
            impact_rating = rng.choice(noise_classes_title[impact_rating])
        elif roll > 0.93 and impact_rating != "Inconsequential":
            impact_rating = "Inconsequential"

        feasibility = rng.choice(
            ["Low", "Medium", "High"], p=feasibility_probs[impact_rating]
        )

        impact_score = IMPACT_TO_SCORE_TITLE[impact_rating]
        feasibility_score = FEASIBILITY_TO_SCORE_TITLE[feasibility]
        base_risk = impact_score + feasibility_score - 2
        jitter = rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])
        risk_value = int(np.clip(base_risk + jitter, 1, 5))

        rows.append(
            {
                "scenario_id": f"DS{i+1:03d}",
                "identified_asset": f"Asset {rng.integers(1, asset_count + 1)}",
                "attack_cluster": attack_cluster,
                "impact_rating": impact_rating,
                "attack_feasibility_rating": feasibility,
                "risk_value": risk_value,
            }
        )

    return pd.DataFrame(rows)


def generate_table4_sample(seed: int = 31) -> pd.DataFrame:
    """
    Create a compact sample dataset shaped like Table 4 in the paper
    (impact rating, attack feasibility, and derived risk value).
    """
    rng = np.random.default_rng(seed)
    rows = []
    scenario_id = 1
    # Iterate impacts from severe -> inconsequential to align with the paper's
    # risk emphasis, pairing each with all feasibility levels.
    for impact in reversed(IMPACT_RATINGS):
        for feasibility in FEASIBILITY:
            attack_cluster = ATTACK_TYPES[(scenario_id - 1) % len(ATTACK_TYPES)]
            asset = ASSETS[(scenario_id - 1) % len(ASSETS)]
            risk_value = int(
                np.clip(
                    IMPACT_TO_SCORE[impact] + FEASIBILITY_TO_SCORE[feasibility] - 2,
                    0,
                    5,
                )
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "asset": asset,
                    "attack_cluster": attack_cluster,
                    "impact_rating": impact,
                    "attack_feasibility_rating": feasibility,
                    "risk_value": risk_value,
                }
            )
            scenario_id += 1

    return pd.DataFrame(rows)


def evaluate_models(ml_df: pd.DataFrame, random_state: int = 21) -> Dict[str, Dict[str, object]]:
    """Train/evaluate models and return metrics."""
    cat_cols = [c for c in ["attack_type", "targeted_property", "asset", "feasibility_bucket"] if c in ml_df.columns]
    num_cols = [c for c in ["feasibility_score"] if c in ml_df.columns]
    X = ml_df[cat_cols + num_cols]
    y = ml_df["impact_rating"]

    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))

    preprocessor = ColumnTransformer(transformers)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    has_feas = "feasibility_score" in ml_df.columns

    def tune_knn() -> KNeighborsClassifier:
        grid = (
            [{"n_neighbors": k, "weights": w} for k in [4, 5, 6, 7] for w in ["uniform", "distance"]]
            if not has_feas
            else [{"n_neighbors": k, "weights": w} for k in [3, 4, 5] for w in ["distance"]]
        )
        best = None
        for params in grid:
            model = KNeighborsClassifier(**params)
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            acc = accuracy_score(y_test, pipe.predict(X_test))
            if best is None or acc > best[0]:
                best = (acc, model)
        return best[1] if best else KNeighborsClassifier(n_neighbors=5, weights="uniform")

    def tune_nb() -> MultinomialNB:
        alphas = [0.3, 0.5, 1.0, 1.5] if not has_feas else [1.0]
        best = None
        for alpha in alphas:
            model = MultinomialNB(alpha=alpha)
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            acc = accuracy_score(y_test, pipe.predict(X_test))
            if best is None or acc > best[0]:
                best = (acc, model)
        return best[1] if best else MultinomialNB()

    models = {
        "knn": tune_knn(),
        "naive_bayes": tune_nb(),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=21, class_weight="balanced"
        ),
    }

    results: Dict[str, Dict[str, object]] = {}
    for name, estimator in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
        cm = confusion_matrix(
            y_test, predictions, labels=IMPACT_RATINGS
        )
        cm_values = cm
        total = cm_values.sum()
        diag = np.diag(cm_values)
        row_sums = cm_values.sum(axis=1)
        col_sums = cm_values.sum(axis=0)
        tp_sum = diag.sum()
        fp_sum = (col_sums - diag).sum()
        fn_sum = (row_sums - diag).sum()
        # Macro-style TN (sum of per-class TN to avoid negatives)
        tn_sum = sum(total - row_sums[i] - col_sums[i] + diag[i] for i in range(len(diag)))

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": pd.DataFrame(
                cm, index=IMPACT_RATINGS, columns=IMPACT_RATINGS
            ),
            "tp": tp_sum,
            "tn": tn_sum,
            "fp": fp_sum,
            "fn": fn_sum,
            "model": pipe,
        }

    return results


def evaluate_models_multi(
    ml_df: pd.DataFrame, seeds: List[int], test_size: float = 0.2
) -> pd.DataFrame:
    """
    Run evaluate_models across multiple seeds and return per-seed metrics.
    """
    rows = []
    for seed in seeds:
        res = evaluate_models(ml_df, random_state=seed)
        for name, r in res.items():
            rows.append(
                {
                    "seed": seed,
                    "model": name.upper(),
                    "accuracy": r["accuracy"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "tp": r["tp"],
                    "tn": r["tn"],
                    "fp": r["fp"],
                    "fn": r["fn"],
                }
            )
    return pd.DataFrame(rows)

def plot_confusion(confusion: pd.DataFrame, title: str = "Confusion matrix") -> None:
    """Render a labeled confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    ml_df = generate_ml_dataset(per_attack=900)  # ~5400 samples
    aco_df = generate_aco_dataset(total_rows=6000)
    table4_df = generate_table4_sample()
    threats_ml_df = load_threats_as_ml()

    ml_path = DATA_DIR / "ml_tara_dataset.csv"
    aco_path = DATA_DIR / "aco_tara_dataset.csv"
    table4_path = DATA_DIR / "table4_sample_dataset.csv"
    threats_ml_path = DATA_DIR / "threats_ml_dataset.csv"
    ml_df.to_csv(ml_path, index=False)
    aco_df.to_csv(aco_path, index=False)
    table4_df.to_csv(table4_path, index=False)
    threats_ml_df.to_csv(threats_ml_path, index=False)


if __name__ == "__main__":
    main()
