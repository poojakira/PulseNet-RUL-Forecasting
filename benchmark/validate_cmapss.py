"""
Validate historical PulseNet anomaly detection against NASA C-MAPSS FD001.

Methodology:
- Uses NASA C-MAPSS FD001 turbofan engine degradation simulation data
- Proper train/test split: train on train_FD001.txt, evaluate on test_FD001.txt
- No data leakage: normalization statistics computed ONLY from training data
- Piecewise linear RUL labeling with clip at 125 cycles
- Sliding window sequences (window=30)
- Isolation Forest for anomaly/degradation detection
- Binary classification: 'degraded' = engine in final 30% of life

Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling
for Aircraft Engine Run-to-Failure Simulation", NASA Prognostics CoE.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_SIZE = 30
RUL_CLIP = 125  # Piecewise linear: cap RUL at this value
DEGRADED_PERCENTILE = 0.30  # Final 30% of life = 'degraded'
RANDOM_STATE = 42
N_BOOTSTRAP = 1000  # Bootstrap iterations for confidence intervals

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "official" / "CMAPSSData"
OUTPUT_DIR = Path("results") / "validation"

COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_fd001() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load NASA C-MAPSS FD001 train, test, and RUL files."""
    train_path = DATA_DIR / "train_FD001.txt"
    test_path = DATA_DIR / "test_FD001.txt"
    rul_path = DATA_DIR / "RUL_FD001.txt"

    for p in (train_path, test_path, rul_path):
        if not p.exists():
            print(f"ERROR: Required data file not found: {p}")
            print(
                "Please ensure NASA C-MAPSS data is extracted to data/official/CMAPSSData/"
            )
            sys.exit(1)

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_series = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])["rul"]

    print(
        f"Loaded train_FD001: {len(train_df)} rows, {train_df['unit_id'].nunique()} engines"
    )
    print(
        f"Loaded test_FD001:  {len(test_df)} rows, {test_df['unit_id'].nunique()} engines"
    )
    print(f"Loaded RUL_FD001:   {len(rul_series)} entries")

    return train_df, test_df, rul_series


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def add_rul_labels(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    """Add piecewise linear RUL to training data (clip at max value)."""
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=clip)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_test_rul(
    test_df: pd.DataFrame, rul_series: pd.Series, clip: int = RUL_CLIP
) -> pd.DataFrame:
    """Add RUL labels to test data using the provided ground truth RUL values."""
    test_df = test_df.copy()

    # For test data, we know the RUL at the LAST cycle of each engine
    # We need to compute RUL for every cycle in the test trajectories
    max_cycles = test_df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]

    # Map ground truth RUL (indexed 0-based) to unit_ids (1-based)
    unit_ids = sorted(test_df["unit_id"].unique())
    rul_map = dict(zip(unit_ids, rul_series.values, strict=False))

    test_df = test_df.merge(max_cycles, on="unit_id")
    test_df["rul_at_end"] = test_df["unit_id"].map(rul_map)
    test_df["rul"] = test_df["rul_at_end"] + (test_df["max_cycle"] - test_df["cycle"])
    test_df["rul"] = test_df["rul"].clip(upper=clip)
    test_df.drop(columns=["max_cycle", "rul_at_end"], inplace=True)
    return test_df


def normalize_sensors(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize sensor columns to [0,1] using ONLY training statistics (no data leakage)."""
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Fit on training data only
    scaler.fit(train_df[SENSOR_COLS])

    # Transform both
    train_df[SENSOR_COLS] = scaler.transform(train_df[SENSOR_COLS])
    test_df[SENSOR_COLS] = scaler.transform(test_df[SENSOR_COLS])

    return train_df, test_df


def create_degradation_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary label: 1 if engine is in final 30% of its total life (degraded).
    Uses per-engine max lifetime to determine threshold.
    """
    df = df.copy()
    # Total life = max_cycle for training, or max_cycle + rul_at_end for test
    # Since we already computed RUL, we can use: degraded = (rul / (rul + cycles_elapsed)) < 0.30
    # Simpler: total_life = cycle + rul; degraded if rul / total_life < 0.30
    df["total_life"] = df["cycle"] + df["rul"]
    df["life_fraction_remaining"] = df["rul"] / df["total_life"]
    df["degraded"] = (df["life_fraction_remaining"] < DEGRADED_PERCENTILE).astype(int)
    return df


def create_sliding_windows(
    df: pd.DataFrame, window_size: int = WINDOW_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window feature matrices from sensor data.
    Returns flattened windows (n_samples, window_size * n_sensors) and labels.
    """
    features_list = []
    labels_list = []

    for unit_id in df["unit_id"].unique():
        unit_data = df[df["unit_id"] == unit_id].sort_values("cycle")
        sensors = unit_data[SENSOR_COLS].values
        labels = unit_data["degraded"].values

        if len(sensors) < window_size:
            # Pad with first row repeated
            padding = np.tile(sensors[0], (window_size - len(sensors), 1))
            sensors = np.vstack([padding, sensors])
            padded_labels = np.concatenate(
                [np.zeros(window_size - len(labels)), labels]
            )
            labels = padded_labels

        for i in range(window_size, len(sensors) + 1):
            window = sensors[i - window_size : i]
            features_list.append(window.flatten())
            labels_list.append(labels[i - 1])

    return np.array(features_list), np.array(labels_list)


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------
def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """Train Isolation Forest on training features."""
    # Contamination estimated from training degradation rate
    model = IsolationForest(
        n_estimators=200,
        contamination=0.3,  # Approximate expected anomaly rate
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def evaluate_model(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate Isolation Forest predictions against ground truth."""
    # Isolation Forest: -1 = anomaly (degraded), 1 = normal
    raw_predictions = model.predict(X_test)
    y_pred = (raw_predictions == -1).astype(int)

    # Anomaly scores (more negative = more anomalous)
    scores = model.decision_function(X_test)
    # Invert so higher = more likely degraded (for AUC-ROC)
    anomaly_scores = -scores

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, anomaly_scores)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "y_pred": y_pred,
        "y_test": y_test,
        "anomaly_scores": anomaly_scores,
    }


def bootstrap_confidence_intervals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: np.ndarray,
    n_iterations: int = N_BOOTSTRAP,
    confidence: float = 0.95,
) -> dict:
    """Compute bootstrap confidence intervals for all metrics."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = len(y_test)

    metrics = {"precision": [], "recall": [], "f1": [], "auc_roc": []}

    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, size=n_samples)
        y_t = y_test[indices]
        y_p = y_pred[indices]
        scores_b = anomaly_scores[indices]

        # Skip degenerate bootstrap samples
        if len(np.unique(y_t)) < 2:
            continue

        metrics["precision"].append(precision_score(y_t, y_p, zero_division=0))
        metrics["recall"].append(recall_score(y_t, y_p, zero_division=0))
        metrics["f1"].append(f1_score(y_t, y_p, zero_division=0))
        metrics["auc_roc"].append(roc_auc_score(y_t, scores_b))

    alpha = (1 - confidence) / 2
    ci = {}
    for name, values in metrics.items():
        values = np.array(values)
        ci[name] = {
            "lower": float(np.percentile(values, alpha * 100)),
            "upper": float(np.percentile(values, (1 - alpha) * 100)),
        }

    return ci


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("NASA C-MAPSS FD001 Validation - Isolation Forest Anomaly Detection")
    print("=" * 70)
    print()

    # 1. Load data
    print("[1/6] Loading NASA C-MAPSS FD001 dataset...")
    train_df, test_df, rul_series = load_fd001()
    print()

    # 2. Add RUL labels
    print("[2/6] Computing RUL labels (piecewise linear, clip=125)...")
    train_df = add_rul_labels(train_df)
    test_df = add_test_rul(test_df, rul_series)
    print(f"  Train RUL range: [{train_df['rul'].min()}, {train_df['rul'].max()}]")
    print(f"  Test RUL range:  [{test_df['rul'].min()}, {test_df['rul'].max()}]")
    print()

    # 3. Normalize sensors (no data leakage)
    print("[3/6] Normalizing sensors using training statistics only...")
    train_df, test_df = normalize_sensors(train_df, test_df)
    print("  Scaler fitted on training data only (no leakage)")
    print()

    # 4. Create binary degradation labels
    print("[4/6] Creating binary degradation labels (final 30% of life)...")
    train_df = create_degradation_labels(train_df)
    test_df = create_degradation_labels(test_df)
    train_pos_rate = train_df["degraded"].mean()
    test_pos_rate = test_df["degraded"].mean()
    print(f"  Train positive rate: {train_pos_rate:.4f}")
    print(f"  Test positive rate:  {test_pos_rate:.4f}")
    print()

    # 5. Create sliding windows
    print(f"[5/6] Creating sliding window sequences (window={WINDOW_SIZE})...")
    t0 = time.time()
    X_train, y_train = create_sliding_windows(train_df, WINDOW_SIZE)
    X_test, y_test = create_sliding_windows(test_df, WINDOW_SIZE)
    window_time = time.time() - t0
    print(f"  Train samples: {X_train.shape[0]} (features: {X_train.shape[1]})")
    print(f"  Test samples:  {X_test.shape[0]} (features: {X_test.shape[1]})")
    print(f"  Window creation time: {window_time:.2f}s")
    print()

    # 6. Train and evaluate
    print("[6/6] Training Isolation Forest and evaluating...")
    t0 = time.time()
    model = train_isolation_forest(X_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.2f}s")

    results = evaluate_model(model, X_test, y_test)
    print()
    print("-" * 50)
    print("RESULTS (Test Set Only - Never Seen During Training)")
    print("-" * 50)
    print(f"  Precision:  {results['precision']:.4f}")
    print(f"  Recall:     {results['recall']:.4f}")
    print(f"  F1 Score:   {results['f1']:.4f}")
    print(f"  AUC-ROC:    {results['auc_roc']:.4f}")
    print()

    # Confidence intervals
    print("Computing bootstrap confidence intervals (95%, n=1000)...")
    ci = bootstrap_confidence_intervals(
        results["y_test"], results["y_pred"], results["anomaly_scores"]
    )
    for metric, interval in ci.items():
        print(f"  {metric}: [{interval['lower']:.4f}, {interval['upper']:.4f}]")
    print()

    # Full classification report
    print("Classification Report:")
    print(
        classification_report(
            y_test, results["y_pred"], target_names=["Normal", "Degraded"]
        )
    )

    # 7. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "methodology": {
            "description": "Isolation Forest trained on NASA C-MAPSS FD001 training set with sliding window features. "
            "Evaluated on held-out test set using ground truth RUL for binary degradation classification.",
            "train_test_split": "Official NASA split: train_FD001.txt (100 engines, run-to-failure) / "
            "test_FD001.txt (100 engines, partial trajectories) with RUL_FD001.txt ground truth",
            "preprocessing": "MinMaxScaler fitted on training data only, applied to both train and test",
            "rul_labeling": f"Piecewise linear degradation, clipped at {RUL_CLIP} cycles",
            "degradation_threshold": f"Final {int(DEGRADED_PERCENTILE * 100)}% of engine life classified as degraded",
            "window_size": WINDOW_SIZE,
            "features_per_sample": int(X_train.shape[1]),
        },
        "data_source": "NASA C-MAPSS FD001 (Prognostics CoE)",
        "data_reference": "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data",
        "no_data_leakage": True,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_positive_rate": float(train_pos_rate),
        "test_positive_rate": float(test_pos_rate),
        "model": {
            "type": "IsolationForest",
            "n_estimators": 200,
            "contamination": 0.3,
            "random_state": RANDOM_STATE,
        },
        "metrics": {
            "precision": round(float(results["precision"]), 4),
            "recall": round(float(results["recall"]), 4),
            "f1": round(float(results["f1"]), 4),
            "auc_roc": round(float(results["auc_roc"]), 4),
        },
        "confidence_intervals_95pct": {
            k: {"lower": round(v["lower"], 4), "upper": round(v["upper"], 4)}
            for k, v in ci.items()
        },
        "training_time_seconds": round(train_time, 3),
        "evaluation_note": "Isolation Forest is an unsupervised anomaly detector; "
        "F1 scores in the 0.4-0.6 range are expected for this approach on C-MAPSS. "
        "Supervised deep learning models (LSTM, CNN) typically achieve RMSE 12-14 on RUL regression.",
        "synthetic_data": False,
    }

    output_path = OUTPUT_DIR / "cmapss_fd001_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")
    print()
    print("VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
