"""
PulseNet Retraining Pipeline — Airflow DAG for Automated Model Retraining.

This DAG runs weekly to:
1. Check for data drift
2. Retrain models if drift detected or on schedule
3. Validate new model against holdout set
4. Canary deploy with shadow traffic
4. Promote if metrics pass thresholds
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago

# Default DAG arguments
default_args = {
    "owner": "pulsenet-mlops",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=15),
    "max_active_runs": 1,
}

dag = DAG(
    "pulsenet_retraining_pipeline",
    default_args=default_args,
    description="Weekly PulseNet model retraining with drift detection",
    schedule_interval="0 2 * * 0",  # Weekly on Sunday at 2 AM UTC
    catchup=False,
    tags=["mlops", "retraining", "pulsenet", "drift"],
)

# === Task Functions ===

def check_drift(**context) -> dict:
    """Check for data and concept drift using Evidently."""
    import pandas as pd
    from pulsenet.monitoring.drift_detection import create_drift_monitoring
    from pulsenet.config import cfg
    
    # Load reference data (from last successful training)
    ref_path = Path(cfg.models.model_dir) / "reference_data.parquet"
    if not ref_path.exists():
        return {"drift_detected": False, "reason": "no_reference_data"}
    
    reference_data = pd.read_parquet(ref_path)
    
    # Load current data (from recent predictions)
    current_path = Path(cfg.models.model_dir) / "current_data.parquet"
    if not current_path.exists():
        return {"drift_detected": False, "reason": "no_current_data"}
    
    current_data = pd.read_parquet(current_path)
    
    # Check minimum samples
    if len(current_data) < 100:
        return {"drift_detected": False, "reason": "insufficient_samples"}
    
    # Run drift detection
    evidently, lightweight = create_drift_monitoring(
        reference_data=reference_data,
        target_column="RUL",
        prediction_column="predicted_RUL",
        numeric_features=[c for c in reference_data.columns if reference_data[c].dtype in ["float64", "int64"]],
        categorical_features=[],
    )
    
    drift_results = evidently.check_drift(current_data)
    
    # Push results to XCom
    context["ti"].xcom_push(key="drift_results", value=drift_results)
    
    return drift_results


def should_retrain(**context) -> str:
    """Determine if retraining should proceed based on drift results."""
    drift_results = context["ti"].xcom_pull(key="drift_results", task_ids="check_drift")
    
    if not drift_results:
        return "skip_retraining"
    
    drift_detected = drift_results.get("dataset_drift", False)
    n_drifted = drift_results.get("n_drifted_features", 0)
    drift_share = drift_results.get("drift_share", 0)
    
    # Force retrain if scheduled (every 4 weeks)
    last_retrain = Variable.get("pulsenet_last_retrain", default_var=None)
    if last_retrain:
        last_date = datetime.fromisoformat(last_retrain)
        if datetime.utcnow() - last_date > timedelta(weeks=4):
            return "retrain"
    
    # Retrain if significant drift
    if drift_detected and (drift_share > 0.1 or n_drifted > 5):
        return "retrain"
    
    return "skip_retraining"


def prepare_data(**context) -> dict:
    """Prepare training data with latest samples."""
    from pulsenet.config import cfg
    from pulsenet.pipeline.orchestrator import PipelineOrchestrator
    
    pipeline = PipelineOrchestrator()
    train_path, test_path = pipeline.prepare_training_data()
    
    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "timestamp": datetime.utcnow().isoformat(),
    }


def train_models(**context) -> dict:
    """Train all candidate models and select best."""
    from pulsenet.pipeline.orchestrator import PipelineOrchestrator
    from pulsenet.config import cfg
    
    # Get data paths from previous task
    data_info = context["ti"].xcom_pull(key="return_value", task_ids="prepare_data")
    
    pipeline = PipelineOrchestrator()
    results = pipeline.train_all_models(
        train_path=data_info["train_path"],
        test_path=data_info["test_path"],
    )
    
    # Select best model based on validation metric
    best_model = max(results.items(), key=lambda x: x[1].get("val_score", 0))
    
    result = {
        "best_model": best_model[0],
        "best_score": best_model[1].get("val_score"),
        "all_results": results,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    context["ti"].xcom_push(key="best_model", value=result)
    return result


def validate_model(**context) -> dict:
    """Validate model against holdout set and check performance thresholds."""
    from pulsenet.config import cfg
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    best_model_info = context["ti"].xcom_pull(key="best_model", task_ids="train_models")
    model_name = best_model_info["best_model"]
    
    # Load model
    model_path = Path(cfg.models.model_dir) / f"{model_name}.skops"
    model = joblib.load(model_path)
    
    # Load holdout test data
    test_data = pd.read_parquet(Path(cfg.models.model_dir) / "holdout_test.parquet")
    X_test = test_data.drop("RUL", axis=1)
    y_test = test_data["RUL"]
    
    # Predict
    preds = model.predict(X_test)
    
    # Compute metrics
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    
    # Check thresholds
    thresholds = {
        "rmse": 15.0,  # Max acceptable RMSE
        "mae": 10.0,   # Max acceptable MAE
        "r2": 0.80,    # Min acceptable R²
    }
    
    passed = all(
        metrics[k] <= v if k in ["rmse", "mae"] else metrics[k] >= v
        for k, v in thresholds.items()
    )
    
    result = {
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": passed,
        "model_name": model_name,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    context["ti"].xcom_push(key="validation_result", value=result)
    
    if not passed:
        raise ValueError(f"Model validation failed: {metrics} does not meet thresholds {thresholds}")
    
    return result


def canary_deploy(**context) -> dict:
    """Deploy model to canary with shadow traffic."""
    from pulsenet.config import cfg
    import subprocess
    
    best_model_info = context["ti"].xcom_pull(key="best_model", task_ids="train_models")
    model_name = best_model_info["best_model"]
    
    # Deploy to canary namespace
    canary_cmd = [
        "kubectl", "apply", "-f", "-",
        "-n", "pulsenet-canary",
    ]
    
    # This would apply a canary deployment manifest
    # For now, we simulate the deployment
    log.info(f"Deploying {model_name} to canary")
    
    # Update model registry
    model_registry_path = Path(cfg.models.model_dir) / "model_registry.json"
    registry = {"canary": model_name, "production": "isolation_forest", "updated_at": datetime.utcnow().isoformat()}
    
    if model_registry_path.exists():
        import json
        with open(model_registry_path) as f:
            registry = json.load(f)
    registry["canary"] = model_name
    registry["updated_at"] = datetime.utcnow().isoformat()
    
    with open(model_registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    return {
        "canary_model": model_name,
        "deployed_at": datetime.utcnow().isoformat(),
        "status": "deployed",
    }


def monitor_canary(**context) -> dict:
    """Monitor canary model performance with shadow traffic."""
    import time
    import requests
    from pulsenet.config import cfg
    
    # Wait for canary to stabilize
    time.sleep(300)  # 5 minutes
    
    # Compare canary vs production metrics
    # This would query Prometheus for metrics
    # For now, simulate
    
    canary_metrics = {
        "latency_p99_ms": 45,
        "error_rate": 0.001,
        "drift_score": 0.05,
    }
    
    production_metrics = {
        "latency_p99_ms": 50,
        "error_rate": 0.002,
        "drift_score": 0.03,
    }
    
    # Check if canary is not significantly worse
    thresholds = {
        "latency_p99_ms": 1.2,   # Canary can be 20% slower
        "error_rate": 1.5,       # Canary can have 50% more errors
        "drift_score": 2.0,      # Canary drift can be 2x
    }
    
    passed = all(
        canary_metrics[k] <= production_metrics[k] * v
        for k, v in thresholds.items()
    )
    
    return {
        "canary_metrics": canary_metrics,
        "production_metrics": production_metrics,
        "passed": passed,
        "timestamp": datetime.utcnow().isoformat(),
    }


def promote_model(**context) -> dict:
    """Promote canary model to production."""
    from pulsenet.config import cfg
    import json
    
    canary_result = context["ti"].xcom_pull(key="return_value", task_ids="canary_deploy")
    model_name = canary_result["canary_model"]
    
    # Update model registry
    registry_path = Path(cfg.models.model_dir) / "model_registry.json"
    with open(registry_path) as f:
        registry = json.load(f)
    
    registry["production"] = model_name
    registry["previous_production"] = registry.get("production")
    registry["promoted_at"] = datetime.utcnow().isoformat()
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    # Update Kubernetes deployment
    subprocess.run([
        "kubectl", "set", "image", "deployment/pulsenet-api",
        f"pulsenet-api={registry['production']}",
        "-n", "pulsenet-prod",
    ], check=True)
    
    Variable.set("pulsenet_last_retrain", datetime.utcnow().isoformat())
    
    return {
        "promoted_model": model_name,
        "previous_model": registry.get("previous_production"),
        "promoted_at": registry["promoted_at"],
        "status": "promoted",
    }


def rollback_model(**context) -> dict:
    """Rollback to previous model if canary fails."""
    from pulsenet.config import cfg
    import json
    
    registry_path = Path(cfg.models.model_dir) / "model_registry.json"
    with open(registry_path) as f:
        registry = json.load(f)
    
    previous = registry.get("previous_production")
    if previous:
        registry["production"] = previous
        registry["rolled_back_at"] = datetime.utcnow().isoformat()
        registry["rollback_reason"] = "canary_failed"
        
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        
        subprocess.run([
            "kubectl", "set", "image", "deployment/pulsenet-api",
            f"pulsenet-api={previous}",
            "-n", "pulsenet-prod",
        ], check=True)
    
    return {
        "rolled_back_to": previous,
        "status": "rolled_back",
    }


# === DAG Definition ===

# Task 1: Check for drift
check_drift_task = PythonOperator(
    task_id="check_drift",
    python_callable=check_drift,
    provide_context=True,
    dag=dag,
)

# Task 2: Decision branch
branch_task = PythonOperator(
    task_id="should_retrain",
    python_callable=should_retrain,
    provide_context=True,
    dag=dag,
)

# Task 3a: Retrain path
prepare_data_task = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data,
    provide_context=True,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id="train_models",
    python_callable=train_models,
    provide_context=True,
    dag=dag,
)

validate_model_task = PythonOperator(
    task_id="validate_model",
    python_callable=validate_model,
    provide_context=True,
    dag=dag,
)

canary_deploy_task = PythonOperator(
    task_id="canary_deploy",
    python_callable=canary_deploy,
    provide_context=True,
    dag=dag,
)

monitor_canary_task = PythonOperator(
    task_id="monitor_canary",
    python_callable=monitor_canary,
    provide_context=True,
    dag=dag,
)

promote_model_task = PythonOperator(
    task_id="promote_model",
    python_callable=promote_model,
    provide_context=True,
    dag=dag,
)

# Task 3b: Skip path
skip_retraining_task = BashOperator(
    task_id="skip_retraining",
    bash_command='echo "No significant drift detected, skipping retraining"',
    dag=dag,
)

# Task 4: Rollback (triggered on failure)
rollback_task = PythonOperator(
    task_id="rollback_model",
    python_callable=rollback_model,
    provide_context=True,
    trigger_rule="one_failed",
    dag=dag,
)

# === Task Dependencies ===

check_drift_task >> branch_task

# Retrain path
branch_task >> prepare_data_task >> train_models_task >> validate_model_task >> canary_deploy_task >> monitor_canary_task >> promote_model_task

# Skip path
branch_task >> skip_retraining_task

# Rollback on any failure in retrain path
for task in [prepare_data_task, train_models_task, validate_model_task, canary_deploy_task, monitor_canary_task, promote_model_task]:
    task >> rollback_task

# Rollback also on skip (no-op but keeps DAG structure)
skip_retraining_task >> rollback_task