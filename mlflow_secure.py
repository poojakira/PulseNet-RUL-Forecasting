import mlflow
import hashlib
from datetime import datetime
import json
import os

def log_mlflow_secure(params, metrics, artifacts=None, file="secure_log.txt"):
   
    # 1. Create a secure local log entry (Audit Trail)
    timestamp = str(datetime.utcnow())
    entry = {"timestamp": timestamp, "params": params, "metrics": metrics}
    
    # Generate Hash for integrity verification
    entry_str = json.dumps(entry, sort_keys=True)
    entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
    
    # Append to local secure log
    with open(file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    # 2. Log to MLflow (The Dashboard Backend)
    # Set the experiment name
    mlflow.set_experiment("NASA_FD001_Predictive_Maintenance")
    
    with mlflow.start_run():
        # Log Hyperparameters (e.g., n_estimators, window_size)
        for k, v in params.items():
            mlflow.log_param(k, v)
        
        # Log Performance Metrics (e.g., Anomaly Ratio, Accuracy)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            
        # Log Artifacts (Plots, CSVs, Model Files)
        if artifacts:
            for artifact in artifacts:
                if os.path.exists(artifact):
                    mlflow.log_artifact(artifact)
                else:
                    print(f"Warning: Artifact {artifact} not found.")

    print(f"Secure logging complete. Audit hash: {entry['hash']}")



