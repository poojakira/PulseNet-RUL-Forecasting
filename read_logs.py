import json
import os

LOG_FILE = "secure_log.txt"

def read_secure_logs():
    print(f"--- READING SECURE AUDIT LOG: {LOG_FILE} ---")
    
    if not os.path.exists(LOG_FILE):
        print("Log file not found. Run a simulation first.")
        return

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} log entries.\n")

    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            timestamp = entry.get("timestamp", "Unknown Time")
            params = entry.get("params", {})
            metrics = entry.get("metrics", {})
            audit_hash = entry.get("hash", "NO HASH")

            print(f" Entry #{i+1} [{timestamp}]")
            print(f"   Simulation Mode: {params.get('sim_mode', params.get('simulation_type', 'N/A'))}")
            print(f"   Anomalies Found: {metrics.get('critical_events', metrics.get('sim_anomalies', 0))}")
            print(f"    SHA-256 Audit Hash: {audit_hash[:16]}...") # Show first 16 chars
            print("-" * 50)
            
        except json.JSONDecodeError:
            print(f" Entry #{i+1}: Corrupt Data")

if __name__ == "__main__":
    read_secure_logs()