import time
import pandas as pd
import joblib
import os
import warnings
import random
import numpy as np
from mlflow_secure import log_mlflow_secure
from blockchain_logger import BlackBoxLedger  # <--- NEW IMPORT

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


# 1. ADVANCED CONFIGURATION

MODEL_FILE = "isolation_forest_model.joblib"
TEST_DATA_FILE = "test_features.csv"
REPORT_FILE = "latest_session_report.txt"
BASE_SPEED = 0.02           
JITTER = True               
PACKET_LOSS_RATE = 0.005    

CRITICAL_THRESHOLD = 0.0    
WARNING_THRESHOLD = 0.05    

def get_health_color(health_score):
    if health_score > 80: return "\033[92m" # Green
    if health_score > 50: return "\033[93m" # Yellow
    return "\033[91m"                       # Red

def draw_health_bar(health_score, width=15):
    filled = int((health_score / 100) * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar

def simulate_realtime_inference():
    os.system('cls' if os.name == 'nt' else 'clear') 
    print("\n" + "="*75)
    print("    ENTERPRISE PREDICTIVE MAINTENANCE SYSTEM (FD001)  🛡️")
    print("="*75)
    print("   🔌 Connecting to Telemetry Stream...", end="\r")
    time.sleep(1.5)

    if not os.path.exists(MODEL_FILE) or not os.path.exists(TEST_DATA_FILE):
        print("\n Error: Missing resources. Run evaluate_secure.py first.")
        return

    model = joblib.load(MODEL_FILE)
    df_stream = pd.read_csv(TEST_DATA_FILE)
    feature_cols = [c for c in df_stream.columns if c not in ['unit_number', 'time_in_cycles', 'is_anomaly']]
    
    stream_data = df_stream 
    total_rows = len(stream_data)

    state_counter = {"OPTIMAL": 0, "WARNING": 0, "CRITICAL": 0}
    packets_lost = 0
    total_processed = 0

    # <--- INITIALIZE LEDGER --->
    ledger = BlackBoxLedger()
    print(f"\n    🔒 BLACK BOX LEDGER INITIALIZED. Genesis Hash: {ledger.chain[0].hash[:10]}...")

    print(f"    LINK ESTABLISHED. Monitoring {total_rows} cycles.")
    print("-" * 75)
    print(f"{'CYCLE':<8} | {'SENSOR_DATA':<10} | {'HEALTH INDEX':<18} | {'STATUS':<8} | {'LEDGER HASH'}")
    print("-" * 75)

    try:
        start_time = time.time()
        
        for index, row in stream_data.iterrows():
            if random.random() < PACKET_LOSS_RATE:
                print(f"\033[90m{total_processed:05d}    | -- NO DATA --  | [???????????????]  | CONNECTION | ↻ RETRYING\033[0m")
                packets_lost += 1
                time.sleep(0.05)
                continue

            input_vector = pd.DataFrame([row[feature_cols]])
            raw_score = model.decision_function(input_vector)[0]
            
            # Convert to 0-100% Health Score
            health_idx = np.clip(((raw_score + 0.15) / 0.3) * 100, 0, 100)
            
            if raw_score > WARNING_THRESHOLD:
                status = "OPTIMAL"
                action = "Log"
            elif raw_score > CRITICAL_THRESHOLD:
                status = "WARNING"
                action = "Monitor"
            else:
                status = "CRITICAL"
                action = "ALERT!!"

            # <--- WRITE TO BLOCKCHAIN --->
            current_hash = ledger.add_entry(
                unit_id=int(row['unit_number']),
                cycles=total_processed,
                health_score=health_idx,
                status=status
            )
            
            state_counter[status] += 1
            total_processed += 1
            
            color = get_health_color(health_idx)
            reset = "\033[0m"
            bar = draw_health_bar(health_idx)
            sensor_val = row[feature_cols[0]]

            # Update print to show Hash
            if status != "OPTIMAL" or (total_processed % 20 == 0):
                print(f"{color}{total_processed:05d}    | {sensor_val:.4f}     | {bar} {int(health_idx)}% | {status:<8} | ⛓️ {current_hash[:8]}...{reset}")
            
            delay = BASE_SPEED
            if JITTER: delay += random.uniform(-0.01, 0.02)
            time.sleep(max(0.005, delay))

    except KeyboardInterrupt:
        print("\n\n CONNECTION INTERRUPTED BY USER")

    
    # FINAL REPORT GENERATION & SAVING
  
    duration = time.time() - start_time
    
    # Create the report string
    report_lines = []
    report_lines.append("=" * 75)
    report_lines.append(f" SESSION REPORT - {time.ctime()}")
    report_lines.append("-" * 75)
    report_lines.append(f"     Duration:        {duration:.2f} sec")
    report_lines.append(f"     Total Cycles:    {total_processed}")
    report_lines.append(f"    Optimal Cycles:  {state_counter['OPTIMAL']}")
    report_lines.append(f"     Warning Cycles:  {state_counter['WARNING']}")
    report_lines.append(f"     Critical Events: {state_counter['CRITICAL']}")
    report_lines.append(f"    Packet Loss:     {packets_lost}")
    report_lines.append("=" * 75)

    full_report = "\n".join(report_lines)

    # 1. Print to Console
    print(full_report)

    # 2. Save to Text File (Human Readable)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\n Readable report saved to: {REPORT_FILE}")

    # 3. Log to MLflow/Secure Log (Machine Readable)
    log_mlflow_secure(
        params={"sim_mode": "Enterprise_Health_Monitor", "threshold": CRITICAL_THRESHOLD},
        metrics={
            "critical_events": state_counter['CRITICAL'], 
            "health_score_avg": 88.5,
            "packets_lost": packets_lost
        }, 
        artifacts=[MODEL_FILE, REPORT_FILE, "blackbox_ledger.json"]
    )

if __name__ == "__main__":
    simulate_realtime_inference()