#!/usr/bin/env python3
"""
PulseNet Edge Robotics Telemetry Bridge
Architected by Rhutvik Pachghare (Robotics Systems & DevOps Engineer)

Description:
This script simulates a ROS (Robot Operating System) edge controller deployed on a 
physical turbofan engine. It continuously transmits live sensor voltages and RPM metrics
to the central PulseNet AI Inference server to receive real-time predictive degradation health.
If the AI scores the hardware health below the critical threshold (50.0), this bridge
executes an automated physical emergency safe-shutdown to prevent catastrophic failure.
"""

import time
import json
import logging
import random
import requests
import sys

# Configure Edge Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [EDGE-NODE-01] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("RobotBridge")

API_URL = "http://localhost:8000"
AUTH_ENDPOINT = f"{API_URL}/token"
PREDICT_ENDPOINT = f"{API_URL}/predict"

def authenticate_hardware(username="operator", password="operator_password"):
    """
    Authenticate the physical hardware node with the PulseNet central server
    using AES-encrypted JWT Tokens.
    """
    log.info(f"Authenticating edge hardware as '{username}'...")
    try:
        response = requests.post(AUTH_ENDPOINT, json={"username": username, "password": password})
        if response.status_code == 200:
            token = response.json().get("access_token")
            log.info("Hardware authentication SUCCESS. Access token acquired.")
            return token
        else:
            log.error(f"Authentication failed: {response.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        log.error("Could not connect to PulseNet API. Is the server running?")
        sys.exit(1)

def trigger_emergency_shutdown(health_score):
    """
    Hardware kill-switch protocol.
    """
    log.error("=" * 60)
    log.error(f"🚨 CRITICAL HARDWARE DEGRADATION RECOGNIZED BY AI (Health: {health_score}%) 🚨")
    log.error("INITIATING PHYSICAL EMERGENCY SHUTDOWN PROTOCOL...")
    log.error("Disengaging primary motors...")
    time.sleep(1)
    log.error("Purging fuel lines...")
    time.sleep(1)
    log.error("Activating mechanical brakes...")
    time.sleep(1)
    log.error("Hardware safely offline. Immediate maintenance required.")
    log.error("=" * 60)
    sys.exit(0)

def simulate_telemetry_stream(token):
    """
    Simulates a 10Hz physical sensor frame loop reading voltage from 14 physical
    channels (temperature, pressure, acoustic vibration, bypass ratios) and sending
    to the AI engine.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # Starting base sensors corresponding to a healthy engine
    base_sensors = {
        "sensor_2": 642.15, "sensor_3": 1589.0, "sensor_4": 1406.2,
        "sensor_7": 554.1, "sensor_8": 2388.0, "sensor_9": 9044.8,
        "sensor_11": 47.5, "sensor_12": 521.9, "sensor_13": 2388.1,
        "sensor_14": 8138.6, "sensor_15": 8.44, "sensor_17": 393.0,
        "sensor_20": 39.0, "sensor_21": 23.4
    }
    
    cycle = 1
    degradation_factor = 0.0
    
    log.info("Starting closed-loop hardware telemetry transmission (1Hz)...")
    
    while True:
        # Simulate slight hardware degradation over time
        degradation_factor += random.uniform(0.1, 0.5)
        
        # Inject physical drift to mimic bearings / fan wear
        payload = {k: v + (random.uniform(-0.1, 0.1) * v) for k, v in base_sensors.items()}
        # Specifically degrade high-pressure compressor temperature (simulated sensor_4)
        payload["sensor_4"] += degradation_factor * 12.0
        
        log.info(f"Cycle {cycle:04} | Transmitting 14-channel sensor frame (HPC Temp: {payload['sensor_4']:.2f})...")
        
        t0 = time.time()
        try:
            res = requests.post(PREDICT_ENDPOINT, json=payload, headers=headers)
            if res.status_code == 200:
                data = res.json()
                latency = (time.time() - t0) * 1000
                health = data.get("health_index", 100.0)
                status = data.get("status", "UNKNOWN")
                
                log.info(f"   [API Reply: {latency:.1f}ms] Health={health:.1f}% | Status={status}")
                
                if health < 50.0:
                    trigger_emergency_shutdown(health)
                    
            else:
                log.warning(f"Unexpected response from AI Server: {res.status_code}")
        except requests.exceptions.RequestException as e:
            log.error(f"Telemetry payload transmission failed: {e}")
            
        cycle += 1
        time.sleep(1.0) # 1Hz control loop

if __name__ == "__main__":
    print(r"""
     _____       _           _   _        _   
    |  __ \     | |         | \ | |      | |  
    | |__) |   _| |___  __ _|  \| | ___  | |_ 
    |  ___/ | | | / __|/ _` | . ` |/ _ \ | __|
    | |   | |_| | \__ \ (_| | |\  |  __/ | |_ 
    |_|    \__,_|_|___/\__,_|_| \_|\___|  \__|
    Edge Robotics Inference Bridge v1.0
    Architected by: Rhutvik Pachghare
    """)
    token = authenticate_hardware()
    simulate_telemetry_stream(token)
