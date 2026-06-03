#!/usr/bin/env python3
"""
demo.py - PulseNet-RUL-Forecasting (Adversarial Telemetry Guard)
Author: Pooja Kiran (github.com/poojakira)

Run: python demo.py  (no GPU, no API keys, no NASA data download)

Dataset: NASA C-MAPSS FD001 (official)
Source: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq
Reference: f7i.ai 2026 (sensor masking attacks), MITRE ATT&CK for ICS T0839
NIST AI RMF 1.0 control register: docs/nist_ai_rmf_controls.yaml
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.pulsenet.security.adversarial_telemetry_guard import (
    AdversarialTelemetryGuard, TelemetryAlertCode, TelemetryGuardConfig
)

SEP = "-" * 65


def main():
    print("=" * 65)
    print("PulseNet-RUL-Forecasting - Adversarial Telemetry Guard Demo")
    print("Author: Pooja Kiran | github.com/poojakira")
    print("Dataset: NASA C-MAPSS FD001 (official)")
    print("Reference: f7i.ai 2026, MITRE ATT&CK ICS T0839")
    print("=" * 65)

    cfg = TelemetryGuardConfig(
        frozen_consecutive_threshold=3,
        max_single_step_delta=0.5,
        bias_sigma_threshold=3.0,
        baseline_window=20,
    )

    # TEST 1: Frozen sensor attack
    print(f"\n{SEP}")
    print("TEST 1: Frozen sensor - attacker masks sensor output")
    guard = AdversarialTelemetryGuard(n_sensors=3, config=cfg)
    base = [0.1, 0.2, 0.3]
    alerts = []
    for _ in range(5):
        alerts = guard.check(base)
    frozen = [a for a in alerts if a.code == TelemetryAlertCode.FROZEN_SENSOR]
    print(f"  5 identical frames -> frozen alerts={len(frozen)} [{'PASS' if frozen else 'FAIL'}]")
    if frozen:
        print(f"  sensor_index={frozen[0].sensor_index} severity={frozen[0].severity}")
    assert frozen, "Frozen sensor not detected"

    # TEST 2: Replay attack
    print(f"\n{SEP}")
    print("TEST 2: Replay attack - attacker resubmits old frame")
    guard2 = AdversarialTelemetryGuard(n_sensors=3, config=cfg)
    frame = [0.5, 0.6, 0.7]
    guard2.check(frame)
    alerts2 = guard2.check(frame)  # exact replay
    replay = [a for a in alerts2 if a.code == TelemetryAlertCode.REPLAY]
    print(f"  Exact replay -> replay alerts={len(replay)} [{'PASS' if replay else 'FAIL'}]")
    if replay:
        print(f"  severity={replay[0].severity}")
    assert replay, "Replay attack not detected"

    # TEST 3: Sudden jump (bias injection)
    print(f"\n{SEP}")
    print("TEST 3: Sudden jump - sensor 0 injected with large delta")
    guard3 = AdversarialTelemetryGuard(n_sensors=3, config=cfg)
    guard3.check([0.1, 0.2, 0.3])
    alerts3 = guard3.check([0.9, 0.2, 0.3])  # sensor 0 delta=0.8 > threshold=0.5
    jump = [a for a in alerts3 if a.code == TelemetryAlertCode.SUDDEN_JUMP]
    print(f"  Sensor 0 delta=0.8 -> jump alerts={len(jump)} [{'PASS' if jump else 'FAIL'}]")
    if jump:
        print(f"  sensor_index={jump[0].sensor_index} severity={jump[0].severity}")
    assert jump, "Sudden jump not detected"

    # TEST 4: Wrong sensor count raises ValueError
    print(f"\n{SEP}")
    print("TEST 4: Wrong sensor count raises ValueError (input validation)")
    try:
        guard3.check([0.1, 0.2])  # only 2 sensors instead of 3
        print("  [FAIL] Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  [PASS] ValueError: {str(e)}")

    print(f"\n{SEP}")
    print("ALL 4 TESTS PASS")
    print()
    print("NOTE: Z-score bias detection assumes approximate stationarity.")
    print("For long degradation sequences, CUSUM/ADWIN is planned (see ROADMAP.md).")
    print("NIST AI RMF controls: docs/nist_ai_rmf_controls.yaml")
    print(SEP)


if __name__ == "__main__":
    main()
