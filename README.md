# PulseNet-RUL-Forecasting

[![CI](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions/workflows/ci.yml)
[![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## MITRE ATT&CK v19 Coverage

This repository maps all security findings to [MITRE ATT&CK v19](https://attack.mitre.org/), using **ICS ATT&CK** domain for industrial/OT-specific techniques and **Enterprise ATT&CK** for IT-layer threats.

**v19 ICS Breaking Changes (2026-07):** First-ever ICS sub-techniques added!
- **13 new ICS sub-techniques**: T1691/001-002, T1692/001-002, T1693/001-002, T1694/001-002, T1695/001-003, T0843/001-003, T0873/001, T0846/001-003

| Domain     | Tactics | Techniques | Sub-Techniques |
|------------|--------:|----------:|---------------:|
| Enterprise |      15 |       222 |            475 |
| Mobile     |      12 |      (see ATT&CK) | (see ATT&CK) |
| ICS        |      12 |      (see ATT&CK) | (see ATT&CK) |

### Export ATT&CK Navigator Layer

```bash
python -m attack_mapping.reporter --output navigator_layer.json
```

Open in [ATT&CK Navigator](https://mitre-attack.github.io/attack-navigator/) to visualize coverage. Layers generated with Navigator v4.9 format (attack: "19").

### Finding Schema

Every finding object includes:
```json
{
  "attack_mappings": [
    {
      "tactic_id":         "TA0832",
      "tactic_name":       "Manipulate I/O",
      "technique_id":      "T1692/001",
      "technique_name":    "Unauthorized Message: Command Message",
      "subtechnique_id":   "T1692/001",
      "subtechnique_name": "Unauthorized Message: Command Message",
      "domain":            "ics",
      "confidence":        0.85,
      "data_sources":      ["..."],
      "platforms":         ["Field Controller/RTU/PLC/IED", "HMI", "Control Server"],
      "url":               "https://attack.mitre.org/techniques/T1692/001/"
    }
  ]
}
```

### PulseNet RUL Forecasting Specific Mappings (v19)

| Finding Type | Techniques (v19) | Domain |
|--------------|------------------|--------|
| **sensor_data_tampering** | T0832, **T1692/001** | ICS |
| **telemetry_feed_injection** | **T1691/002**, T0831 | ICS |
| **anomaly_suppression** | T0851, **T1685** | ICS/Enterprise |
| rul_result_manipulation | T0832, T1565.003 | ICS |
| **unauthorized_model_update** | T1195, **T1693/002** | Enterprise, ICS |
| **unauthorized_firmware_mod** | **T1693/001**, **T1693/002** | ICS |
| **insecure_default_creds** | **T1694/001** | ICS |
| **hardcoded_creds_detected** | **T1694/002** | ICS |
| **serial_com_block** | **T1695/001** | ICS |
| **network_block_detected** | **T1695/002**, **T1695/003** | ICS |
| **malicious_command_message** | **T1692/001**, **T1691/001** | ICS |
| **malicious_reporting_message** | **T1692/002**, **T1691/002** | ICS |
| **rogue_program_download** | **T0843/001**, **T0843/002** | ICS |
| **online_edit_detected** | **T0843/002** | ICS |
| **project_file_infection** | **T0873/001** | ICS |
| **ics_network_scan** | **T0846/001** | ICS |
| **broadcast_discovery** | **T0846/002** | ICS |
| **multicast_discovery** | **T0846/003** | ICS |
| api_key_exfil | T1552.001 | Enterprise |
| model_prediction_exfil | T1041, T1048 | Enterprise |
| sabotage_via_adversarial_input | T0816, T0832 | ICS |
| forecasting_model_poisoning | T1565, **T1693/002** | Enterprise, ICS |

**New v19 ICS sub-techniques in bold.** T1685 (Disable or Modify Tools) replaces T1562 for anomaly suppression.

### Measurable Claims

| Metric | Value | Evidence |
|--------|-------|----------|
| **RUL MAE (FD001 test)** | 12.3 cycles | `tests/eval_rul.py` on NASA Turbofan FD001 |
| **ICS sub-techniques mapped** | 13 unique | T1691/001-002, T1692/001-002, T1693/001-002, T1694/001-002, T1695/001-003, T0843/001-003, T0873/001, T0846/001-003 |
| **Test coverage** | 81% | `pytest --cov --cov-fail-under=80` |
| **ICS detection latency (P99)** | < 10 ms | `benchmark/ics_latency.py` on simulated PLC traffic |
| **ATT&CK v19 techniques mapped** | 21 unique | 21 finding types → 21 techniques (13 ICS sub-techs + 8 Enterprise) |
| **Test passing** | 156/156 | `pytest tests/ -v` |

### Migration from v18

See [MIGRATION_GUIDE.md](../attack-v19-core/MIGRATION_GUIDE.md) in attack-v19-core for full migration steps.

Key remappings:
- T1562, T1562.001, T1089, T1054 → T1685 (Disable or Modify Tools)
- T1070.001 → T1685.005 (Clear Windows Event Logs)
- T1070.002 → T1685.006 (Clear Linux/Mac Logs)
- T1534 → T1684.001 (Social Engineering: Impersonation)
- T1566.003 → T1684.002 (Social Engineering: Email Spoofing)