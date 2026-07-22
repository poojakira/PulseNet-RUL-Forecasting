## MITRE ATT&CK v19 Coverage

This repository maps all security findings to [MITRE ATT&CK v19](https://attack.mitre.org/), using **ICS ATT&CK** domain for industrial/OT-specific techniques and **Enterprise ATT&CK** for IT-layer threats.

| Domain     | Tactics | Techniques | Sub-Techniques |
|------------|--------:|----------:|---------------:|
| Enterprise |      15 |       222 |            475 |
| Mobile     |      12 |      (see ATT&CK) | (see ATT&CK) |
| ICS        |      12 |      (see ATT&CK) | (see ATT&CK) |

### Export ATT&CK Navigator Layer

```bash
python -m attack_mapping.reporter --output navigator_layer.json
```

Open in [ATT&CK Navigator](https://mitre-attack.github.io/attack-navigator/) to visualize coverage.

### Finding Schema

Every finding object includes:
```json
{
  "attack_mappings": [
    {
      "tactic_id":         "TA0832",
      "tactic_name":       "Manipulate I/O",
      "technique_id":      "T0832",
      "technique_name":    "Manipulate I/O",
      "subtechnique_id":   null,
      "subtechnique_name": null,
      "domain":            "ics",
      "confidence":        0.85,
      "data_sources":      ["..."],
      "platforms":         ["Field Controller/RTU/PLC/IED", "HMI", "Control Server"],
      "url":               "https://attack.mitre.org/techniques/T0832/"
    }
  ]
}
```

### PulseNet RUL Forecasting Specific Mappings

| Finding Type | Techniques | Domain |
|--------------|------------|--------|
| sensor_data_tampering | T0832, T0831 | ICS |
| forecasting_model_poisoning | T0839, T0820 | ICS |
| telemetry_feed_injection | T0830, T0831 | ICS |
| anomaly_suppression | T0851, T0800 | ICS |
| rul_result_manipulation | T0832 | ICS |
| unauthorized_model_update | T1195, T0839 | Enterprise, ICS |
| api_key_exfil | T1552.001 | Enterprise |
| model_prediction_exfil | T1041, T1048 | Enterprise |
| sabotage_via_adversarial_input | T0816, T0832 | ICS |