# Changelog - PulseNet-RUL-Forecasting

## [1.0.0] - 2026-07-22

### Changed - ATT&CK v19 Migration (ICS Domain - Major Update)

**ICS ATT&CK v19 introduced sub-techniques for the first time.** Complete rule table rewrite.

#### Technique Remappings
| Old ID | New ID | Notes |
|--------|--------|-------|
| T1562 (Enterprise) | T1685 | anomaly_suppression |

#### New ICS Sub-techniques Added (13 new IDs)
| Parent | Sub-techniques | Rule Table Keys |
|--------|----------------|-----------------|
| T1691 | /001 Command Message, /002 Reporting Message | sensor_data_tampering, telemetry_feed_injection, malicious_command_message, malicious_reporting_message |
| T1692 | /001 Command Message, /002 Reporting Message | malicious_command_message, malicious_reporting_message |
| T1693 | /001 System Firmware, /002 Module Firmware | unauthorized_firmware_mod, forecasting_model_poisoning, unauthorized_model_update |
| T1694 | /001 Default Credentials, /002 Hardcoded Credentials | insecure_default_creds, hardcoded_creds_detected |
| T1695 | /001 Serial COM, /002 Ethernet, /003 Wi-Fi | serial_com_block, network_block_detected |
| T0843 | /001 Download All, /002 Online Edit, /003 Program Append | rogue_program_download, online_edit_detected |
| T0873 | /001 Siemens Project File Format | project_file_infection |
| T0846 | /001 Port Scan, /002 Broadcast Discovery, /003 Multicast Discovery | ics_network_scan, broadcast_discovery, multicast_discovery |

#### Rule Table - Complete Rewrite (v19 ICS Structure)
```python
# NEW ICS v19 Rule Table
_rule_table = {
    # Sensor and telemetry layer
    "sensor_data_tampering":        ["T0832", "T1692/001"],
    "telemetry_feed_injection":     ["T1691/002", "T0831"],
    "anomaly_suppression":          ["T0851", "T1685"],

    # Firmware and program integrity
    "unauthorized_model_update":    ["T1195", "T1693/002"],
    "unauthorized_firmware_mod":    ["T1693/001", "T1693/002"],
    "insecure_default_creds":       ["T1694/001"],
    "hardcoded_creds_detected":     ["T1694/002"],

    # Communications blocking
    "serial_com_block":             ["T1695/001"],
    "network_block_detected":       ["T1695/002", "T1695/003"],

    # OT message attacks
    "malicious_command_message":    ["T1692/001", "T1691/001"],
    "malicious_reporting_message":  ["T1692/002", "T1691/002"],

    # Program download attacks
    "rogue_program_download":       ["T0843/001", "T0843/002"],
    "online_edit_detected":         ["T0843/002"],

    # Project file attacks (Siemens-specific)
    "project_file_infection":       ["T0873/001"],

    # Discovery in ICS context
    "ics_network_scan":             ["T0846/001"],
    "broadcast_discovery":          ["T0846/002"],
    "multicast_discovery":          ["T0846/003"],

    # Enterprise layer threats retained
    "rul_result_manipulation":      ["T0832", "T1565.003"],
    "api_key_exfil":                ["T1552.001"],
    "model_prediction_exfil":       ["T1041", "T1048"],
    "sabotage_via_adversarial_input": ["T0816", "T0832"],
    "forecasting_model_poisoning":  ["T1565", "T1693/002"],
}
```

### Added
- First-ever ICS sub-technique detection coverage (13 new sub-technique IDs)
- Complete ICS v19 technique mapping across OT message, firmware, communications, credentials, and discovery domains

### Migration
**CRITICAL**: This is a complete rule table rewrite. Legacy ICS rules using flat T08xx IDs must be updated to use new sub-technique IDs. See [attack-v19-core MIGRATION_GUIDE.md](../attack-v19-core/MIGRATION_GUIDE.md) for full migration steps.