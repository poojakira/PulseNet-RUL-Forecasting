"""
ATT&CK Enricher for PulseNet-RUL-Forecasting.
Uses ICS ATT&CK domain where applicable.
"""

from typing import Any

from attack_core.index import ATTACKIndex
from attack_core.mapping import ATTACKMappingBuilder
from attack_core.models import ATTACKMapping


class ATTACKEnricher:
    def __init__(self, index: ATTACKIndex):
        self.index = index
        self.mapping_builder = ATTACKMappingBuilder(index)
        self._rule_table = {
            # Sensor and telemetry layer
            "sensor_data_tampering": ["T0832", "T1692/001", "T0831"],
            "telemetry_feed_injection": ["T1691/002", "T0831"],
            "anomaly_suppression": ["T0851", "T1685", "T0800"],
            # Firmware and program integrity
            "unauthorized_model_update": ["T1195", "T1693/002"],
            "unauthorized_firmware_mod": ["T1693/001", "T1693/002"],
            "insecure_default_creds": ["T1694/001"],
            "hardcoded_creds_detected": ["T1694/002"],
            # Communications blocking
            "serial_com_block": ["T1695/001"],
            "network_block_detected": ["T1695/002", "T1695/003"],
            # OT message attacks
            "malicious_command_message": ["T1692/001", "T1691/001"],
            "malicious_reporting_message": ["T1692/002", "T1691/002"],
            # Program download attacks
            "rogue_program_download": ["T0843/001", "T0843/002"],
            "online_edit_detected": ["T0843/002"],
            # Project file attacks (Siemens-specific)
            "project_file_infection": ["T0873/001"],
            # Discovery in ICS context
            "ics_network_scan": ["T0846/001"],
            "broadcast_discovery": ["T0846/002"],
            "multicast_discovery": ["T0846/003"],
            # Enterprise layer threats retained
            "rul_result_manipulation": ["T0832", "T1565.003"],
            "api_key_exfil": ["T1552.001"],
            "model_prediction_exfil": ["T1041", "T1048"],
            "sabotage_via_adversarial_input": ["T0816", "T0832"],
            "forecasting_model_poisoning": ["T1565", "T1693/002"],
        }

    def enrich(
        self, finding_type: str, metadata: dict[str, Any]
    ) -> list[ATTACKMapping]:
        confidence = metadata.get("confidence", 0.5)
        technique_ids = self._rule_table.get(finding_type, [])
        return self.mapping_builder.build_many(technique_ids, confidence)
