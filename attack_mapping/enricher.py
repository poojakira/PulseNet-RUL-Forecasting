"""
ATT&CK Enricher for PulseNet-RUL-Forecasting.
Uses ICS ATT&CK domain where applicable.
"""
from attack_core.index import ATTACKIndex
from attack_core.models import ATTACKMapping, Domain
from typing import List, Dict, Any


class ATTACKEnricher:
    def __init__(self, index: ATTACKIndex):
        self.index = index
        self._rule_table = {
            "sensor_data_tampering": ["T0832", "T0831"],
            "forecasting_model_poisoning": ["T0839", "T0820"],
            "telemetry_feed_injection": ["T0830", "T0831"],
            "anomaly_suppression": ["T0851", "T0800"],
            "rul_result_manipulation": ["T0832"],
            "unauthorized_model_update": ["T1195", "T0839"],
            "api_key_exfil": ["T1552.001"],
            "model_prediction_exfil": ["T1041", "T1048"],
            "sabotage_via_adversarial_input": ["T0816", "T0832"],
        }

    def enrich(self, finding_type: str, metadata: Dict[str, Any]) -> List[ATTACKMapping]:
        technique_ids = self._rule_table.get(finding_type, [])
        mappings = []
        for tid in technique_ids:
            tech = self.index.get(tid)
            if tech:
                tactic = self.index._tactics.get(tech.tactic_ids[0] if tech.tactic_ids else "", None)
                mappings.append(ATTACKMapping(
                    tactic_id=tech.tactic_ids[0] if tech.tactic_ids else "unknown",
                    tactic_name=tactic.name if tactic else "unknown",
                    technique_id=tech.attack_id,
                    technique_name=tech.name,
                    subtechnique_id=tech.attack_id if tech.is_subtechnique else None,
                    subtechnique_name=tech.name if tech.is_subtechnique else None,
                    domain=tech.domain,
                    confidence=metadata.get("confidence", 0.5),
                    data_sources=tech.data_sources,
                    platforms=tech.platforms,
                    url=tech.url,
                ))
        return mappings