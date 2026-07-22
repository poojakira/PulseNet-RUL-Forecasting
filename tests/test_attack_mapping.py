import pytest
from attack_core import ATTACKLoader, ATTACKIndex, Domain
from attack_mapping.enricher import ATTACKEnricher


@pytest.fixture
def enricher():
    loader = ATTACKLoader()
    index = ATTACKIndex(loader)
    return ATTACKEnricher(index)


class TestPulseNetEnricher:
    def test_sensor_tampering(self, enricher):
        mappings = enricher.enrich("sensor_data_tampering", {"confidence": 0.9})
        technique_ids = [m.technique_id for m in mappings]
        assert "T0832" in technique_ids
        assert "T0831" in technique_ids

    def test_anomaly_suppression(self, enricher):
        mappings = enricher.enrich("anomaly_suppression", {"confidence": 0.85})
        technique_ids = [m.technique_id for m in mappings]
        assert "T0851" in technique_ids
        assert "T0800" in technique_ids

    def test_sabotage(self, enricher):
        mappings = enricher.enrich("sabotage_via_adversarial_input", {"confidence": 0.9})
        technique_ids = [m.technique_id for m in mappings]
        assert "T0816" in technique_ids
        assert "T0832" in technique_ids

    def test_ics_domain_used(self, enricher):
        mappings = enricher.enrich("sensor_data_tampering", {"confidence": 0.9})
        for m in mappings:
            if m.technique_id in ("T0832", "T0831"):
                assert m.domain == Domain.ICS