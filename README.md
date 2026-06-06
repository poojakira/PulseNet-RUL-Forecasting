PulseNet — Secure RUL Forecasting for Safety‑Critical Telemetry

PulseNet is a security‑aware Remaining Useful Life (RUL) forecasting pipeline designed for aviation, industrial IoT, satellites, and other safety‑critical systems where telemetry integrity directly determines operational safety.

Modern assets rely on telemetry streams to estimate RUL.
If telemetry is corrupted, delayed, replayed, or silently drifted, RUL forecasts become untrustworthy — and failures can be catastrophic. 

PulseNet treats telemetry as a high‑value security asset, combining:

Data lineage & integrity verification

Drift & anomaly detection

Secure RUL modeling

Append‑only audit logging

Supply‑chain security artifacts (SBOM + SARIF)

🚀 Quickstart
bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting
cd PulseNet-RUL-Forecasting
pip install -r requirements.txt

# Run secure pipeline
python main_pipeline.py --config config.example.yaml
🧠 Core Capabilities
1. Data Lineage & Integrity
PulseNet performs:

SHA‑256 hashing of raw telemetry

Hashing of preprocessed features

Hash‑chained lineage linking telemetry → features → model inputs

Replay detection via timestamp monotonicity checks

(Your README already describes hashing + lineage; this section formalizes it.) 

2. Drift & Anomaly Detection
PulseNet detects:

Statistical drift in key telemetry features

Distribution shifts beyond configured thresholds

Missing or truncated data segments

(Directly based on your “drift checks” and “missing/truncated detection”.) 

3. RUL Modeling
Supports:

Sequence models (LSTM/GRU/Transformer)

Regression on engineered features

Evaluation on held‑out telemetry

(From your “RUL modeling” section.) 

4. Audit Logging
Append‑only audit log

Hash‑chained entries

Detects insider tampering or unauthorized modification

(From your “audit logging” section.) 

🛡️ Threat Model
PulseNet’s threat model includes the following (all sourced from your README):

Assets
Telemetry streams

Preprocessed feature sets

RUL models & predictions

Audit logs & lineage metadata

Adversaries
Network‑adjacent attacker injecting/dropping/replaying telemetry

Insider modifying intermediate data

Misconfigured or compromised upstream services

Attack Surfaces
Telemetry ingestion endpoints

Intermediate feature stores

Model input pipeline

Logging/monitoring infrastructure

Defended (In Scope)
Tampering/replay detection via lineage + hashing

Drift detection

Missing/truncated segment detection

Not Defended (Out of Scope)
Physical sensor spoofing

Model extraction/IP theft

Full host compromise

For full details, see docs/threat_model.md.

🔐 Threat → Control Mapping
Threat	PulseNet Control
Telemetry replay	Hash‑chain lineage + timestamp monotonicity
Data tampering	SHA‑256 hashing of raw + processed data
Silent drift	Statistical drift detection
Missing/truncated segments	Sequence completeness validator
Insider modification	Append‑only hash‑chained audit log


🏗️ Architecture
A high‑level architecture diagram should be placed here:

Code
docs/architecture.png
(We will generate this when we reach /docs.)

📊 Evaluation Metrics
Component	Metric	Value
RUL Model	MAE	TBD
Drift Detector	KL‑div threshold	TBD
Lineage Integrity	Hash‑chain verification	100%
Missing Data Detection	Recall	TBD


🧪 Demo
bash
python demo.py
This runs:

Drift detection

Lineage verification

RUL prediction

Audit logging

🧩 Supply‑Chain Security
PulseNet ships with:

sbom.json — Software Bill of Materials

sarif_output.json — Static analysis results

These can be integrated into CI/CD gates for artifact integrity.

🗺️ Roadmap (Summary)
From ROADMAP.md:

Transformer‑based RUL models

Real‑time streaming ingestion

Anomaly localization

Grafana dashboards for drift monitoring

📁 Repository Structure
Code
├── main_pipeline.py
├── verify.py
├── demo.py
├── src/
├── docs/
├── sbom.json
├── sarif_output.json
├── provenance.json
├── config.example.yaml
└── smoke_test.sh

📜 License
MIT License


## Security & Limitations
This project is a research prototype and is not intended for production use. It has not been formally audited and may contain vulnerabilities. Specific limitations include:
- No formal guarantees of security or robustness.
- May not protect against all classes of attacks.


### Threat Model
This section outlines the assumed attacker capabilities and the scope of protection. We assume a "white-box" attacker with access to the model and data, but not necessarily the training infrastructure. We do not explicitly protect against zero-day exploits or highly sophisticated, targeted attacks beyond the scope of typical research prototypes.


## Data, Privacy, and Ethics
This project uses data that is either synthetic, publicly available, or anonymized. No sensitive personal data is used unless explicitly stated and justified. Users should be aware of the ethical implications of deploying ML models and ensure compliance with relevant privacy regulations.


## Supply Chain Security
To ensure the integrity of dependencies, we recommend running `pip-audit` or `safety` regularly. For model artifacts, hashes and verification steps should be documented to prevent tampering.


## Threat Model for Predictive Maintenance
**Threat Model**: We consider attackers who might attempt to poison telemetry data, spoof sensor readings, or inject malicious updates into the predictive maintenance models. Such attacks could lead to incorrect RUL predictions, potentially causing equipment failure or unnecessary maintenance.
