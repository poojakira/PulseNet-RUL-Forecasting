# Data Lineage

## Official Source

- Dataset: NASA C-MAPSS Jet Engine Simulated Data
- Landing page: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Archive URL: https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- Local archive: `data/official/CMAPSSData.zip`
- SHA-256: `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`

## Files Used

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

FD001 contains 100 training engine trajectories and 100 test trajectories under
one operating condition and one fault mode.

## Controls

- The loader verifies the archive SHA-256 before extraction.
- Zip extraction rejects path traversal.
- Raw rows are parsed with the 26-column C-MAPSS schema.
- Tests and CI read official NASA data only.
- Generated data fixtures are not used.

## Evidence Policy

Metrics in this repository are evidence only when produced by checked-in
commands against the official archive. Do not add unverifiable performance,
latency, GPU, or production claims to README or GitHub About text.
