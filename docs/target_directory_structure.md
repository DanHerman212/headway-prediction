# Target Directory Structure

> Reference for the workspace cleanup after the mobile app is complete.
> Everything below replaces the current organic layout.

## Directory Tree

```
headway-prediction/
├── pyproject.toml                  # Single dependency source (replaces all requirements.txt + setup.py)
├── Makefile                        # Human entry point: make train, make deploy, make test, etc.
├── README.md
├── .github/
│   └── workflows/
│       ├── ci.yaml                 # Lint + unit tests on push
│       └── deploy.yaml             # Build containers + deploy on tag
│
├── infra/                          # All GCP infrastructure (Terraform)
│   ├── main.tf                     # Root module
│   ├── variables.tf
│   ├── outputs.tf
│   ├── terraform.tfvars.example
│   └── modules/
│       ├── pubsub/                 # Topic + subscription
│       ├── firestore/              # Database + indexes
│       ├── vertex_ai/              # Endpoint + model registry
│       ├── compute/                # Poller VM template
│       ├── iam/                    # Service accounts + bindings
│       └── storage/                # GCS buckets (artifacts, staging, side inputs)
│
├── src/
│   ├── ingestion/                  # GTFS-RT poller (runs on VM)
│   │   ├── Dockerfile
│   │   ├── poller.py               # Core polling loop
│   │   ├── run_ace.py              # ACE-only entry point
│   │   └── arrival_detector.py     # ArrivalDetector + track cache
│   │
│   ├── streaming/                  # Apache Beam streaming pipeline
│   │   ├── Dockerfile
│   │   ├── pipeline.py             # Pipeline DAG definition
│   │   ├── transforms/
│   │   │   ├── transforms.py       # Shared feature engineering DoFns
│   │   │   ├── window_buffer.py    # Rolling 20-obs encoder window
│   │   │   ├── predict.py          # Vertex AI endpoint caller
│   │   │   └── firestore_sink.py   # Firestore writer
│   │   └── side_inputs/
│   │       └── build_side_inputs.py  # Compute empirical_map + median_tt_map
│   │
│   ├── training/                   # ML training pipeline (ZenML)
│   │   ├── Dockerfile
│   │   ├── pipeline.py             # ZenML pipeline definition
│   │   ├── run.py                  # CLI entry point
│   │   ├── conf/                   # Hydra configs
│   │   │   ├── config.yaml
│   │   │   ├── model/
│   │   │   └── processing/
│   │   └── steps/
│   │       ├── data_processing.py
│   │       ├── train.py
│   │       ├── evaluate.py
│   │       ├── onnx_export.py
│   │       └── deploy.py
│   │
│   ├── serving/                    # ONNX prediction server (Vertex AI container)
│   │   ├── Dockerfile
│   │   └── predictor.py
│   │
│   └── batch/                      # Batch dataset generation (Beam)
│       └── generate_dataset.py
│
├── mobile/                         # Flutter app (separate build target)
│   ├── pubspec.yaml
│   ├── lib/
│   ├── android/
│   ├── ios/
│   └── test/
│
├── notebooks/                      # EDA only — no production logic
│   ├── eda_processed_data.ipynb
│   ├── feature_target_eda.ipynb
│   ├── baseline_analysis.ipynb
│   └── eda_utils.py
│
├── scripts/                        # One-off utilities (not production)
│   ├── test_endpoint.py
│   ├── check_firestore_windows.py
│   ├── inspect_gtfs.py
│   └── generate_baseline.py
│
├── tests/
│   ├── unit/
│   │   ├── test_transforms.py
│   │   ├── test_arrival_detector.py
│   │   ├── test_window_buffer.py
│   │   └── test_data_processing.py
│   └── integration/
│       └── test_pipeline_local.py
│
└── docs/
    ├── architecture.md             # System diagram + data flow
    ├── data_representation.md
    ├── runbook.md                  # Operational procedures
    └── decisions/                  # ADRs (architecture decision records)
        ├── 001_tft_over_convlstm.md
        ├── 002_beam_shared_transforms.md
        └── 003_firestore_native_mode.md
```

## What Gets Deleted

| Current Path | Reason |
|---|---|
| `archive/` | Dead code (ConvLSTM experiment). In git history. |
| `bash/` | Replaced by Makefile targets + Terraform |
| `infra/*.sh` | Replaced by Terraform modules |
| `infra/Dockerfile.*` | Moved to per-component Dockerfiles in `src/` |
| `infra/cloudbuild_*.yaml` | Replaced by GitHub Actions |
| `infra/*zenml*` | ZenML infra managed by Terraform |
| `local_artifacts/` | Add to `.gitignore`. Training outputs stay in GCS. |
| `lightning_logs/` | Add to `.gitignore`. Ephemeral training logs. |
| `tensorboard_logs/` | Add to `.gitignore`. |
| `headway_prediction_pipeline.egg-info/` | Build artifact. Add to `.gitignore`. |
| `archive/ml_pipelines/` | Superseded by `src/training/` |
| `archive/data/`, `archive/json_files/` | Test fixtures, not needed |
| Multiple `requirements.txt` | Replaced by `pyproject.toml` dependency groups |
| `setup.py` | Replaced by `pyproject.toml` |
| `staging_area/`, `workflows/` | Unused |

## pyproject.toml Dependency Groups

```toml
[project]
name = "headway-prediction"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []  # No base deps — everything is optional group

[project.optional-dependencies]
streaming = [
    "apache-beam[gcp]>=2.50",
    "google-cloud-firestore>=2.11",
    "google-cloud-aiplatform>=1.25",
    "google-cloud-storage>=2.10",
    "protobuf>=4.21",
    "gtfs-realtime-bindings>=1.0",
]
training = [
    "pytorch-lightning>=2.0",
    "pytorch-forecasting>=1.0",
    "zenml[gcp]>=0.93",
    "hydra-core>=1.3",
    "onnx>=1.14",
    "onnxruntime>=1.15",
    "pandas>=2.0",
    "pyarrow>=12.0",
]
serving = [
    "flask>=3.0",
    "onnxruntime>=1.15",
    "numpy>=1.24",
    "google-cloud-storage>=2.10",
]
dev = [
    "pytest>=7.0",
    "ruff>=0.1",
    "jupyter>=1.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
]
```

## Makefile Targets (Final)

```makefile
# Infrastructure
make infra-plan         # terraform plan
make infra-apply        # terraform apply
make infra-destroy      # terraform destroy

# Training
make train              # Full ZenML training pipeline
make train-quick        # 5 epoch test run
make build-side-inputs  # Recompute empirical_map + median_tt_map

# Streaming
make pipeline-local     # DirectRunner against real Pub/Sub
make pipeline-deploy    # DataflowRunner
make pipeline-stop      # Cancel Dataflow job

# Ingestion
make poller-start       # Start poller VM
make poller-stop        # Stop poller VM
make poller-logs        # Tail poller logs

# Monitoring
make check-firestore    # One-shot Firestore snapshot
make watch-firestore    # Poll Firestore every 30s

# Integration Test (current)
make up                 # Full local integration test
make down               # Tear down everything
make logs               # Tail pipeline log

# Mobile
make app-run            # flutter run
make app-build          # flutter build apk / ipa

# Development
make test               # pytest
make lint               # ruff check + ruff format --check
make fmt                # ruff format
```

## Migration Order

1. Set up `pyproject.toml` — install with `pip install -e ".[streaming,training,dev]"`
2. Move source files into `src/` layout, update all imports
3. Move tests to `tests/`
4. Write Terraform modules, validate with `terraform plan`
5. Set up GitHub Actions CI
6. Delete `archive/`, dead scripts, stale configs
7. Update README with new structure + quickstart
