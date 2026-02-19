# Optimize Workspace — Migration Plan

## Overview

Replace ZenML with native **Kubeflow Pipelines (KFP) on Vertex AI** and introduce
**Prefect** for batch ingestion orchestration.  The goal is a leaner stack with
fewer moving parts, faster developer iteration, and full artifact
tracking / lineage through Vertex AI Experiments.

---

## 1. Remove ZenML

### What Goes Away
- `zenml` Python dependency and all `@step` / `@pipeline` decorators
- ZenML Cloud Run server (`Dockerfile.zenml`, `start_zenml.sh`, `register_stack.sh`)
- ZenML-specific Hydra configs (stack references, orchestrator settings)
- MLflow tracking server (`Dockerfile.mlflow`, `start_mlflow.sh`) — superseded by
  Vertex AI Experiments + TensorBoard

### What Replaces It

| Concern | Current (ZenML) | New (KFP + Vertex AI) |
|---|---|---|
| Pipeline definition | `@pipeline` / `@step` decorators | `kfp.dsl.pipeline` / `@kfp.dsl.component` |
| Orchestration | ZenML → Vertex AI Pipelines | KFP SDK → Vertex AI Pipelines directly |
| Artifact tracking | ZenML artifact store (GCS) | Vertex ML Metadata (auto-logged by KFP) |
| Experiment tracking | Vertex Experiment Tracker (via ZenML) | `aiplatform.start_run()` / `log_metrics()` natively |
| Model registry | Vertex AI Model Registry (unchanged) | Vertex AI Model Registry (unchanged) |
| Caching | ZenML step caching | KFP component caching (built-in) |

### Migration Steps
1. Convert each ZenML step to a KFP `@component` (lightweight Python or
   container-based):
   - `load_config` → pipeline parameter injection (no component needed)
   - `ingest_data` → `ingest_data_op` (KFP component)
   - `process_data` → `process_data_op`
   - `train_model` → `train_model_op` (GPU container component, `a2-highgpu-1g`)
   - `evaluate_model` → `evaluate_model_op`
   - `register_model` → `register_model_op`
2. Define pipeline DAG in a single `pipeline.py` using `@kfp.dsl.pipeline`.
3. Wire artifact I/O through KFP `Input[Dataset]` / `Output[Model]` types for
   automatic lineage.
4. Log metrics to Vertex AI Experiments inside each component via
   `aiplatform.log_metrics()`.
5. Compile pipeline to JSON, submit via `aiplatform.PipelineJob`.

---

## 2. Prefect for Batch Ingestion

### Current State
Manual scripts run ad-hoc:
- `download_historical_data.py` — fetch compressed CSVs from subwaydata.nyc
- `load_to_bigquery_monthly.py` — load CSVs → BQ `raw` table
- SQL transforms (`01_create_clean_table.sql`, `02_feature_engineering.sql`)
- `generate_dataset` Beam pipeline — BQ → training parquet

### New: Prefect Flow
Create a single Prefect flow (`batch_ingestion/flows/ingest_training_data.py`)
that orchestrates the full ETL:

```
@flow
def ingest_training_data(start_date, end_date):
    download_csvs(start_date, end_date)    # task
    load_to_bigquery()                      # task
    run_sql_transforms()                    # task: 01 → 02
    generate_training_parquet()             # task: Beam → GCS parquet
```

- Run locally or on a Prefect worker (Cloud Run job for production)
- Schedule monthly retraining data refreshes
- Track each run with Prefect UI for visibility

---

## 3. CLI Tool for Developer Velocity

Create `cli.py` at the repo root — a single entry point for all MLOps
operations, built with `click` or `typer`.

### Commands

```
# Build & push images
headway build training          # Build + push training image to Artifact Registry
headway build serving           # Build + push serving image

# Submit pipelines
headway train                   # Compile + submit KFP training pipeline to Vertex AI
headway train --hpo             # Submit HPO pipeline (Vizier study)
headway train --local           # Run training locally (DirectRunner equivalent)

# Deployment
headway deploy model            # Find latest model → create endpoint → deploy → monitoring
headway deploy model --skip-monitoring
headway deploy streaming        # Provision Pub/Sub + poller VM + Dataflow pipeline
headway deploy streaming --local  # Local DirectRunner for integration testing

# Operations
headway status                  # Endpoint health, Dataflow job state, poller status
headway logs [streaming|poller] # Tail logs
headway pause                   # Stop poller
headway resume                  # Start poller

# Data
headway ingest                  # Trigger Prefect batch ingestion flow
headway clean-bq                # Truncate monitoring tables
headway clean-firestore         # Clear Firestore predictions
```

### Implementation
- Each command is a thin wrapper calling existing logic (moved from scripts/ and
  Makefile targets into importable modules)
- `setup.py` or `pyproject.toml` registers `headway` as a console entry point
- Replaces the Makefile for most operations (Makefile can delegate to `headway` CLI
  for backward compatibility)

---

## 4. Programmatic Model Deployment

Refactor `scripts/deploy_endpoint.py` into an importable module
(`mlops_pipeline/src/deploy/`) so it can be called from:
- The CLI (`headway deploy model`)
- A KFP post-training component (automatic deploy after successful evaluation)
- CI/CD (Cloud Build trigger on model registry event)

### Module Structure
```
mlops_pipeline/src/deploy/
    __init__.py
    endpoint.py       # create/update endpoint, deploy model
    monitoring.py     # schema upload, monitoring job creation
    smoke_test.py     # synthetic request validation
    streaming.py      # Pub/Sub + poller VM + Dataflow provisioning
```

---

## 5. Programmatic Streaming Pipeline Deployment

Extract the Makefile `deploy-infra` / `start-ingestion` logic into
`mlops_pipeline/src/deploy/streaming.py`:

- `provision_infrastructure()` — Pub/Sub topic/sub, poller VM, Dataflow staging
- `launch_pipeline(runner="DataflowRunner")` — submit streaming pipeline
- `teardown()` — drain Dataflow, delete VM + Pub/Sub

Callable from `headway deploy streaming` and programmatically from CI/CD.

---

## 6. Artifact Tracking & Lineage

With KFP on Vertex AI, artifact tracking is automatic:

| Artifact | KFP Type | Tracked How |
|---|---|---|
| Training parquet | `Input[Dataset]` / `Output[Dataset]` | Vertex ML Metadata |
| Processed datasets | `Output[Dataset]` | Vertex ML Metadata |
| Trained model | `Output[Model]` | Vertex ML Metadata → Model Registry |
| ONNX export | `Output[Artifact]` | GCS URI in model metadata |
| Evaluation metrics | `Output[Metrics]` | Vertex AI Experiments |
| Evaluation plots | `Output[HTML]` | Vertex AI Experiments |
| dataset_params.json | `Output[Artifact]` | Packaged with model artifacts |

Each pipeline run is a Vertex AI Experiment run with full lineage:
`data → processing → model → evaluation → registry`.

---

## 7. Files to Delete

After migration is complete, remove:
- `mlops_pipeline/hpo_pipeline.py` (replaced by KFP HPO pipeline)
- `mlops_pipeline/pipeline.py` (replaced by KFP pipeline definition)
- `mlops_pipeline/run.py` (replaced by CLI)
- `infra/Dockerfile.zenml`, `infra/Dockerfile.mlflow`
- `infra/start_zenml.sh`, `infra/start_mlflow.sh`
- `infra/deploy_zenml.sh`, `infra/deploy_mlflow.sh`
- `infra/register_stack.sh`
- `infra/cloudbuild_zenml.yaml`, `infra/cloudbuild_mlflow.yaml`
- `infra/zenml-cloudrun-service.yaml`
- `infra/setup_mlops_infra.sh` (replaced by CLI + IaC)
- `scripts/deploy_endpoint.py` (logic moves to `mlops_pipeline/src/deploy/`)

---

## 8. Execution Order

| # | Task | Est. Time |
|---|---|---|
| 1 | Set up CLI skeleton (`cli.py` with `click`) | 30 min |
| 2 | Convert ZenML steps → KFP components | 2 hr |
| 3 | Define KFP training pipeline + compile | 1 hr |
| 4 | Wire Vertex AI Experiments tracking | 30 min |
| 5 | Create Prefect batch ingestion flow | 1 hr |
| 6 | Refactor deploy logic into modules | 1 hr |
| 7 | Wire CLI commands to new modules | 1 hr |
| 8 | End-to-end test: `headway train` → `headway deploy model` | 1 hr |
| 9 | Clean up deleted files, update README | 30 min |

---

## Dependencies to Add

```
kfp>=2.7.0
google-cloud-pipeline-components>=2.14.0
prefect>=3.0
click>=8.1
```

## Dependencies to Remove

```
zenml[server,templates,gcp]
mlflow
```
