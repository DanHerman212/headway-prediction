# Data Pipeline Architecture for ML Dataset Creation

## Overview

This document describes the 3-stage batch data pipeline for creating a Graph WaveNet training dataset from MTA transit data on Google Cloud Platform.

---

## Data Sources

| Source | Size | Files | Update Frequency | Ingestion Method |
|--------|------|-------|------------------|------------------|
| **Historic Arrivals** | ~10MB/day | ~120 files (4 months) | Daily/Weekly | Python → decompress → partition → BigQuery |
| **Historic Schedules** | ~9GB | 1 CSV | One-time | Python + requests → Cloud Storage |
| **Historic Alerts** | ~120MB | 1 CSV | Archive for training | Python + requests → Cloud Storage |
| **Static GTFS** | ~40MB | 6 .txt files | Rarely | Python + requests → unzip → Cloud Storage |

---

## Architecture Diagram

```mermaid
flowchart TB
    subgraph SOURCES["External Data Sources"]
        A1[("MTA Arrivals\nArchive")]
        A2[("Historic Schedules\nHTTP Endpoint")]
        A3[("Historic Alerts\nHTTP Endpoint")]
        A4[("Static GTFS\nHTTP Endpoint")]
    end

    subgraph STAGE1["Stage 1: Ingestion (Cloud Run Jobs)"]
        CR1["download_arrivals.py\n~120 files"]
        CR2["download_schedules.py\n~9GB CSV"]
        CR3["download_alerts.py\n~120MB CSV"]
        CR4["download_gtfs.py\n~40MB ZIP"]
    end

    subgraph GCS_STAGING["Cloud Storage - Staging"]
        S1["gs://bucket/staging/arrivals/"]
        S2["gs://bucket/staging/schedules/"]
        S3["gs://bucket/staging/alerts/"]
        S4["gs://bucket/staging/gtfs/"]
    end

    subgraph GCS_RAW["Cloud Storage - Partitioned"]
        R1["gs://bucket/raw/arrivals/2025-06/\ngs://bucket/raw/arrivals/2025-07/\n..."]
    end

    subgraph STAGE2["Stage 2: Load & Transform (BigQuery)"]
        BQ1[("mta_raw\n├── arrivals\n├── schedules\n├── alerts\n└── gtfs_*")]
        SQL1["SQL Transforms:\n• Clean arrivals\n• Join stops\n• Compute headways\n• Compute lateness"]
        BQ2[("mta_transformed\n├── headways\n├── schedule_deviation\n└── alerts_windowed")]
    end

    subgraph STAGE3["Stage 3: Dataset Creation (Dataflow)"]
        DF1["Apache Beam Pipeline:\n• 5-min time bins\n• Node mapping\n• Impute missing\n• Time embeddings\n• Sliding windows\n• Adjacency matrix"]
    end

    subgraph OUTPUT["ML Dataset (Cloud Storage)"]
        OUT1["gs://bucket/ml-dataset/\n├── train/X.npy, Y.npy\n├── val/X.npy, Y.npy\n├── test/X.npy, Y.npy\n├── adjacency_matrix.npy\n└── node_mapping.json"]
    end

    subgraph ORCHESTRATION["Orchestration"]
        WF["Cloud Workflows\n+ Cloud Scheduler"]
    end

    A1 --> CR1
    A2 --> CR2
    A3 --> CR3
    A4 --> CR4

    CR1 --> S1
    CR2 --> S2
    CR3 --> S3
    CR4 --> S4

    S1 -->|decompress\npartition| R1
    R1 --> BQ1
    S2 --> BQ1
    S3 --> BQ1
    S4 --> BQ1

    BQ1 --> SQL1
    SQL1 --> BQ2

    BQ2 --> DF1
    DF1 --> OUT1

    WF -.->|orchestrates| STAGE1
    WF -.->|orchestrates| STAGE2
    WF -.->|orchestrates| STAGE3
```

---

## Infrastructure Components

| Component | GCP Service | Purpose |
|-----------|-------------|---------|
| Ingestion scripts | **Cloud Run Jobs** | Containerized Python, pay-per-use |
| File storage | **Cloud Storage** | Staging, raw, and final datasets |
| Data warehouse | **BigQuery** | SQL transforms, joins, aggregations |
| Tensor creation | **Cloud Dataflow** | Apache Beam for complex processing |
| Orchestration | **Cloud Workflows** | Chain pipeline stages |
| Scheduling | **Cloud Scheduler** | Trigger weekly ingestion |

---

## Pipeline Stages

### Stage 1: Ingestion

**Goal:** Download all source data to Cloud Storage

| Job | Input | Output | Frequency |
|-----|-------|--------|-----------|
| `download_arrivals` | MTA archive API | `gs://bucket/raw/arrivals/{yyyy-mm}/` | Weekly (Sunday) |
| `download_schedules` | HTTP endpoint | `gs://bucket/staging/schedules/` | One-time |
| `download_alerts` | HTTP endpoint | `gs://bucket/staging/alerts/` | One-time (archive) |
| `download_gtfs` | HTTP endpoint | `gs://bucket/staging/gtfs/` | One-time |

### Stage 2: Load & Transform

**Goal:** Clean data, compute derived features in BigQuery

| Transform | Input Tables | Output Table | Logic |
|-----------|--------------|--------------|-------|
| Clean arrivals | `arrivals` | `arrivals_cleaned` | Filter route_id ∈ {A, C, E}, parse timestamps |
| Join stops | `arrivals_cleaned` + `gtfs_stops` | `arrivals_with_stops` | Add stop_name for interpretability |
| Compute headways | `arrivals_with_stops` | `headways` | `arrival_time - LAG(arrival_time)` per node |
| Compute lateness | `arrivals_with_stops` + `schedules` | `schedule_deviation` | `actual - scheduled` per trip/stop |
| Window alerts | `alerts` | `alerts_windowed` | Aggregate to 5-min bins per route |

### Stage 3: Dataset Creation

**Goal:** Build tensors for Graph WaveNet training

| Step | Description |
|------|-------------|
| Time binning | Aggregate to 5-minute intervals |
| Node mapping | Create (station, line, direction) → node_id |
| Imputation | Forward-fill missing headways |
| Time features | sin/cos encoding of time-of-day, day-of-week |
| Sliding windows | Create (X, Y) pairs: 12 steps in → 12 steps out |
| Adjacency matrix | Build graph from route topology |
| Train/Val/Test split | Temporal split (no shuffle) |

---

## Cost Optimization

| Decision | Rationale |
|----------|-----------|
| Cloud Run Jobs (not VMs) | Pay only during execution |
| BigQuery on-demand | Pay per TB scanned; small dataset = low cost |
| Dataflow FlexRS | 40% cheaper for batch jobs |
| Weekly batch (not daily) | Fewer job runs |
| Bash setup (not Terraform) | Simpler for small infra footprint |

---

## Weekly Operations

```mermaid
flowchart LR
    A["Cloud Scheduler\n(Sunday 2 AM)"] --> B["Cloud Workflow"]
    B --> C["Download new arrivals\n(7 files)"]
    C --> D["Decompress & partition"]
    D --> E["Load to BigQuery"]
    E --> F["Incremental transform"]
    F --> G["Rebuild tensors\n(if needed)"]
```

---

## File Locations

| Type | Path |
|------|------|
| **Staging** | `gs://{project}-data/staging/{source}/` |
| **Raw partitioned** | `gs://{project}-data/raw/arrivals/{yyyy-mm}/` |
| **ML dataset** | `gs://{project}-data/ml-dataset/{version}/` |

---

## Next Steps

1. ✅ Create project scaffolding
2. ⏳ Run `setup_gcp.sh` to create GCP resources
3. ⏳ Add existing download scripts
4. ⏳ Write SQL transforms
5. ⏳ Build Dataflow pipeline
6. ⏳ Configure Cloud Workflows
7. ⏳ Test end-to-end
