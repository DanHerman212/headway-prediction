"""Quick test: verify _save_dataset_params produces non-empty normalizer_params."""
import json
import tempfile
import sys
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

sys.path.insert(0, ".")
from mlops_pipeline.src.steps.deploy_model import _save_dataset_params

# Load training data
df = pd.read_parquet("local_artifacts/processed_data/training_data.parquet")
# Apply same cleaning as data_processing.clean_data()
cats = ["route_id", "regime_id", "track_id", "preceding_route_id", "group_id"]
for c in cats:
    df[c] = df[c].fillna("None").astype(str)
for c in ["preceding_train_gap", "upstream_headway_14th", "travel_time_14th", "travel_time_34th"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
for c in df.columns:
    if "deviation" in c:
        df[c] = df[c].fillna(0.0)
if "travel_time_23rd" in df.columns:
    df["travel_time_23rd"] = df["travel_time_23rd"].fillna(df["travel_time_23rd"].median())
print(f"Loaded {len(df)} rows")
print(f"Unique group_ids: {sorted(df['group_id'].unique())}")

# Reconstruct the TimeSeriesDataSet exactly as in training
training = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="service_headway",
    group_ids=["group_id"],
    min_encoder_length=10,
    max_encoder_length=20,
    min_prediction_length=1,
    max_prediction_length=1,
    static_categoricals=["route_id"],
    time_varying_known_categoricals=["regime_id", "track_id"],
    time_varying_known_reals=[
        "time_idx", "day_of_week", "hour_sin", "hour_cos", "empirical_median",
    ],
    time_varying_unknown_categoricals=["preceding_route_id"],
    time_varying_unknown_reals=[
        "service_headway", "preceding_train_gap", "upstream_headway_14th",
        "travel_time_14th", "travel_time_14th_deviation",
        "travel_time_23rd", "travel_time_23rd_deviation",
        "travel_time_34th", "travel_time_34th_deviation", "stops_at_23rd",
    ],
    target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
)

# Test the fixed _save_dataset_params end-to-end
tmpdir = tempfile.mkdtemp()
path = _save_dataset_params(training, tmpdir)

with open(path) as f:
    result = json.load(f)

np_ = result["normalizer_params"]
print(f"normalizer_params keys: {list(np_.keys())}")
print(f"center: {np_.get('center', {})}")
print(f"scale: {np_.get('scale', {})}")
print()

# Verify all 4 group_ids are present
expected = {"A_South", "C_South", "E_South", "OTHER_South"}
actual_center = set(np_.get("center", {}).keys())
actual_scale = set(np_.get("scale", {}).keys())

if actual_center == expected and actual_scale == expected:
    print("TEST PASSED")
else:
    print(f"TEST FAILED — expected {expected}, got center={actual_center}, scale={actual_scale}")
    sys.exit(1)
