"""Local smoke test for Vizier integration."""
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/Users/danherman/Desktop/headway-prediction")

# 1. Test imports
from mlops_pipeline.src.steps.fetch_best_vizier_params import fetch_best_vizier_params, _cast_vizier_value
print("✅ fetch_best_vizier_params imports OK")

# 2. Test _cast_vizier_value
assert _cast_vizier_value(64.0) == 64 and isinstance(_cast_vizier_value(64.0), int), "int cast failed"
assert _cast_vizier_value(0.001) == 0.001 and isinstance(_cast_vizier_value(0.001), float), "float passthrough failed"
assert _cast_vizier_value("True") is True, "bool True cast failed"
assert _cast_vizier_value("false") is False, "bool False cast failed"
assert _cast_vizier_value("Ranger") == "Ranger", "string passthrough failed"
assert _cast_vizier_value(0.0) == 0, "zero int cast failed"  # edge case
assert isinstance(_cast_vizier_value(0.0), int), "zero should be int"
print("✅ _cast_vizier_value type casting OK")

# 3. Test OmegaConf.update pattern with a mock config matching our YAML structure
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    "model": {
        "learning_rate": 0.001,
        "hidden_size": 128,
        "dropout": 0.1,
        "attention_head_size": 4,
        "hidden_continuous_size": 64,
    },
    "training": {
        "batch_size": 128,
        "max_epochs": 100,
    },
})

# Simulate Vizier output (dot-notation keys, values already cast)
vizier_params = {
    "model.learning_rate": 0.0005,
    "model.hidden_size": 256,
    "model.dropout": 0.15,
    "model.attention_head_size": 8,
    "model.hidden_continuous_size": 128,
    "training.batch_size": 512,
}

for key, value in vizier_params.items():
    OmegaConf.update(cfg, key, value)

assert cfg.model.learning_rate == 0.0005, f"lr={cfg.model.learning_rate}"
assert cfg.model.hidden_size == 256, f"hidden_size={cfg.model.hidden_size}"
assert cfg.model.dropout == 0.15
assert cfg.model.attention_head_size == 8
assert cfg.model.hidden_continuous_size == 128
assert cfg.training.batch_size == 512
# max_epochs should be UNTOUCHED (Vizier doesn't tune it)
assert cfg.training.max_epochs == 100, "max_epochs should not be modified"
print("✅ OmegaConf.update with Vizier dot-notation keys OK")

# 4. Verify Hydra CLI overrides still win AFTER Vizier
# Simulate: user passes training.max_epochs=15 on CLI, Vizier sets batch_size=512
# Load base, apply Vizier, then Hydra overrides on top
cfg2 = OmegaConf.create({
    "model": {"learning_rate": 0.001, "hidden_size": 128},
    "training": {"batch_size": 128, "max_epochs": 100},
})
# Vizier
OmegaConf.update(cfg2, "model.learning_rate", 0.0005)
OmegaConf.update(cfg2, "training.batch_size", 512)
# User CLI override (applied later by Hydra before the step sees it)
OmegaConf.update(cfg2, "training.max_epochs", 15)
# User can also override a Vizier param if they want
OmegaConf.update(cfg2, "model.learning_rate", 0.01)

assert cfg2.model.learning_rate == 0.01, "User override should win"
assert cfg2.training.batch_size == 512, "Vizier param should stick when no user override"
assert cfg2.training.max_epochs == 15, "CLI override should apply"
print("✅ Override precedence (Vizier then user CLI) OK")

# 5. Test pipeline.py imports
from mlops_pipeline.pipeline import headway_training_pipeline
print("✅ pipeline.py imports OK")

# 6. Test run.py argparse
import argparse
# Just verify the module loads without error
import importlib
run_mod = importlib.import_module("mlops_pipeline.run")
print("✅ run.py imports OK")

print()
print("Final config after Vizier overrides:")
print(OmegaConf.to_yaml(cfg))
print("All tests passed.")
