"""Smoke test: verify Hydra config loads with all sections and resolves correctly."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

GlobalHydra.instance().clear()
with hydra.initialize(version_base=None, config_path="../conf"):
    cfg = hydra.compose(config_name="config", overrides=["training=hpo", "+hpo_search_space=vizier_v1"])

# Verify all sections exist
for section in ["model", "training", "processing", "infra", "hpo_search_space"]:
    assert section in cfg, f"Missing section: {section}"

# Verify infra resolves interpolations correctly
assert cfg.infra.project_id == "realtime-headway-prediction"
assert cfg.infra.trial_image_uri.startswith("us-east1-docker.pkg.dev")
assert "hpo_cache" in cfg.infra.hpo_cache_dir

# Verify HPO training profile
assert cfg.training.max_epochs == 15
assert cfg.training.limit_train_batches == 0.5

# Verify search space
assert "model.learning_rate" in cfg.hpo_search_space.parameters
assert cfg.hpo_search_space.metric.id == "val_loss"

print("=== Resolved infra config ===")
print(OmegaConf.to_yaml(cfg.infra))
print("=== HPO search space ===")
print(OmegaConf.to_yaml(cfg.hpo_search_space))
print()
print("All config sections load and resolve correctly.")
