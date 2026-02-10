"""
test_arg_utils.py
-----------------
Unit tests for mlops_pipeline.src.arg_utils.

These tests have ZERO heavy dependencies (no torch, hydra, etc.)
and should run in < 1 second.
"""

import os
import sys
import unittest

# Ensure the src directory is importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hpo_entrypoint import vizier_args_to_hydra_overrides, parse_and_convert_args


class TestVizierArgsToHydraOverrides(unittest.TestCase):
    """Tests for the low-level conversion function."""

    def test_equals_separated(self):
        raw = ["--model.learning_rate=0.01", "--training.batch_size=64"]
        self.assertEqual(
            vizier_args_to_hydra_overrides(raw),
            ["model.learning_rate=0.01", "training.batch_size=64"],
        )

    def test_boolean_flag(self):
        self.assertEqual(vizier_args_to_hydra_overrides(["--verbose"]), ["verbose=true"])

    def test_empty_input(self):
        self.assertEqual(vizier_args_to_hydra_overrides([]), [])

    def test_stray_positional_ignored(self):
        self.assertEqual(vizier_args_to_hydra_overrides(["stray", "--a=1"]), ["a=1"])

    def test_realistic_vizier_payload(self):
        """Exact args Vizier sent in the failed jobs."""
        raw = [
            "--model.attention_head_size=4",
            "--model.hidden_continuous_size=64",
            "--training.batch_size=256",
            "--model.learning_rate=0.001",
            "--model.dropout=0.175",
            "--model.hidden_size=128",
        ]
        result = vizier_args_to_hydra_overrides(raw)
        self.assertEqual(len(result), 6)
        self.assertTrue(all("=" in o for o in result))

    def test_no_double_equals_regression(self):
        """The original bug: --key=val produced key=val=true."""
        result = vizier_args_to_hydra_overrides(["--model.learning_rate=0.01"])
        self.assertEqual(result, ["model.learning_rate=0.01"])
        # Exactly one '=' in the key=value portion
        self.assertEqual(result[0].count("="), 1)


class TestParseAndConvertArgs(unittest.TestCase):
    """Tests for the full parse_and_convert_args entry point."""

    def test_known_args_extracted(self):
        raw = [
            "--train_dataset_path", "gs://bucket/train.pt",
            "--val_dataset_path", "gs://bucket/val.pt",
        ]
        known, overrides = parse_and_convert_args(raw)
        self.assertEqual(known.train_dataset_path, "gs://bucket/train.pt")
        self.assertEqual(known.val_dataset_path, "gs://bucket/val.pt")
        self.assertEqual(overrides, [])

    def test_known_plus_vizier_args(self):
        raw = [
            "--train_dataset_path", "gs://bucket/train.pt",
            "--val_dataset_path", "gs://bucket/val.pt",
            "--model.lr=0.01",
            "--training.batch_size=256",
        ]
        known, overrides = parse_and_convert_args(raw)
        self.assertEqual(known.train_dataset_path, "gs://bucket/train.pt")
        self.assertEqual(overrides, ["model.lr=0.01", "training.batch_size=256"])

    def test_end_to_end_realistic(self):
        """Full realistic payload as Vizier would send it (equals-separated)."""
        raw = [
            "--train_dataset_path", "gs://mlops-artifacts/hpo_cache/train.pt",
            "--val_dataset_path", "gs://mlops-artifacts/hpo_cache/val.pt",
            "--model.learning_rate=0.00058972749080905915",
            "--model.hidden_size=128",
            "--model.dropout=0.13629917967695293",
            "--model.attention_head_size=2",
            "--model.hidden_continuous_size=32",
            "--training.batch_size=512",
        ]
        known, overrides = parse_and_convert_args(raw)
        self.assertEqual(known.train_dataset_path, "gs://mlops-artifacts/hpo_cache/train.pt")
        self.assertEqual(len(overrides), 6)
        # Verify no double-equals anywhere
        for o in overrides:
            parts = o.split("=", 1)
            self.assertFalse(parts[1].endswith("=true"),
                             f"Malformed override: {o!r}")


if __name__ == "__main__":
    unittest.main()
