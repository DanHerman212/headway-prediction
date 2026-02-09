"""
verify_deployment.py

Minimal test pipeline to verify the ZenML + MLflow deployment is working
end-to-end. If this succeeds, the run will appear in:
  - ZenML dashboard (your Cloud Run ZenML URL)
  - MLflow UI (your Cloud Run MLflow URL)

Usage:
    python infra/verify_deployment.py
"""

from zenml import pipeline, step

try:
    import mlflow
except ImportError:
    raise ImportError(
        "mlflow is not installed locally. Run: pip install mlflow"
    )


@step(experiment_tracker="mlflow_tracker")
def log_test_metrics() -> dict:
    """Logs test parameters and metrics to the remote MLflow server."""

    mlflow.log_param("deployment_target", "cloud_run")
    mlflow.log_param("database_engine", "mysql_8.0")
    mlflow.log_param("driver", "pymysql")
    mlflow.log_metric("test_accuracy", 0.95)
    mlflow.log_metric("test_loss", 0.05)

    print("✓ Parameters and metrics logged to remote MLflow")

    return {
        "status": "success",
        "deployment_target": "cloud_run",
    }


@step
def validate_result(result: dict) -> None:
    """Validates the output from the tracking step."""
    assert result["status"] == "success", "Verification failed!"
    print("✓ Pipeline output validated")
    print("✓ End-to-end verification PASSED")
    print("")
    print("Check your dashboards:")
    print("  - ZenML: run `zenml show` to open the dashboard")
    print("  - MLflow: visit your MLflow Cloud Run URL")


@pipeline
def verification_pipeline():
    """Minimal pipeline to verify ZenML + MLflow are connected."""
    result = log_test_metrics()
    validate_result(result)


if __name__ == "__main__":
    verification_pipeline()
