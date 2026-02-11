from zenml import step
from google.cloud import aiplatform_v1
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def _cast_vizier_value(value: Union[float, str]) -> Union[int, float, bool, str]:
    """
    Heuristic to cast Vizier output (often generic floats/strings)
    back to strict Python types for Hydra/Pydantic compatibility.
    """
    # 1. Handle Booleans (Vizier treats them as Categorical Strings)
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        return value

    # 2. Handle Integers (Vizier returns Integers as Floats, e.g., 64.0)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return value

    return value


@step
def fetch_best_vizier_params(
    project_id: str,
    location: str,
    study_display_name: str,
) -> Dict[str, Any]:
    """
    Robustly queries Vertex AI Vizier for the best trial and formats
    parameters for direct injection into Hydra via OmegaConf.update.
    """
    api_endpoint = f"{location}-aiplatform.googleapis.com"
    client = aiplatform_v1.VizierServiceClient(
        client_options={"api_endpoint": api_endpoint}
    )

    # Locate the study
    parent = f"projects/{project_id}/locations/{location}"
    studies = client.list_studies(parent=parent)
    target_study = next(
        (s for s in studies if s.display_name == study_display_name), None
    )

    if not target_study:
        raise ValueError(f"Vizier Study '{study_display_name}' not found.")

    # Get the best trial
    optimal_trials = client.list_optimal_trials(parent=target_study.name)
    if not optimal_trials:
        raise RuntimeError(f"Study '{study_display_name}' has no completed optimal trials.")

    best_trial = optimal_trials[0]

    # Extract and Cast Parameters
    best_params = {}
    print(f"--- Loading Best Params from Trial {best_trial.name} ---")

    for param in best_trial.parameters:
        raw_val = param.value
        clean_val = _cast_vizier_value(raw_val)

        if type(raw_val) != type(clean_val):
            logger.debug(
                "Casting %s: %s (%s) -> %s (%s)",
                param.parameter_id, raw_val, type(raw_val).__name__,
                clean_val, type(clean_val).__name__,
            )

        best_params[param.parameter_id] = clean_val
        print(f"  • {param.parameter_id}: {clean_val}")

    return best_params
