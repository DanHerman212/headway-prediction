from zenml import step
from google.cloud import aiplatform
from omegaconf import DictConfig
from typing import Dict, Any, Union
import math
import logging

logger = logging.getLogger(__name__)

def _cast_vizier_value(value: Union[float, str]) -> Union[int, float, bool, str]:
    """
    Casts Vizier output to strict Python types for Hydra compatibility.
    """
    if isinstance(value, str):
        if value.lower() == "true": return True
        if value.lower() == "false": return False
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value

@step
def fetch_best_vizier_params(config: DictConfig) -> Dict[str, Any]:
    """
    Queries Vertex AI for the best trial to MINIMIZE validation loss.
    """
    project_id = config.infra.project_id
    location = config.infra.location
    study_display_name = config.infra.study_display_name

    aiplatform.init(project=project_id, location=location)

    # 1. Efficiency: Filter on server side
    try:
        # Note: 'limit' is not supported by the Python SDK's list() method, 
        # so we fetch all successful jobs matching the name and pick the latest one below.
        jobs = aiplatform.HyperparameterTuningJob.list(
            filter=f'display_name="{study_display_name}" AND state="JOB_STATE_SUCCEEDED"',
            order_by="create_time desc",
        )
    except Exception as e:
        logger.error(f"Failed to list HyperparameterTuningJobs: {e}")
        raise

    if not jobs:
        raise ValueError(f"No successful HPO job found for '{study_display_name}'")
    
    target_job = jobs[0]
    
    # 2. Filter for successful trials only
    valid_trials = [t for t in target_job.trials if t.state.name == "SUCCEEDED"]
    if not valid_trials:
        raise RuntimeError(f"Job '{target_job.display_name}' has no successful trials.")

    # 3. SELECT BEST TRIAL (Minimization Logic)
    # Python's default sort is Ascending (Smallest -> Largest).
    # Since we want MINIMUM loss, we take the first element [0].
    
    def get_metric(t):
        val = t.final_measurement.metrics[0].value
        # Safety: Treat NaN/Inf as infinity so they go to the end of the list
        if math.isnan(val) or math.isinf(val):
            return float('inf')
        return val

    best_trial = sorted(valid_trials, key=get_metric)[0]

    logger.info(
        f"Selected Best Trial {best_trial.id} with Validation Loss: "
        f"{best_trial.final_measurement.metrics[0].value}"
    )

    # 4. Extract Params
    best_params = {}
    for param in best_trial.parameters:
        best_params[param.parameter_id] = _cast_vizier_value(param.value)
        print(f"  • {param.parameter_id}: {best_params[param.parameter_id]}")

    return best_params
