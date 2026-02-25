"""
Prediction DoFn for streaming headway pipeline.

Calls the Vertex AI endpoint with a 20-observation encoder window
and returns the prediction (p10, p50, p90) alongside the window.
"""

import logging
import time

import apache_beam as beam
from google.cloud import aiplatform

logger = logging.getLogger(__name__)

ENDPOINT_DISPLAY_NAME = "headway-prediction-endpoint"


class PredictHeadwayFn(beam.DoFn):
    """Call Vertex AI endpoint with a buffered window.

    Input:  {"group_id": str, "observations": list[dict]}
    Output: {"group_id": str, "observations": list[dict],
             "headway_p10": float, "headway_p50": float, "headway_p90": float}
    """

    def __init__(self, project: str, location: str = "us-east1"):
        self._project = project
        self._location = location
        self._endpoint = None

    def setup(self):
        """Resolve endpoint once per worker (not per element)."""
        aiplatform.init(project=self._project, location=self._location)
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"'
        )
        if not endpoints:
            raise RuntimeError(
                f"No endpoint found with display_name='{ENDPOINT_DISPLAY_NAME}'"
            )
        self._endpoint = endpoints[0]
        logger.info("Resolved endpoint: %s", self._endpoint.resource_name)

    def process(self, element):
        group_id = element["group_id"]
        observations = element["observations"]

        try:
            start = time.time()
            response = self._endpoint.predict(instances=[element])
            elapsed_ms = (time.time() - start) * 1000

            pred = response.predictions[0]
            p10 = round(float(pred["headway_p10"]), 2)
            p50 = round(float(pred["headway_p50"]), 2)
            p90 = round(float(pred["headway_p90"]), 2)

            logger.info(
                "Prediction for %s: P10=%.2f P50=%.2f P90=%.2f (%.0fms)",
                group_id, p10, p50, p90, elapsed_ms,
            )

            yield {
                "group_id": group_id,
                "observations": observations,
                "headway_p10": p10,
                "headway_p50": p50,
                "headway_p90": p90,
            }

        except Exception:
            logger.exception("Prediction failed for %s", group_id)
            # Still yield the window without predictions so Firestore gets updated
            yield {
                "group_id": group_id,
                "observations": observations,
                "headway_p10": None,
                "headway_p50": None,
                "headway_p90": None,
            }
