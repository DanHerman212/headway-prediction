"""
Firestore sink for streaming headway prediction pipeline.

Writes each prediction as a new document (append-only) for historical
analysis. No input observations are stored — predictions only.

Document structure (collection: "predictions"):
    doc id: auto-generated
    fields:
      - group_id:                str
      - headway_p10:             float | None
      - headway_p50:             float | None
      - headway_p90:             float | None
      - predicted_at:            timestamp    (server-set)
      - last_observation_time_idx: int         (time_idx of newest encoder obs)
"""

import logging

import apache_beam as beam
from google.cloud import firestore

logger = logging.getLogger(__name__)

COLLECTION = "predictions"


class WriteToFirestoreFn(beam.DoFn):
    """Appends each prediction as a new Firestore document.

    Input: {"group_id": str, "observations": list[dict],
            "headway_p10": float|None, "headway_p50": float|None,
            "headway_p90": float|None}

    Each prediction creates a new document (auto-ID) — no overwrites.
    Only prediction fields are stored; the observation window is not persisted.

    The Firestore client is created per-bundle (setup/teardown) to
    avoid serialization issues with Beam's DoFn lifecycle.
    """

    def __init__(self, project: str, collection: str = COLLECTION,
                 database: str = "headway-streaming"):
        self._project = project
        self._collection = collection
        self._database = database
        self._client = None

    def setup(self):
        self._client = firestore.Client(
            project=self._project, database=self._database
        )

    def process(self, element):
        group_id = element["group_id"]
        observations = element.get("observations", [])

        # Extract time_idx from the most recent observation for traceability
        last_time_idx = None
        if observations:
            last_time_idx = observations[-1].get("time_idx")

        doc_data = {
            "group_id": group_id,
            "predicted_at": firestore.SERVER_TIMESTAMP,
        }

        if last_time_idx is not None:
            doc_data["last_observation_time_idx"] = last_time_idx

        # Include predictions if present
        for field in ("headway_p10", "headway_p50", "headway_p90"):
            if element.get(field) is not None:
                doc_data[field] = element[field]

        # Auto-generated document ID — every prediction is a new document
        self._client.collection(self._collection).add(doc_data)

        p50 = element.get("headway_p50")
        pred_str = f" P50={p50:.2f}" if p50 is not None else ""
        logger.debug("Stored prediction for %s%s", group_id, pred_str)
        yield element

    def teardown(self):
        if self._client:
            self._client.close()
            self._client = None
