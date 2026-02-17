"""
Firestore sink for streaming headway prediction pipeline.

Writes the latest 20-observation window and prediction per group_id
to Firestore for real-time serving.

Document structure (collection: "headway_windows"):
    doc id: group_id (e.g. "A_South")
    fields:
      - group_id:      str
      - observations:   list[dict]  (20 enriched records)
      - headway_p10:   float | None
      - headway_p50:   float | None
      - headway_p90:   float | None
      - updated_at:    timestamp    (server-set)
"""

import logging

import apache_beam as beam
from google.cloud import firestore

logger = logging.getLogger(__name__)

COLLECTION = "headway_windows"


class WriteToFirestoreFn(beam.DoFn):
    """Writes a complete window + prediction to Firestore, keyed by group_id.

    Input: {"group_id": str, "observations": list[dict],
            "headway_p10": float|None, "headway_p50": float|None,
            "headway_p90": float|None}

    Each write overwrites the previous document for that group_id,
    so Firestore always holds the most recent window and prediction.

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
        observations = element["observations"]

        doc_data = {
            "group_id": group_id,
            "observations": observations,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }

        # Include predictions if present
        for field in ("headway_p10", "headway_p50", "headway_p90"):
            if element.get(field) is not None:
                doc_data[field] = element[field]

        doc_ref = self._client.collection(self._collection).document(group_id)
        doc_ref.set(doc_data)

        p50 = element.get("headway_p50")
        pred_str = f" (P50={p50:.2f})" if p50 is not None else ""
        logger.debug("Wrote window for %s (%d obs)%s", group_id, len(observations), pred_str)
        yield element

    def teardown(self):
        if self._client:
            self._client.close()
            self._client = None
