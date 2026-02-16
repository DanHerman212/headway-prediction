"""
Firestore sink for streaming headway prediction pipeline.

Writes the latest 20-observation window per group_id to Firestore
so the serving layer can read it for real-time inference.

Document structure (collection: "headway_windows"):
    doc id: group_id (e.g. "A_South")
    fields:
      - group_id:     str
      - observations:  list[dict]  (20 enriched records)
      - updated_at:   timestamp    (server-set)
"""

import logging

import apache_beam as beam
from google.cloud import firestore

logger = logging.getLogger(__name__)

COLLECTION = "headway_windows"


class WriteToFirestoreFn(beam.DoFn):
    """Writes a complete window to Firestore, keyed by group_id.

    Input: {"group_id": str, "observations": list[dict]}

    Each write overwrites the previous document for that group_id,
    so Firestore always holds the most recent window.

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

        doc_ref = self._client.collection(self._collection).document(group_id)
        doc_ref.set({
            "group_id": group_id,
            "observations": observations,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

        logger.debug("Wrote window for %s (%d obs)", group_id, len(observations))
        yield element

    def teardown(self):
        if self._client:
            self._client.close()
            self._client = None
