"""
Rolling window buffer for streaming headway prediction.

Accumulates enriched observations per group_id in Beam state.
Once the buffer reaches the encoder length (20 observations),
emits a complete window on every new arrival — matching the
input format expected by the Vertex AI serving endpoint:

    {"group_id": "A_South", "observations": [...20 dicts...]}
"""

import logging

import apache_beam as beam
from apache_beam.coders import PickleCoder
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec

logger = logging.getLogger(__name__)

ENCODER_LENGTH = 20


class BufferWindowFn(beam.DoFn):
    """Stateful DoFn that maintains a rolling window per group_id.

    Input:  (group_id, enriched_record)
    Output: {"group_id": str, "observations": list[dict]}

    On each new record:
      1. Append to buffer
      2. Trim to ENCODER_LENGTH (keep most recent)
      3. If buffer is full, emit the window

    Records arrive in chronological order within a group_id because
    feed snapshots are processed sequentially through Steps 1-3.
    """

    BUFFER = ReadModifyWriteStateSpec("buffer", PickleCoder())

    def process(self, element, buffer_state=beam.DoFn.StateParam(BUFFER)):
        group_id, record = element

        # Load existing buffer
        buffer = buffer_state.read() or []

        # Append new record
        buffer.append(record)

        # Trim to window size (keep most recent)
        if len(buffer) > ENCODER_LENGTH:
            buffer = buffer[-ENCODER_LENGTH:]

        # Persist updated buffer
        buffer_state.write(buffer)

        buf_len = len(buffer)
        if buf_len < ENCODER_LENGTH:
            logger.info(
                "Buffer warmup %s: %d/%d observations",
                group_id, buf_len, ENCODER_LENGTH,
            )
        else:
            logger.info(
                "Buffer full %s: emitting window (%d obs)",
                group_id, buf_len,
            )
            yield {
                "group_id": group_id,
                "observations": list(buffer),
            }
