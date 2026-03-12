"""
Stateful Beam transforms for Graph WaveNet dataset labeling.

LabelTimeToNextTrainFn -- the core DoFn:
  Keyed by node_id.  For each 1-minute snapshot row it:

  1. Computes minutes_since_last_train (backward feature X)
     = snapshot_time - last_departure_time

  2. Buffers rows until an arrival event (train_present 0->1)
     then retroactively labels every buffered row with
     time_to_next_train (forward target Y) = arrival_time - snapshot_time

  3. A 60-minute processing-time timer flushes zombie buffers
     with time_to_next_train = None to prevent unbounded memory.

State transitions:
  prev=0, curr=1  ->  ARRIVAL   : flush old buffer with labels, start new buffer
  prev=1, curr=0  ->  DEPARTURE : update last_departure_time, add row to buffer
  prev=0, curr=0  ->  WAITING   : add row to buffer
  prev=1, curr=1  ->  STILL_HERE: add row to buffer
  prev=None        ->  FIRST     : add row to buffer (cold start)

The arrival row itself (0->1) is NOT emitted immediately.  It starts a
new buffer and will be labeled when the FOLLOWING train arrives.  This
matches the mobile-app UX: the user can already see the current train.
"""

import logging
import time as time_mod
from datetime import datetime

import apache_beam as beam
from apache_beam.coders import PickleCoder
from apache_beam.transforms.userstate import (
    BagStateSpec,
    ReadModifyWriteStateSpec,
    TimerSpec,
    on_timer,
)
from apache_beam.transforms.timeutil import TimeDomain

logger = logging.getLogger(__name__)

_TIME_FMT = "%Y-%m-%d %H:%M:%S"
_BUFFER_TIMEOUT_SECONDS = 60 * 60  # 60 minutes


def _minutes_between(earlier_str, later_str):
    """Return (later - earlier) in minutes, rounded to 1 decimal."""
    t1 = datetime.strptime(earlier_str, _TIME_FMT)
    t2 = datetime.strptime(later_str, _TIME_FMT)
    return round((t2 - t1).total_seconds() / 60.0, 1)


class LabelTimeToNextTrainFn(beam.DoFn):
    """Stateful DoFn keyed by node_id.

    Input:  (node_id, row_dict)
    Output: row_dict augmented with minutes_since_last_train and
            time_to_next_train (emitted only on flush).

    State (per key):
      prev_train_present  -- int: previous minutes train_present (None on cold start)
      last_departure_time -- str: snapshot_time of most recent 1->0 transition
      buffer              -- bag of row dicts awaiting the next arrival

    Timer:
      flush_timer -- processing-time, resets every element, fires after 60 min
                     of silence to prevent unbounded memory growth.
    """

    PREV_STATE = ReadModifyWriteStateSpec("prev_train_present", PickleCoder())
    LAST_DEPARTURE = ReadModifyWriteStateSpec("last_departure_time", PickleCoder())
    BUFFER = BagStateSpec("buffer", PickleCoder())
    FLUSH_TIMER = TimerSpec("flush_timer", TimeDomain.REAL_TIME)

    def process(
        self,
        element,
        prev_state=beam.DoFn.StateParam(PREV_STATE),
        last_departure=beam.DoFn.StateParam(LAST_DEPARTURE),
        buffer=beam.DoFn.StateParam(BUFFER),
        flush_timer=beam.DoFn.TimerParam(FLUSH_TIMER),
    ):
        node_id, row = element
        current = row["train_present"]
        snapshot_time = row["snapshot_time"]

        prev = prev_state.read()  # None on cold start

        # -- Feature X: minutes since last departure -----------------------
        dep_time = last_departure.read()
        if dep_time is not None:
            row["minutes_since_last_train"] = _minutes_between(dep_time, snapshot_time)
        else:
            row["minutes_since_last_train"] = None

        # -- State machine --------------------------------------------------
        if prev == 0 and current == 1:
            # ARRIVAL -- flush old buffer with retroactive labels
            arrival_time = snapshot_time
            buffered = sorted(buffer.read(), key=lambda r: r["snapshot_time"])
            buffer.clear()

            for buf_row in buffered:
                buf_row["time_to_next_train"] = _minutes_between(
                    buf_row["snapshot_time"], arrival_time
                )
                yield buf_row

            # Current (arrival) row starts the NEW buffer
            row["time_to_next_train"] = None
            buffer.add(row)

        elif prev == 1 and current == 0:
            # DEPARTURE -- record departure time, then buffer
            last_departure.write(snapshot_time)
            row["time_to_next_train"] = None
            buffer.add(row)

        else:
            # WAITING (0->0), STILL_HERE (1->1), or FIRST (None->*)
            row["time_to_next_train"] = None
            buffer.add(row)

        # -- Housekeeping ---------------------------------------------------
        prev_state.write(current)
        flush_timer.set(int(time_mod.time()) + _BUFFER_TIMEOUT_SECONDS)

    @on_timer(FLUSH_TIMER)
    def flush_expired(self, buffer=beam.DoFn.StateParam(BUFFER)):
        """Flush zombie buffer with NULL targets after 60 min of silence."""
        buffered = sorted(buffer.read(), key=lambda r: r["snapshot_time"])
        buffer.clear()
        if buffered:
            logger.info(
                "Timer flush: %d rows for node %s (oldest: %s)",
                len(buffered),
                buffered[0].get("node_id", "?"),
                buffered[0].get("snapshot_time", "?"),
            )
        for row in buffered:
            row["time_to_next_train"] = None
            yield row
