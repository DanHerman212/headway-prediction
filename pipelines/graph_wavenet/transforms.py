"""
Stateful Beam transforms for Graph WaveNet dataset labeling.

LabelTimeToNextTrainFn -- the core DoFn:
  Keyed by node_id.  For each 1-minute snapshot row it:

  Feature engineering (per-row):
    elapsed_headway   -- Integer minutes since last departure.  Ticks up
                         1, 2, 3, ... while the node is empty; resets to
                         0 whenever train_present == 1.  (Replaces the
                         former minutes_since_last_train float.)
    is_A, is_C, is_E  -- One-hot encoding of route_id.
    time_sin, time_cos -- Sine/cosine encoding of time-of-day
                         (minutes past midnight / 1440).
    dow_sin, dow_cos   -- Sine/cosine encoding of day-of-week (0-6 / 7).

  Stateful labeling:
    Buffers rows until an arrival event (train_present 0->1) then
    retroactively labels every buffered row with
    time_to_next_train (forward target Y) = arrival_time - snapshot_time.

    A 60-minute processing-time timer flushes zombie buffers with
    time_to_next_train = None to prevent unbounded memory.

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
import math
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
_TWO_PI = 2.0 * math.pi
_MINUTES_PER_DAY = 1440.0
_DAYS_PER_WEEK = 7.0


def _minutes_between(earlier_str, later_str):
    """Return (later - earlier) in whole integer minutes."""
    t1 = datetime.strptime(earlier_str, _TIME_FMT)
    t2 = datetime.strptime(later_str, _TIME_FMT)
    return int(round((t2 - t1).total_seconds() / 60.0))


def _add_temporal_features(row):
    """Add cyclical time-of-day and day-of-week features in-place."""
    dt = datetime.strptime(row["snapshot_time"], _TIME_FMT)
    minutes_past_midnight = dt.hour * 60 + dt.minute
    row["time_sin"] = round(math.sin(_TWO_PI * minutes_past_midnight / _MINUTES_PER_DAY), 6)
    row["time_cos"] = round(math.cos(_TWO_PI * minutes_past_midnight / _MINUTES_PER_DAY), 6)
    dow = dt.weekday()  # Monday=0 ... Sunday=6
    row["dow_sin"] = round(math.sin(_TWO_PI * dow / _DAYS_PER_WEEK), 6)
    row["dow_cos"] = round(math.cos(_TWO_PI * dow / _DAYS_PER_WEEK), 6)


def _add_route_onehot(row):
    """Replace string route_id with binary is_A / is_C / is_E columns."""
    route = row.pop("route_id", None)
    row["is_A"] = 1 if route == "A" else 0
    row["is_C"] = 1 if route == "C" else 0
    row["is_E"] = 1 if route == "E" else 0


class LabelTimeToNextTrainFn(beam.DoFn):
    """Stateful DoFn keyed by node_id.

    Input:  (node_id, row_dict)
    Output: row_dict with ML-ready features + retroactive label:
              elapsed_headway       (int, X feature)
              is_A / is_C / is_E    (int, X features)
              time_sin / time_cos   (float, X features)
              dow_sin / dow_cos     (float, X features)
              time_to_next_train    (int, Y target -- filled on flush)

    State (per key):
      prev_train_present  -- int: previous train_present (None on cold start)
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

        # -- Stateless features (every row) --------------------------------
        _add_temporal_features(row)    # time_sin, time_cos, dow_sin, dow_cos
        _add_route_onehot(row)         # is_A, is_C, is_E  (pops route_id)
        row.pop("trip_id", None)       # not an ML feature

        # -- Elapsed headway (stateful) ------------------------------------
        dep_time = last_departure.read()
        if current == 1:
            # A train is at the node right now -- headway is 0
            row["elapsed_headway"] = 0
        elif dep_time is not None:
            row["elapsed_headway"] = _minutes_between(dep_time, snapshot_time)
        else:
            # Cold start -- no departure observed yet
            row["elapsed_headway"] = None

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
