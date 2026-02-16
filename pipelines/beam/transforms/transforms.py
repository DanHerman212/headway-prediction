import apache_beam as beam
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.coders import FloatCoder, PickleCoder, StrUtf8Coder
from datetime import datetime
import math


class EnrichRecordFn(beam.DoFn):
    """
    Consolidated row-level feature engineering.
    1. time_idx: minutes since epoch
    2. group_id: route_id + direction (e.g. "A_South")
    3. route_id: clean route_id (map rare routes to OTHER)
    """
    
    ALLOWED_ROUTES = {'A', 'C', 'E'}
    ALLOWED_TRACKS = {'A1', 'A3'}

    def process(self, element):
        # Element is a dict like {'arrival_time': '2025-10-13 15:27:06...', 'route_id': 'A', ...}
        
        # --- PREPARATION ---
        raw_time = element.get('arrival_time')
        dt = None
        
        if isinstance(raw_time, str):
            try:
                 clean_time = raw_time.replace(' UTC', '')
                 dt = datetime.fromisoformat(clean_time)
            except ValueError:
                dt = datetime.strptime(clean_time, '%Y-%m-%d %H:%M:%S')
        elif isinstance(raw_time, (int, float)):
            dt = datetime.fromtimestamp(raw_time)
        else:
            dt = raw_time

        # --- FEATURE 1: time_idx ---
        time_idx = None
        if dt:
            timestamp = dt.timestamp()
            time_idx = int(timestamp / 60)

        # --- FEATURE 3: route_id cleaning ---
        raw_route = element.get('route_id')
        clean_route = 'OTHER'
        if raw_route in self.ALLOWED_ROUTES:
            clean_route = raw_route
        
        # --- FEATURE 2: group_id ---
        direction = element.get('direction')
        group_id = None
        if direction:
            dir_str = "South" if direction == 'S' else direction
            group_id = f"{clean_route}_{dir_str}"

        # --- FEATURE 4: track_id ---
        raw_track = element.get('track')
        clean_track = 'OTHER'
        if raw_track in self.ALLOWED_TRACKS:
            clean_track = raw_track

        # --- FEATURE 5: regime_id ---
        # A train runs local 22:00 - 05:00 (Night), Express otherwise (Day)
        regime = 'Day'
        if dt:
            hour = dt.hour
            # night is 10pm (22) to 4:59AM (4)
            if hour >= 22 or hour < 5:
                regime = 'Night'
        
        # --- FEATURE 6: Cyclical Time ---
        rads = 0.0
        if dt:
            # using exact hour + fractional minute for smoother signal
            distinct_hour = dt.hour + (dt.minute / 60.0)

            # 2 * pi * t / 24
            rads = (distinct_hour / 24.0) * 2 * math.pi
        
        # --- FEATURE 7: day_of_week ---
        day_of_week = None
        if dt:
            day_of_week = dt.weekday()

        # --- CONSTRUCT OUTPUT ---
        new_element = element.copy()
        
        if time_idx is not None:
            new_element['time_idx'] = time_idx
        
        new_element['route_id'] = clean_route 
        
        if group_id:
            new_element['group_id'] = group_id

        new_element['track_id'] = clean_track

        new_element['regime_id'] = regime

        if dt:
            new_element['hour_sin'] = math.sin(rads)
            new_element['hour_cos'] = math.cos(rads)
            new_element['day_of_week'] = day_of_week
            new_element['timestamp'] = timestamp
            
            # ensure arrival_time is string for parquet export
            # usually bigquery source gives a string, but if we parsed it to object
            # we must convert it back or ensure the original string is preserved.
            if isinstance(dt, datetime):
                new_element['arrival_time'] = dt.isoformat()
            elif dt is not None and not isinstance(element.get('arrival_time'), str):
                new_element['arrival_time'] = str(dt)

        # Check other string fields for objects (like datetime.date for trip_date)
        for field in ['trip_date', 'trip_uid', 'route_id', 'direction', 'stop_id', 'track']:
             val = new_element.get(field)
             if val is not None and not isinstance(val, str):
                 new_element[field] = str(val)

        yield new_element

class CalculateServiceHeadwayFn(beam.DoFn):
    """
    Stateful DoFn to calculate 'service_headway' (Target).
    Partition Key: group_id (e.g. "A_South")
    State: Last Arrival Time (timestamp float)
    """
    LAST_ARRIVAL = ReadModifyWriteStateSpec('last_arrival', FloatCoder())

    # target station ID
    TARGET_STATION ="A32S"

    def process(self, element, last_arrival=beam.DoFn.StateParam(LAST_ARRIVAL)):
        key, record = element

        # 1 filter: only calculate target for the specific station
        # if upstream events pass through here, we ignore them for target generation
        if record.get('stop_id') != self.TARGET_STATION:
            yield record
            return
    
        # 2 get timestamp (feature already prepared)
        current_ts = record.get('timestamp')
        if current_ts is None:
            yield record
            return
        
        # 3 read and calculate
        previous_ts = last_arrival.read()

        if previous_ts is not None:
            # Session Timeout: if gap > 90 mins, treat as new session (headway=None)
            if (current_ts - previous_ts) > (90 * 60):
                previous_ts = None
            else:
                # difference in minutes
                headway_min = (current_ts - previous_ts) / 60.0
                record['service_headway'] = headway_min
        
        # 4 update state
        last_arrival.write(current_ts)

        yield record

class CalculateTrackGapFn(beam.DoFn):
    """
     Stateful DoFn calculate 'preceding_train_gap' (cross-line interaction).
    Partition key: track_id (e.g. A1, A3 (local, express))
    state: 
      - last arrival time on this track (timestamp float)
      - last route id on this track (string)
    """

    LAST_ARR_TRACK = ReadModifyWriteStateSpec('last_arr_track', FloatCoder())
    LAST_ROUTE_ID = ReadModifyWriteStateSpec('last_route_id', StrUtf8Coder())

    # target station ID (still fildered because we only predict at W4th for now)
    TARGET_STATION = "A32S"

    def process(self, element, 
                last_arrival=beam.DoFn.StateParam(LAST_ARR_TRACK),
                last_route=beam.DoFn.StateParam(LAST_ROUTE_ID)):
        key, record = element

        # 1. Filter: Only output features for target station rows
        if record.get('stop_id') != self.TARGET_STATION:
            yield record
            return
        
        current_ts = record.get('timestamp')
        if current_ts is None:
            yield record
            return
        
        # 2 read state
        previous_ts = last_arrival.read()
        previous_route = last_route.read()

        # 3 calculate gap
        if previous_ts is not None:
            gap_min = (current_ts - previous_ts) / 60.0
            record['preceding_train_gap'] = gap_min
        
        if previous_route is not None:
            record['preceding_route_id'] = previous_route

        # 4 update state
        last_arrival.write(current_ts)

        current_route = record.get('route_id')
        if current_route:
            last_route.write(current_route)

        yield record


class EnrichWithEmpiricalFn(beam.DoFn):
    """
    Enrich using broadcasted empirical schedule map
    Key: (route_id, day_type, hour)
    Val: median_headway
    """
    def process(self, element, empirical_map):
        # 1 extract keys
        route = element.get('route_id', 'OTHER')

        # day type: weekday (0-4) vs Weekend (5-6)
        # Assuming 'day_of_week' is 0=Mon, 6=Sun
        day_of_week = element.get('day_of_week')
        day_type = 'Weekend' if day_of_week and day_of_week >= 5 else 'Weekday'

        # Hour
        ts = element.get('timestamp')
        hour = 0
        if ts: 
            hour = datetime.fromtimestamp(ts).hour
        
        # 2 lookup
        key = (route, day_type, hour)
        median = empirical_map.get(key)

        # 3 handle missing (default or global mean)
        # using a fallback to ensure we don't break the model with nulls
        # 8 mins is reasonable
        if median is None:
            median = 8.0
        
        element['empirical_median'] = float(median)
        yield element

class CalculateUpstreamTravelTimeFn(beam.DoFn):
    """
    Stateful DoFn to calculate travel times from upstream stations.
    Robust to out-of-order arrival of events using Buffering ("Park and Wait").
    partition key: trip_uid
    state: 
      - trip_arrivals: dict of {station_id: timestamp}
      - pending_target: dict (the target row waiting for upstream)
    """
    TRIP_ARRIVALS = ReadModifyWriteStateSpec('trip_arrivals', PickleCoder())
    PENDING_TARGET = ReadModifyWriteStateSpec('pending_target', PickleCoder())

    # Correct Station IDs for 8th Ave Line (A/C/E)
    STATION_34TH = 'A28S' # 34 St - Penn Station
    STATION_23RD = 'A30S' # 23 St
    STATION_14TH = 'A31S' # 14 St
    TARGET_STATION = 'A32S' # W 4 St Washington Sq

    RELEVANT_STATIONS = {STATION_34TH, STATION_23RD, STATION_14TH, TARGET_STATION}

    def _has_sufficient_upstream(self, arrivals):
        # Relaxed rule: Emit if we have ANY upstream station
        return (self.STATION_34TH in arrivals or 
                self.STATION_23RD in arrivals or 
                self.STATION_14TH in arrivals)

    def _get_upstream_metrics(self, arrivals, station_id):
        data = arrivals.get(station_id)
        if data is None:
            return None, None
        
        # backward compatibility if state was just a float timestamp
        if isinstance(data, float):
            return data, None 
            
        return data.get('ts'), data.get('hw')

    def _enrich_and_yield(self, record, arrivals):
        current_ts = record.get('timestamp')
        
        ts_34th, hw_34th = self._get_upstream_metrics(arrivals, self.STATION_34TH)
        record['travel_time_34th'] = (current_ts - ts_34th) / 60.0 if ts_34th else None
        
        ts_23rd, hw_23rd = self._get_upstream_metrics(arrivals, self.STATION_23RD)
        record['travel_time_23rd'] = (current_ts - ts_23rd) / 60.0 if ts_23rd else None

        ts_14th, hw_14th = self._get_upstream_metrics(arrivals, self.STATION_14TH)
        record['travel_time_14th'] = (current_ts - ts_14th) / 60.0 if ts_14th else None

        if hw_14th is not None:
            record['upstream_headway_14th'] = hw_14th

        yield record

    def process(self, element, 
                arrivals_state=beam.DoFn.StateParam(TRIP_ARRIVALS),
                pending_state=beam.DoFn.StateParam(PENDING_TARGET)):
        
        trip_id, record = element
        stop_id = record.get('stop_id')

        # optimize: ignore stations we don't care about at all
        if stop_id not in self.RELEVANT_STATIONS:
            return

        current_ts = record.get('timestamp')
        if current_ts is None:
            return
        
        # Load current state
        arrivals = arrivals_state.read() or {}
        pending = pending_state.read()

        # Update knowledge base
        current_payload = {'ts': current_ts}
        if 'upstream_headway_14th' in record:
             current_payload['hw'] = record['upstream_headway_14th']

        arrivals[stop_id] = current_payload
        arrivals_state.write(arrivals)

        # LOGIC BRANCH 1: We just received the Target (W4th)
        if stop_id == self.TARGET_STATION:
            if self._has_sufficient_upstream(arrivals):
                yield from self._enrich_and_yield(record, arrivals)
            else:
                pending_state.write(record)

        # LOGIC BRANCH 2: We just received Upstream (34th/23rd/14th)
        else:
            if pending is not None:
                if self._has_sufficient_upstream(arrivals):
                    yield from self._enrich_and_yield(pending, arrivals)
                    pending_state.clear()

class CalculateUpstreamHeadwayFn(beam.DoFn):
    """
    Stateful DoFn to calculate headways at Upstream stations (14th)
    we want to know: Was this strain already bunched at 14th?
    Partition Key: group_id
    State: Last arrival time at 14th
    """
    LAST_ARRIVAL_14TH = ReadModifyWriteStateSpec('last_arrival_14th', FloatCoder())

    # 14 st - 1 Av (A/C/E)
    UPSTREAM_STATION_14TH = 'A31S'

    def process(self, element, last_arrival=beam.DoFn.StateParam(LAST_ARRIVAL_14TH)):
        key, record = element

        # we only calculate this metric when we see event at 14th st
        # we attach it to the record so it travels with the trip_id grouping downstrea.
        if record.get('stop_id')==self.UPSTREAM_STATION_14TH:
            current_ts = record.get('timestamp')

            if current_ts is not None:
                previous_ts = last_arrival.read()

                if previous_ts is not None:
                # timeout logic: if gap > 90 mins, reset
                    if (current_ts - previous_ts) < (90 * 60):
                        headway_min = (current_ts - previous_ts) / 60.0
                        record['upstream_headway_14th'] = headway_min
                
                last_arrival.write(current_ts)
        
        yield record


class CalculateTravelTimeDeviationFn(beam.DoFn):
    """
    Calculates how much the upstream travel time deviates from the historical median.
    
    Input: Record with 'travel_time_34th', 'travel_time_14th' etc.
    Side Input (median_tt_map): 
        Key: (route_id, day_type, hour, origin_station_id)
        Val: median_travel_time_minutes (float)
    """
    
    # Map friendly names to station IDs for lookup
    SEGMENTS = [
        ('travel_time_34th', 'A28S'), # 34 St
        ('travel_time_23rd', 'A30S'), # 23 St
        ('travel_time_14th', 'A31S')  # 14 St
    ]

    def process(self, element, median_tt_map):
        # 1. Setup Keys (Copy logic from EnrichWithEmpiricalFn)
        route = element.get('route_id', 'OTHER')
        
        # Day Type
        day_of_week = element.get('day_of_week')
        day_type = 'Weekend' if day_of_week and day_of_week >= 5 else 'Weekday'

        # Hour
        ts = element.get('timestamp')
        hour = 0
        if ts: 
            hour = datetime.fromtimestamp(ts).hour
        
        # 2. Iterate over known upstream segments
        for feature_name, origin_station_id in self.SEGMENTS:
            actual_tt = element.get(feature_name)
            
            # If we don't have an actual travel time (e.g. came from local track, missed 34th), skip
            if actual_tt is None:
                continue

            # 3. Lookup Expected
            # Key must match how you built the map in generate_dataset.py
            key = (route, day_type, hour, origin_station_id)
            expected_tt = median_tt_map.get(key)

            if expected_tt is not None:
                # DEVIATION: Positive = Slower than usual, Negative = Faster than usual
                deviation = actual_tt - expected_tt
                
                # Suffix feature name (e.g. "travel_time_34th_deviation")
                element[f'{feature_name}_deviation'] = deviation
            else:
                # If no history exists (rare), assume 0.0 (Normal) to keep data continuous
                element[f'{feature_name}_deviation'] = 0.0

        yield element


class ReindexTimeInGroupsFn(beam.DoFn):
    """
    Re-generates 'time_idx' to be strictly sequential (0, 1, 2...) 
    after filtering has created gaps.
    Input: (key, List[records]) where records are sorted by time.
    """
    def process(self, element):
        key, records = element
        
        # Sort just in case
        sorted_records = sorted(records, key=lambda x: x['timestamp'])
        
        for i, record in enumerate(sorted_records):
            # Overwrite the global time_idx with the sequential series index
            record['time_idx'] = i 
            yield record