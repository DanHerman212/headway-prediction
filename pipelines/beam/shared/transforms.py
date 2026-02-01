import apache_beam as beam
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec
from apache_beam.coders import FloatCoder
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
        # 1 filter: only calculate target for the specific station
        # if upstream events pass through here, we ignore them for target generation
        if element.get('stop_id') != self.TARGET_STATION:
            yield element
            return
    
        # 2 get timestamp (feature already prepared)
        current_ts = element.get('timestamp')
        if current_ts is None:
            yield element
            return
        
        # 3 read and calculate
        previous_ts = last_arrival.read()

        if previous_ts is not None:
            # difference in minutes
            headway_min = (current_ts - previous_ts) / 60.0
            element['service_headway'] = headway_min
        
        # 4 update state
        last_arrival.write(current_ts)

        yield element

class CalculateTrackGapFn(beam.DoFn):
    """
    Stateful DoFn calculate 'preceding_train_gap' (cross-line interaction).
    Partition key: track_id (e.g. A1, A3 (local, express))
    state: last arrival time on this track (timestamp float)
    """

    LAST_ARR_TRACK = ReadModifyWriteStateSpec('last_arr_track', FloatCoder())

    # target station ID (still fildered because we only predict at W4th for now)

    def process(self, element, last_arrival=beam.DoFn.StateParam(LAST_ARR_TRACK)):
        # 1. Filter: Only output features for target station rows
        if element.get('stop_id') != self.TARGET_STATION:
            yield element
            return
        
        current_ts = element.get('timestamp')
        if current_ts is None:
            yield element
            return
        
        # 2 read state
        previous_ts = last_arrival.read()

        # 3 calculate gap
        if previous_ts is not None:
            gap_min = (current_ts - previous_ts) / 60.0
            element['preceding_train_gap'] = gap_min

        # 4 update state
        last_arrival.write(current_ts)

        yield element


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