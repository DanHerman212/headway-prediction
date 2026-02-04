import argparse 
import logging
from datetime import datetime, date
import pyarrow as pa
import apache_beam as beam
from apache_beam.options.pipeline_options import (PipelineOptions,               
                                                 GoogleCloudOptions, 
                                                 SetupOptions)

from pipelines.beam.shared.transforms import (EnrichRecordFn, 
                                               CalculateServiceHeadwayFn, 
                                               CalculateTrackGapFn, 
                                               EnrichWithEmpiricalFn,
                                               CalculateUpstreamTravelTimeFn,
                                               CalculateUpstreamHeadwayFn,
                                               CalculateTravelTimeDeviationFn,
                                               ReindexTimeInGroupsFn)

# --- define schema for parque export ---
output_schema = pa.schema([
    # metadata
    ('trip_uid',pa.string()),
    # ('trip_date',pa.string()), # REMOVED
    ('arrival_time',pa.string()), 
    # ('timestamp',pa.float64()), # REMOVED

    # keys
    ('group_id',pa.string()),
    ('route_id',pa.string()),
    # ('direction',pa.string()), # REMOVED
    # ('stop_id',pa.string()),   # REMOVED

    # features
    ('time_idx',pa.int64()),
    ('day_of_week',pa.int64()),
    ('hour_sin',pa.float64()),
    ('hour_cos',pa.float64()),
    ('regime_id',pa.string()),
    ('track_id',pa.string()),

    # calculated features
    ('service_headway',pa.float64()), # target
    ('preceding_train_gap',pa.float64()),
    ('preceding_route_id', pa.string()),
    ('empirical_median',pa.float64()),
    ('upstream_headway_14th', pa.float64()),

    # upstream lags (nullable)
    ('travel_time_14th',pa.float64()),
    ('travel_time_14th_deviation', pa.float64()),
    ('travel_time_23rd',pa.float64()),
    ('travel_time_23rd_deviation', pa.float64()),
    ('travel_time_34th',pa.float64()),
    ('travel_time_34th_deviation', pa.float64()),
])

# --- helper functions ---
def extract_travel_time_key(element, feature_name, station_id):
    """
    Extract key-value pair for calculating median travel time.
    Key: (route, day_type, hour, origin_station_id)
    Value: travel_time
    """
    if element.get(feature_name) is None:
        return []
        
    route = element.get('route_id', 'OTHER')
    day_of_week = element.get('day_of_week')
    day_type = 'Weekend' if day_of_week and day_of_week > 5 else 'Weekday'
    ts = element.get('timestamp')
    hour = datetime.fromtimestamp(ts).hour if ts else 0
    
    return [((route, day_type, hour, station_id), element[feature_name])]

def extract_schedule_key(element):
    """
    Used to key the data for aggregation: ((route, daytype, hour), headway)
    only keeps row that have a valid headway
    """
    if element.get('service_headway') is None:
        return [] # filter out first row of session 
    
    route = element.get('route_id', 'OTHER')
    day_of_week = element.get('day_of_week')
    day_type = 'Weekend' if day_of_week and day_of_week >= 5 else 'Weekday'
    ts = element.get('timestamp')
    hour = datetime.fromtimestamp(ts).hour if ts else 0

    return [((route, day_type, hour), element['service_headway'])]

# -- median combiner logic --
import statistics

def compute_median(item):
    (key, headways) = item
    return (key, statistics.median(headways))

def sort_events(item):
    key, records = item
    sorted_records = sorted(records, key=lambda x: x['timestamp'])
    return [(key, r) for r in sorted_records]

def sanitize_record(record):
    # 1. Remove unnecessary columns that were used for logic but not needed in output
    keys_to_remove = ['trip_date', 'timestamp', 'direction', 'stop_id']
    for k in keys_to_remove:
        record.pop(k, None)

    # 2. Ensure no datetime objects reach Parquet writer to prevent ArrowTypeError
    for k, v in record.items():
        if isinstance(v, (datetime, date)):
            record[k] = v.isoformat()
    return record


def run(argv=None):
    # 1. Parse arguments for GCP configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True, help='GCP project ID')
    parser.add_argument('--temp_location', required=True, help='GCP location for temp files')
    known_args, pipeline_args = parser.parse_known_args(argv)

    # 2. configure beam pipelines
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = known_args.project_id
    google_cloud_options.temp_location = known_args.temp_location

    # 3. define the query
    query = """
    SELECT
        trip_uid, stop_id, track, stop_name, trip_date, route_id, direction, arrival_time
    FROM `headway_prediction.ml_baseline`
    WHERE track IN ('A1','A3')
        AND direction = 'S'
    """


    # 4. Build the pipeline
    with beam.Pipeline(options=pipeline_options) as p:

        # --- STEP 0: Read from BigQuery ---
        raw_rows = (
            p
            | 'ReadFromBQ' >> beam.io.ReadFromBigQuery(
                query=query,
                use_standard_sql=True,
                project=known_args.project_id,
                gcs_location=known_args.temp_location
            )
        )
        
        # --- STEP 1: Row Operations ---
        processed_rows = (
            raw_rows
            | 'EnrichRecords' >> beam.ParDo(EnrichRecordFn())
        )

        # --- STEP 2: Upstream Headway (Needs GroupByKey) ---
        # 1A: Calculate Headway AT Upstream Stations (14th St)
        with_upstream_headway = (
            processed_rows
            | 'KeyByGroupUpstream' >> beam.Map(lambda x: (x.get('group_id', 'Unknown'), x))
            | 'GroupUpstream' >> beam.GroupByKey()
            | 'SortUpstream' >> beam.FlatMap(sort_events)
            | 'CalcUpstreamHeadway' >> beam.ParDo(CalculateUpstreamHeadwayFn())
            | 'UnKeyUpstream' >> beam.Map(lambda x: x)
        )

        # --- STEP 3: Upstream Travel Time (Needs Trip Grouping) ---
        # 1B: travel time from upstream stations
        target_rows = (
            with_upstream_headway
            | 'KeyByTrip' >> beam.Map(lambda x: (x['trip_uid'], x))
            | 'GroupForSort' >> beam.GroupByKey()
            | 'SortByTime' >> beam.FlatMap(sort_events)
            | 'CalcTravelTime' >> beam.ParDo(CalculateUpstreamTravelTimeFn())
            | 'UnKeyTrip' >> beam.Map(lambda x: x)
        )

        # --- STEP 4: Service Headway (Target) ---
        # 2: stateful feature: service headway target
        with_target = (
            target_rows
            | 'KeyByGroup' >> beam.Map(lambda x: (x['group_id'], x))
            | 'GroupHeadway' >> beam.GroupByKey()
            | 'SortHeadway' >> beam.FlatMap(sort_events)
            | 'CalcServiceHeadway' >> beam.ParDo(CalculateServiceHeadwayFn())
            | 'UnKeyService' >> beam.Map(lambda x: x) 
            | 'FilterInvalidHeadways' >> beam.Filter(lambda x: x.get('service_headway') is None or (x.get('service_headway') > 0.5 and x.get('service_headway') < 120))
        )

        # --- STEP 5: Track Gap (Needs Track Grouping) ---
        # 3: calculate train gap on track
        with_track_gap = (
            with_target
            | 'KeyByTrack' >> beam.Map(lambda x: (x['track_id'], x)) # key by track
            | 'GroupTrack' >> beam.GroupByKey()
            | 'SortTrack' >> beam.FlatMap(sort_events)
            | 'CalcTrackGap' >> beam.ParDo(CalculateTrackGapFn())
            | 'UnkeyTrack' >> beam.Map(lambda x: x)
        )
        
        # --- STEP 6: Side Input Generation (Maps) ---
        TRAINING_CUTOFF = 1763424000.0
        
        # A: Empirical Schedule Map
        schedule_map = (
            with_track_gap
            | 'FilterTraining' >> beam.Filter(lambda x: x.get('timestamp', 0) < TRAINING_CUTOFF)
            | 'ExtractKeys'    >> beam.FlatMap(extract_schedule_key)
            | 'GroupKeys'      >> beam.GroupByKey()
            | 'CalcMedian'     >> beam.Map(compute_median)
        )

        # B: Empirical Travel Time Map
        training_data_for_tt = (
            with_track_gap 
            | 'FilterTrainTT' >> beam.Filter(lambda x: x.get('timestamp', 0) < TRAINING_CUTOFF)
        )
        
        tt_map_34 = (training_data_for_tt | 'Keys34' >> beam.FlatMap(lambda x: extract_travel_time_key(x, 'travel_time_34th', 'A28S')))
        tt_map_23 = (training_data_for_tt | 'Keys23' >> beam.FlatMap(lambda x: extract_travel_time_key(x, 'travel_time_23rd', 'A30S')))
        tt_map_14 = (training_data_for_tt | 'Keys14' >> beam.FlatMap(lambda x: extract_travel_time_key(x, 'travel_time_14th', 'A31S')))
        
        median_tt_map = (
            (tt_map_34, tt_map_23, tt_map_14)
            | 'FlattenTT' >> beam.Flatten()
            | 'GroupTT' >> beam.GroupByKey()
            | 'CalcMedianTT' >> beam.Map(compute_median)
        )

        # --- STEP 7: Apply Deviations ---
        with_deviations = (
            with_track_gap
            | 'CalcDeviations' >> beam.ParDo(
                CalculateTravelTimeDeviationFn(),
                median_tt_map=beam.pvalue.AsDict(median_tt_map)
            )
        )

        # --- STEP 8: Apply Empirical Headway ---
        with_empirical = (
            with_deviations
            | 'EnrichEmpirical' >> beam.ParDo(
                EnrichWithEmpiricalFn(),
                empirical_map=beam.pvalue.AsDict(schedule_map)
            )
        )

        # --- STEP 9: Finalize Dataset for Training ---
        final_training_data = (
            with_empirical
            # 1. Strict Drop of Missing Targets (so we don't have gaps)
            | 'FilterMissingTargets' >> beam.Filter(lambda x: x.get('service_headway') is not None)
            
            # 2. Re-Group to generate sequential index
            | 'KeyForIndexing' >> beam.Map(lambda x: (x.get('group_id'), x))
            | 'GroupForIndexing' >> beam.GroupByKey()
            
            # 3. Apply Re-indexing
            | 'ReindexTime' >> beam.ParDo(ReindexTimeInGroupsFn())
        )

        write_to_parque = (
            final_training_data
            | 'SanitizeTypes' >> beam.Map(sanitize_record)
            | 'WriteToParquet' >> beam.io.WriteToParquet(
                file_path_prefix=f"{known_args.temp_location.rstrip('/')}/training_data",
                schema=output_schema,
                file_name_suffix='.parquet',
                num_shards=1 # keep as 1 files for 75k rows since its small
            )
        )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
    