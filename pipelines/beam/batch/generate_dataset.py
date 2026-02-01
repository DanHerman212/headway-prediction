import argparse 
import logging
from datetime import datetime
import pyarrow as pa
import apache_beam as beam
from apache_beam.options.pipeline_options import (PipelineOptions,               
                                                 GoogleCloudOptions, 
                                                 SetupOptions)

from pipelines.beam.shared.transforms import (EnrichRecordFn, 
                                               CalculateServiceHeadwayFn, 
                                               CalculateTrackGapFn, 
                                               EnrichWithEmpiricalFn,
                                               CalculateUpstreamTravelTimeFn)

# --- define schema for parque export ---
output_schema = pa.schema([
    # metadata
    ('trip_uid',pa.string()),
    ('trip_date',pa.string()), # or date32 if you parse it
    ('arrival_time',pa.string()), # Input was datetime object, needs to be handled
    ('timestamp',pa.float64()),

    # keys
    ('group_id',pa.string()),
    ('route_id',pa.string()),
    ('direction',pa.string()),
    ('stop_id',pa.string()),

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
    ('empirical_median',pa.float64()),

    # upstream lags (nullable)
    ('travel_time_14th',pa.float64()),
    ('travel_time_23rd',pa.float64()),
    ('travel_time_34th',pa.float64()),
])

# --- helper functions ---
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

        # read from bigquery
        # this returns a PCollection of dictionaries: {'trip_uid': '...', 'arrival_time': '...', ...}
        raw_rows = (
            p
            | 'ReadFromBQ' >> beam.io.ReadFromBigQuery(
                query=query,
                use_standard_sql=True,
                project=known_args.project_id,
                gcs_location=known_args.temp_location
            )
        )
        #1 row-level enrichments
        processed_rows = (
            raw_rows
            | 'EnrichRecords' >> beam.ParDo(EnrichRecordFn())
        )

        #1B travel time from upstream stations
        target_rows = (
            processed_rows
            | 'KeyByTrip' >> beam.Map(lambda x: (x['trip_uid'], x))
            | 'CalcTravelTime' >> beam.ParDo(CalculateUpstreamTravelTimeFn())
            | 'UnKeyTrip' >> beam.Map(lambda x: x)
        )

        #2 stateful feature: service headway target
        # key: group_id | value: row
        with_target = (
            target_rows
            | 'KeyByGroup' >> beam.Map(lambda x: (x['group_id'], x))
            | 'GroupHeadway' >> beam.GroupByKey()
            | 'SortHeadway' >> beam.FlatMap(sort_events)
            | 'CalcServiceHeadway' >> beam.ParDo(CalculateServiceHeadwayFn())
            | 'UnKeyService' >> beam.Map(lambda x: x) # strip the key back to just value
        )

        # 3 calculate train gap on track
        with_track_gap = (
            with_target
            | 'KeyByTrack' >> beam.Map(lambda x: (x['track_id'], x)) # key by track
            | 'GroupTrack' >> beam.GroupByKey()
            | 'SortTrack' >> beam.FlatMap(sort_events)
            | 'CalcTrackGap' >> beam.ParDo(CalculateTrackGapFn())
            | 'UnkeyTrack' >> beam.Map(lambda x: x)
        )
        
        # -- Branch A: Create Empirical Schedule (training data only)
        # cutoff Nov 18, 2025
        TRAINING_CUTOFF = 1763424000.0

        schedule_map = (
            with_track_gap
            | 'FilterTraining' >> beam.Filter(lambda x: x.get('timestamp', 0) < TRAINING_CUTOFF)
            # extract returns list of (key, value) tuples for FlatMap
            | 'ExtractKeys'    >> beam.FlatMap(extract_schedule_key)
            | 'GroupKeys'      >> beam.GroupByKey()
            | 'CalcMedian'     >> beam.Map(compute_median)
        )
        # apply to all data
        with_empirical = (
            with_track_gap
            | 'EnrichEmpirical' >> beam.ParDo(
                EnrichWithEmpiricalFn(),
                empirical_map=beam.pvalue.AsDict(schedule_map)
            )
        )
        write_to_parque = (
            with_empirical
            | 'WriteToParquet' >> beam.io.WriteToParquet(
                file_path_prefix=f"{known_args.temp_location}/training_data",
                schema=output_schema,
                file_name_suffix='.parquet',
                num_shards=1 # keep as 1 files for 75k rows since its small
            )
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
    