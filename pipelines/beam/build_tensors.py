"""
Apache Beam Pipeline: Build ML Tensors
=======================================
Reads transformed headway data from BigQuery and creates
the final ML dataset tensors for Graph WaveNet training.

Output:
- X.npy: Input tensor (Samples, T_in, Nodes, Features)
- Y.npy: Target tensor (Samples, T_out, Nodes)
- adjacency_matrix.npy: Graph structure
- node_mapping.json: node_id -> index mapping

Usage:
    python build_tensors.py \
        --input_table=headway_dataset.ml_final \
        --output_path=gs://bucket/ml-dataset/ \
        --runner=DataflowRunner \
        --project=your-project \
        --region=us-central1 \
        --temp_location=gs://bucket/temp/
"""

import argparse
import json
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Configuration
TIME_BIN_MINUTES = 5
T_IN = 12   # Input sequence length (60 minutes)
T_OUT = 12  # Output sequence length (60 minutes)


class BuildTimeBins(beam.DoFn):
    """Aggregate arrivals into 5-minute time bins."""
    
    def process(self, element):
        # TODO: Implement time binning logic
        # Input: row from BigQuery with (node_id, arrival_time, headway_minutes, ...)
        # Output: (time_bin, node_id, headway_value, features...)
        yield element


class ImputeMissingHeadways(beam.DoFn):
    """Forward-fill missing headways with running headway."""
    
    def process(self, element):
        # TODO: Implement forward-fill imputation
        yield element


class AddTimeFeatures(beam.DoFn):
    """Add sin/cos time embeddings."""
    
    def process(self, element):
        import math
        
        # TODO: Extract timestamp from element
        # minute_of_day = ...
        # day_of_week = ...
        
        # time_sin = math.sin(2 * math.pi * minute_of_day / 1440)
        # time_cos = math.cos(2 * math.pi * minute_of_day / 1440)
        # dow_sin = math.sin(2 * math.pi * day_of_week / 7)
        # dow_cos = math.cos(2 * math.pi * day_of_week / 7)
        
        yield element


class CreateSlidingWindows(beam.DoFn):
    """Create (X, Y) pairs with sliding window."""
    
    def __init__(self, t_in, t_out):
        self.t_in = t_in
        self.t_out = t_out
    
    def process(self, element):
        # TODO: Implement sliding window creation
        # Input: sorted time series for all nodes
        # Output: (X, Y) pairs
        yield element


def run_pipeline(args):
    """Main pipeline execution."""
    
    pipeline_options = PipelineOptions(args)
    
    with beam.Pipeline(options=pipeline_options) as p:
        # Read from BigQuery
        headways = (
            p 
            | 'ReadFromBQ' >> beam.io.ReadFromBigQuery(
                query=f'''
                    SELECT 
                        node_id,
                        arrival_time,
                        headway_minutes,
                        lateness_minutes,
                        service_date,
                        hour_of_day,
                        day_of_week
                    FROM `{args.input_table}`
                    ORDER BY arrival_time
                ''',
                use_standard_sql=True
            )
        )
        
        # Build time bins
        binned = (
            headways
            | 'BuildTimeBins' >> beam.ParDo(BuildTimeBins())
        )
        
        # Impute missing values
        imputed = (
            binned
            | 'ImputeMissing' >> beam.ParDo(ImputeMissingHeadways())
        )
        
        # Add time features
        with_features = (
            imputed
            | 'AddTimeFeatures' >> beam.ParDo(AddTimeFeatures())
        )
        
        # Create sliding windows
        windows = (
            with_features
            | 'CreateWindows' >> beam.ParDo(CreateSlidingWindows(T_IN, T_OUT))
        )
        
        # TODO: Convert to numpy arrays and write to GCS
        # This is a placeholder - actual implementation needs numpy serialization
        
        _ = (
            windows
            | 'WriteOutput' >> beam.io.WriteToText(
                args.output_path + 'samples',
                file_name_suffix='.json'
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_table', required=True,
                        help='BigQuery table: project.dataset.table')
    parser.add_argument('--output_path', required=True,
                        help='GCS path for output: gs://bucket/path/')
    
    args, pipeline_args = parser.parse_known_args()
    
    logging.getLogger().setLevel(logging.INFO)
    run_pipeline(args)


if __name__ == '__main__':
    main()
