/*
  Create Node Mapping for Graph WaveNet
  
  This script creates a mapping from node_id (station-line-direction) 
  to integer indices for tensor construction.
  
  Source: {{ params.project_id }}.mta_transformed.headways_all_nodes
  Target: {{ params.project_id }}.mta_transformed.node_mapping
*/
CREATE OR REPLACE TABLE `{{ params.project_id }}.mta_transformed.node_mapping` AS

WITH unique_nodes AS (
  SELECT DISTINCT
    node_id,
    route_id,
    direction,
    stop_id,
    stop_name,
    stop_lat,
    stop_lon
  FROM `{{ params.project_id }}.mta_transformed.headways_all_nodes`
  WHERE node_id IS NOT NULL
)

SELECT
  node_id,
  -- Create integer index (0-based for numpy compatibility)
  ROW_NUMBER() OVER (ORDER BY route_id, direction, stop_name) - 1 AS node_index,
  route_id,
  direction,
  stop_id,
  stop_name,
  stop_lat,
  stop_lon
FROM unique_nodes
ORDER BY node_index;
