import os
import argparse
from google.cloud import aiplatform

def delete_experiment_runs(project_id, location, experiment_name):
    print(f"Initializing Vertex AI with Project: {project_id}, Location: {location}")
    aiplatform.init(project=project_id, location=location)

    try:
        experiment = aiplatform.Experiment(experiment_name=experiment_name)
        print(f"Found Experiment: {experiment.name}")
    except Exception as e:
        print(f"Experiment {experiment_name} not found or error accessing it: {e}")
        return

    print("Listing experiment runs...")
    # List all runs
    runs = aiplatform.ExperimentRun.list(experiment=experiment_name)
    
    if not runs:
        print("No runs found for this experiment.")
        return

    print(f"Found {len(runs)} runs. Deleting...")
    
    for run in runs:
        print(f"Deleting run: {run.name} ({run.resource_name})")
        try:
            run.delete()
            print(f"Deleted {run.name}")
        except Exception as e:
            print(f"Failed to delete {run.name}: {e}")

    print("All runs deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete all runs in a Vertex AI Experiment")
    parser.add_argument("--project_id", type=str, required=True, help="GCP Project ID")
    parser.add_argument("--location", type=str, default="us-east1", help="GCP Region")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment Name")

    args = parser.parse_args()

    delete_experiment_runs(args.project_id, args.location, args.experiment_name)
