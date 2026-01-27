import argparse
from google.cloud import aiplatform

def delete_all_experiments(project_id, location):
    print(f"Initializing Vertex AI with Project: {project_id}, Location: {location}")
    aiplatform.init(project=project_id, location=location)

    print("Listing all experiments...")
    experiments = aiplatform.Experiment.list()

    if not experiments:
        print("No experiments found.")
        return

    print(f"Found {len(experiments)} experiments.")

    for experiment in experiments:
        print(f"Deleting experiment: {experiment.name} (Resource Name: {experiment.resource_name})")
        try:
            # delete() deletes the experiment. 
            # Note: This might not delete the backing TensorBoard runs if they are associated, 
            # but it removes the experiment metadata from Vertex AI.
            experiment.delete() 
            print(f"Successfully deleted: {experiment.name}")
        except Exception as e:
            print(f"Failed to delete {experiment.name}: {e}")

    print("Deletion process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete ALL Vertex AI Experiments in a project")
    parser.add_argument("--project_id", type=str, required=True, help="GCP Project ID")
    parser.add_argument("--location", type=str, default="us-east1", help="GCP Region")

    args = parser.parse_args()

    # interactive confirmation
    confirm = input(f"WARNING: This will delete ALL experiments in {args.project_id}/{args.location}. Are you sure? (y/N): ")
    if confirm.lower() == 'y':
        delete_all_experiments(args.project_id, args.location)
    else:
        print("Operation cancelled.")
