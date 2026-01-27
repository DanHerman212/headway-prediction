
import os
import sys
import time
import subprocess
import typer
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv, find_dotenv

# Load Environment Variables
load_dotenv(find_dotenv())

app = typer.Typer(help="MLOps Pipeline CLI Manager")
console = Console()

def get_env_var(name: str, default: str = None) -> str:
    """Helper to get env vars with error handling."""
    val = os.getenv(name, default)
    if not val and default is None:
        console.print(f"[bold red]Error: Environment variable {name} is not set.[/bold red]")
        raise typer.Exit(code=1)
    return val

@app.command()
def build(
    tag: Optional[str] = typer.Option(None, help="Custom image tag. Defaults to timestamp."),
    push: bool = typer.Option(True, help="Whether to push the image to the registry."),
    local: bool = typer.Option(False, "--local", help="Build locally using Docker instead of Cloud Build.")
):
    """
    Builds the Docker image for the pipeline components.
    """
    project_id = get_env_var("GCP_PROJECT_ID")
    # Using 'headway-pipelines/training' as defined in bash script
    image_base = f"us-docker.pkg.dev/{project_id}/headway-pipelines/training"
    
    if not tag:
        tag = f"v{int(time.time())}"
    
    image_uri = f"{image_base}:{tag}"
    
    console.print(Panel(f"Building Docker Image: [bold cyan]{image_uri}[/bold cyan]", title="Build Step"))
    
    # Check if local is a Typer Option (default) or explicit boolean
    is_local = local
    if isinstance(local, typer.models.OptionInfo):
         is_local = local.default

    if is_local:
        # Local Docker Build
        cmd = ["docker", "build", "-t", image_uri, "."]
        if push:
            # We don't usually push local builds unless asked, but we'll queue it up
            # Docker push is separate command
            pass
    else:
        # Cloud Build (Remote)
        cmd = ["gcloud", "builds", "submit", ".", "--tag", image_uri]
    
    if not push and not is_local:
        cmd.append("--no-source") 

    try:
        subprocess.run(cmd, check=True)
        console.print(f"[bold green]Successfully built: {image_uri}[/bold green]")
        
        if is_local and push:
            console.print("Pushing local build to registry...")
            subprocess.run(["docker", "push", image_uri], check=True)
            
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Build failed with exit code {e.returncode}[/bold red]")
        raise typer.Exit(code=1)
    
    return image_uri

@app.command()
def compile(
    image_uri: Optional[str] = typer.Option(None, help="Docker image URI to bake into the pipeline."),
    output: str = "headway_pipeline.json"
):
    """
    Compiles the KFP pipeline to a JSON file.
    """
    # Set the Image URI env var BEFORE importing pipeline logic
    if image_uri:
        os.environ["TENSORFLOW_IMAGE_URI"] = image_uri
        console.print(f"Setting Pipeline Image to: [cyan]{image_uri}[/cyan]")
    else:
        # Check current env
        current = os.getenv("TENSORFLOW_IMAGE_URI")
        if current:
            console.print(f"Using Environment Image: [cyan]{current}[/cyan]")
        else:
            console.print("[yellow]Warning: No image URI provided for compilation. Using default in pipeline.py[/yellow]")

    # Import here to pick up Env Vars
    try:
        from kfp import compiler
        # Import your pipeline function dynamically or assuming standard path
        # from pipeline import headway_pipeline -> this runs top-level code
        import pipeline
    except ImportError as e:
         console.print(f"[bold red]Failed to import KFP or pipeline module: {e}[/bold red]")
         raise typer.Exit(code=1)

    console.print(Panel(f"Compiling Pipeline to [bold]{output}[/bold]", title="Compile Step"))
    
    compiler.Compiler().compile(
        pipeline_func=pipeline.headway_pipeline,
        package_path=output
    )
    console.print(f"[bold green]Compilation successful: {output}[/bold green]")

@app.command()
def run(
    skip_build: bool = typer.Option(False, "--skip-build", "-s", help="Skip Docker build, use latest tag."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compile only, do not submit."),
    project_id: str = typer.Option(None, envvar="GCP_PROJECT_ID"),
    region: str = typer.Option("us-east1", envvar="VERTEX_LOCATION"),
    bucket: str = typer.Option(None, envvar="GCS_BUCKET_NAME"),
    service_account: str = typer.Option(None, envvar="SERVICE_ACCOUNT"),
    pipeline_root: str = typer.Option(None, envvar="PIPELINE_ROOT")
):
    """
    End-to-End Pipeline Execution: Build -> Compile -> Submit.
    """
    # 1. Build
    image_uri = os.getenv("TENSORFLOW_IMAGE_URI")
    
    # If explicit URI is provided in env, use it unless we are forcing a build
    if image_uri and skip_build:
        console.print(f"Using pinned image from environment: [cyan]{image_uri}[/cyan]")
    elif not skip_build:
        # Pass local=False explicitly to avoid Typer Option object evaluation issues
        image_uri = build(tag=None, push=True, local=False)
    else:
        # Fetch latest tag logic
        project_id_val = project_id or get_env_var("GCP_PROJECT_ID")
        repo_path = f"us-docker.pkg.dev/{project_id_val}/headway-pipelines/training"
        console.print(f"[dim]Searching for latest tag in {repo_path}...[/dim]")
        try:
            # Replicating the gcloud logic from bash script using subprocess because python artifact libraries are complex
            cmd = [
                "gcloud", "artifacts", "docker", "tags", "list", repo_path,
                "--sort-by=~UPDATE_TIME", "--limit=1", "--format=value(tag)"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            latest_tag = result.stdout.strip()
            if not latest_tag:
                console.print("[red]No tags found. Cannot skip build.[/red]")
                raise typer.Exit(code=1)
            
            image_uri = f"{repo_path}:{latest_tag}"
            console.print(f"Skipping build. Using latest image: [cyan]{image_uri}[/cyan]")
        except subprocess.CalledProcessError:
            console.print("[red]Failed to fetch latest tag.[/red]")
            raise typer.Exit(code=1)

    # 2. Compile
    compile(image_uri=image_uri)
    
    if dry_run:
        console.print("[yellow]Dry run execution complete. Pipeline not submitted.[/yellow]")
        return

    # 3. Submit
    console.print(Panel("Submitting Job to Vertex AI", title="Submit Step"))
    
    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id, location=region)
    
    # Determine Pipeline Root
    if not pipeline_root and bucket:
        pipeline_root = f"gs://{bucket}/pipeline_root"
    
    if not pipeline_root:
        console.print("[red]Pipeline Root not configured. Set PIPELINE_ROOT or GCS_BUCKET_NAME.[/red]")
        raise typer.Exit(code=1)

    ts = int(time.time())
    run_name = f"headway-run-{ts}"
    
    try:
        job = aiplatform.PipelineJob(
            display_name=f"headway-training-{ts}",
            template_path="headway_pipeline.json",
            pipeline_root=pipeline_root,
            enable_caching=True,
            parameter_values={
                "project_id": project_id,
                "region": region,
                "run_name": run_name
            }
        )
        
        sa = service_account if service_account and service_account != "None" else None
        
        console.print(f"Submitting job: [bold]{run_name}[/bold]")
        job.submit(service_account=sa)
        console.print(f"[bold green]Job Submitted![/bold green] Dashboard link: {job._dashboard_uri()}")
        
    except Exception as e:
        console.print(f"[bold red]Submission Failed: {e}[/bold red]")
        raise typer.Exit(code=1)

@app.command()
def local_train(
    config_name: str = "config", 
    overrides: list[str] = typer.Argument(None, help="Hydra overrides like 'training.epochs=10'")
):
    """
    Runs the training script locally using Hydra configuration.
    """
    cmd = ["python", "src/train.py", f"config_name={config_name}"]
    if overrides:
        cmd.extend(overrides)
        
    console.print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    app()
