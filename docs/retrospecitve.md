This is a fantastic retrospective question. You are validating a common pain point in MLOps: The gap between "Valid Python Code" and "Valid Cloud Pipeline" is deceptively large.

Your assumption—managing training from a Single Source of Truth (SSOT) with Modular Components—is absolutely the correct architectural pattern for a production-grade system. You are not over-optimizing; you are engineering for reliability.

However, the friction you experienced suggests that while the Pattern was right, the Tooling/Workflow imposed a heavy "DevOps Tax."

Here is an honest breakdown of better approaches or slight pivots that would have made this faster and easier to maintain.

1. The "Glue" Code Problem
The Issue: 90% of our bugs weren't ML logic; they were "Glue" bugs (e.g., argparse errors, environment variables not passing, serialization formats, path existence). We were manually bridging the gap between Kubeflow (KFP) and your Python scripts using brittle bash -c strings inside dsl.ContainerSpec.

A Better Approach:

Lightweight Wrappers (ZenML, Metaflow): Tools like ZenML or Metaflow act as a layer above KFP/Vertex. You write standard Python functions, and they handle the docker creation, argument parsing, and artifact passing automatically.
Result: You catch errors locally, and the "deployment script" is just python run.py --stack vertex.
Vertex AI Custom Jobs (running directly): Instead of jumping straight to a Pipeline (DAG), start by deploying train.py as a single Vertex AI Custom Job. This allows you to debug the cloud environment (GPU, permissions, paths) on a single container. Once stable, then split it into a Pipeline for ETL/Eval separation.
2. The Feedback Loop
The Issue: Your feedback loop was: Edit -> Commit -> Build Image (3 mins) -> Deploy -> Wait for Provisioning (5 mins) -> Fail -> Repeat. This is too slow for debugging syntax errors.

A Better Approach:

The "Local Mode" First: The debug_eval_local.py script we just wrote should have been a "First Class Citizen" of the repo, not an afterthought.
Dev Containers: Developing inside the exact Docker container used for training (using VS Code Dev Containers) prevents "it works on my machine" issues regarding libraries and versions.
3. Dependency Management
The Issue: We had conflicts where the Pipeline definition (Python code A) was sending arguments to a Docker Image (Python code B) that hadn't been rebuilt yet.

A Better Approach:

KFP Lightweight Python Components: Instead of building a massive Docker image and calling it via bash, you can decorate Python functions with @dsl.component. KFP will package that exact function's source code at deploy time.
Result: No mismatch between pipeline definition and executing code. No separate "build image" step for logic changes.
Summary: Did you make the right choice?
Yes.
Your architecture (Config Object + Artifacts on GCS + Managed TensorBoard) is robust. Now that it is working, it will be incredibly stable and auditable.

What to do for the next project:
If you start a new project next week, keep the Configuration Object, but simplify the deployment:

Use Lightweight Python Components for the pipeline steps (removes the Docker rebuild loop for code changes).
Write a Local Mock Runner immediately that runs the full chain on 10 rows of data on your laptop.
You have built a "Ferrari" of a pipeline. It was hard to tune the engine, but now that it's running, it will outperform a "hacky script" significantly.