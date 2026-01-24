# Prototyping Patterns for Fast Iteration

This directory contains examples of architectural patterns that speed up development by reducing the "DevOps Tax" of container builds and cloud deployments.

## 1. Lightweight Python Components (`lightweight_component_example.py`)

**Concept:** 
Instead of building a full Docker image for every code change, Kubeflow Pipelines (KFP) can "pickle" a Python function and package it into a standard base image at deployment time.

**Workflow:**
1. Write a function decorated with `@dsl.component`.
2. Define dependencies in `packages_to_install` (e.g., `pandas`, `sklearn`).
3. Compile and upload pipeline.

**Pros:**
*   **Zero Build Time:** No `docker build`. No `docker push`.
*   **Self-Contained:** Dependencies live with the code definition.

**Cons:**
*   **Dependency Installation:** Installs packages at *runtime* (startup), which can add 30s-1min to every step execution.
*   **Complexity Limit:** Hard to share code between components unless you build a python package.

## 2. Local Runner (`local_runner_example.py`)

**Concept:**
A simple Python script that mimics the pipeline steps locally on your machine, bypassing KFP, Vertex AI, and Docker entirely.

**Workflow:**
1. Import your `Trainer`, `Preprocessor`, etc. classes directly.
2. Mock the inputs (create dummy pandas DataFrames or small CSVs).
3. Run the logic in your IDE debugger (`F5`).

**Pros:**
*   **Instant Feedback:** Catch syntax errors, shape mismatches, and logic bugs in seconds, not minutes.
*   **Full Debugging:** Use breakpoints, inspect variables.

**Cons:**
*   **No Hardware Parity:** Your laptop is not an A100 GPU cluster.
*   **Environment Drift:** "Works on my machine" vs "Fails in Cloud". Use Dev Containers to mitigate this.
