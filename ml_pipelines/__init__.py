"""
ML Pipeline for Headway Prediction

A modular, production-ready machine learning pipeline with comprehensive
experiment tracking via Vertex AI Experiments and TensorBoard integration.

Structure:
    config/         - Configuration management for models and tracking
    data/           - ETL pipeline from BigQuery and data processing
    models/         - Model architectures and components
    training/       - Training logic and optimization
    evaluation/     - Model evaluation and metrics
    tracking/       - Vertex AI Experiments + TensorBoard tracking
    pipelines/      - Orchestration for Vertex AI Pipelines (KFP)
"""

__version__ = "1.0.0"
