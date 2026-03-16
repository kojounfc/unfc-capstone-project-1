"""
Profit Erosion E-commerce Capstone Project - Source Package.

Modules:
    - config: Configuration constants and paths
    - data_processing: Data loading, cleaning, and merging
    - feature_engineering: Feature creation for profit erosion analysis
    - visualization: Plotting and chart generation
    - modeling: Profit erosion analysis and customer segmentation
"""
from src.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
)

__all__ = [
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "FIGURES_DIR",
]
