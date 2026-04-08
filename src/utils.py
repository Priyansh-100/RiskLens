from loguru import logger
import polars as pl
import sys
import argparse
from pathlib import Path


def setup_logging():
    """Configure Loguru for professional output."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )


def init_polars():
    """Setup Polars configuration for modern data standards."""
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(10)
    logger.info("Polars engine initialized with high-performance configuration.")


def get_standard_parser(description: str):
    """Unified CLI argument parser for all RiskLens modules."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/credit_risk.csv",
        help="Path to the input CSV data file",
    )
    return parser


def load_and_preprocess(file_path: str) -> pl.DataFrame:
    """Standardized high-performance data ingestion and target mapping."""
    if not Path(file_path).exists():
        logger.error(f"Data file not found at: {file_path}")
        sys.exit(1)

    logger.info(f"Loading raw data from: {file_path}")
    df = pl.read_csv(file_path)

    # Standard Credit Risk Mapping: 1=Good (0), 2=Bad (1)
    if "default" in df.columns:
        df = df.with_columns([((pl.col("default") == 2).cast(pl.Int8)).alias("target")])
        logger.info(
            "Target mapping successful: {0 -> Good, 1 -> Bad} (Standard Format)"
        )
    elif "loan_status" in df.columns:
        # Many datasets (Kaggle) use 1 for default, 0 for non-default directly
        df = df.rename({"loan_status": "target"})
        logger.info(
            "Target mapping successful: {loan_status -> target} (Kaggle Format)"
        )

    logger.info(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features.")
    return df
