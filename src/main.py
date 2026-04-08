import sys
from pathlib import Path
import polars as pl
from loguru import logger

# Add src to path for internal imports
sys.path.append(str(Path(__file__).parent))

from utils import setup_logging, init_polars
from discovery.engine import FactorDiscoveryEngine


def load_and_preprocess(file_path: str) -> pl.DataFrame:
    """Load credit data and transform target to binary (0/1)."""
    logger.info(f"Loading raw data from: {file_path}")

    df = pl.read_csv(file_path)

    # In this dataset: 1 = Good, 2 = Bad (Default)
    # We map 1 -> 0 and 2 -> 1
    df = df.with_columns([((pl.col("default") == 2).cast(pl.Int8)).alias("target")])

    logger.info(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features.")
    return df


def main():
    """Main entry point for the RiskLens analytical pipeline."""
    setup_logging()
    init_polars()

    logger.info("Starting RiskLens: Factor Discovery Sequence on Real Data.")

    data_path = "data/raw/credit_risk.csv"
    if not Path(data_path).exists():
        logger.error(
            f"Data file not found at {data_path}. Please run download script or provide data."
        )
        sys.exit(1)

    # 1. Load Data
    df = load_and_preprocess(data_path)

    # 2. Initialize Engine
    engine = FactorDiscoveryEngine(target_column="target")

    # 3. Define features to analyze (numerical and categorical)
    # Excluding 'default' and our new 'target'
    all_features = [col for col in df.columns if col not in ["default", "target"]]

    # 4. Batch Discovery
    logger.info("Executing Batch Factor Discovery...")
    ranking = engine.batch_discovery(df, all_features)

    # 5. Output Findings
    print("\n" + "=" * 50)
    print("RISKLENS: FACTOR DISCOVERY RANKING")
    print("=" * 50)
    # Formatting the output for better readability
    with pl.Config(tbl_rows=30):
        print(ranking)
    print("=" * 50)

    # Save results
    output_path = "data/processed/factor_ranking.csv"
    ranking.write_csv(output_path)
    logger.info(f"Results successfully saved to {output_path}")


if __name__ == "__main__":
    main()
