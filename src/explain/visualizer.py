import sys
from pathlib import Path
import polars as pl
import matplotlib

# Use Agg backend for non-interactive plot generation (no popups)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from optbinning import OptimalBinning

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logging, init_polars


def generate_risk_plots(
    data_path: str = "data/raw/credit_risk.csv",
    ranking_path: str = "data/processed/factor_ranking.csv",
    top_n: int = 5,
):
    """Generate and save enhanced binning plots for the top factors."""
    setup_logging()
    init_polars()

    logger.info("Initializing RiskLens Visualization Engine.")

    # 1. Load Data and Ranking
    df = pl.read_csv(data_path)
    df = df.with_columns([((pl.col("default") == 2).cast(pl.Int8)).alias("target")])

    ranking = pl.read_csv(ranking_path)
    top_features = ranking["feature"].head(top_n).to_list()
    top_ivs = ranking["iv"].head(top_n).to_list()

    # 2. Iterate and Plot
    for i, feature in enumerate(top_features):
        logger.info(f"Generating optimized report for: {feature}")

        x = df[feature].to_numpy()
        y = df["target"].to_numpy()

        # Determine dtype
        col_type = df[feature].dtype
        dtype = "categorical" if col_type == pl.String else "numerical"

        optb = OptimalBinning(name=feature, dtype=dtype)
        optb.fit(x, y)

        # Build Table to extract labels
        bt = optb.binning_table
        bt.build()

        # Plot using OptBinning (Note: plt.show() might still trigger in some versions,
        # but Agg backend avoids GUI blocking)
        optb.binning_table.plot()

        # Enhance Current Plot
        fig = plt.gcf()
        fig.set_size_inches(12, 7)
        fig.suptitle(f"Risk Factor Analysis: {feature}", fontsize=16, fontweight="bold")
        plt.title(f"Information Value (IV): {top_ivs[i]:.4f}", fontsize=12, pad=10)

        # Save
        output_file = f"reports/figures/binning_{feature}.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=150)
        plt.close()

        logger.info(f"Finalized report: {output_file}")

    print("\n" + "=" * 50)
    print("VISUALIZATION SEQUENCE COMPLETE")
    print(f"Total reports generated: {len(top_features)}")
    print("Files saved to: reports/figures/")
    print("=" * 50)


if __name__ == "__main__":
    generate_risk_plots()
