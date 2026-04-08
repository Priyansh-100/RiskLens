import sys
from pathlib import Path
import polars as pl
import matplotlib

# Use Agg backend for non-interactive plot generation
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from optbinning import OptimalBinning

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logging, init_polars, get_standard_parser, load_and_preprocess


def generate_risk_plots(
    data_path: str = "data/raw/credit_risk.csv",
    ranking_path: str = "data/processed/factor_ranking.csv",
    top_n: int = 5,
):
    """Generate and save enhanced binning plots for the top factors."""
    setup_logging()
    init_polars()

    # 1. Load Data via shared Utility
    df = load_and_preprocess(data_path)

    # 2. Load Ranking Results
    ranking = pl.read_csv(ranking_path)
    top_features = ranking["feature"].head(top_n).to_list()
    top_ivs = ranking["iv"].head(top_n).to_list()

    # 3. Iterate and Plot
    for i, feature in enumerate(top_features):
        logger.info(f"Generating optimized report for: {feature}")

        x = df[feature].to_numpy()
        y = df["target"].to_numpy()

        # Determine dtype
        col_type = df[feature].dtype
        dtype = "categorical" if col_type == pl.String else "numerical"

        optb = OptimalBinning(name=feature, dtype=dtype)
        optb.fit(x, y)
        optb.binning_table.build()

        # Plot
        optb.binning_table.plot()

        # Enhance Plot
        fig = plt.gcf()
        fig.set_size_inches(12, 7)
        fig.suptitle(f"Risk Factor Analysis: {feature}", fontsize=16, fontweight="bold")
        plt.title(f"Information Value (IV): {top_ivs[i]:.4f}", fontsize=12, pad=10)

        # Save
        output_file = f"reports/figures/binning_{feature}.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=150)
        plt.close()

    print("\n" + "=" * 50)
    print("VISUALIZATION SEQUENCE COMPLETE")
    print("Total reports saved to: reports/figures/")
    print("=" * 50)


def main():
    parser = get_standard_parser("RiskLens: Visualization Engine")
    args = parser.parse_args()
    generate_risk_plots(data_path=args.data)


if __name__ == "__main__":
    main()
