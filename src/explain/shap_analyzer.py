import sys
from pathlib import Path
import xgboost as xgb
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from loguru import logger
import polars as pl

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logging, init_polars, get_standard_parser, load_and_preprocess


def run_explainability_suite(data_path: str = "data/raw/credit_risk.csv"):
    """Generate SHAP-based global and local explanations for the risk model."""
    setup_logging()
    init_polars()

    # 1. Load Data via shared Utility
    df = load_and_preprocess(data_path)

    # 2. Dynamic Feature Selection
    try:
        ranking = pl.read_csv("data/processed/factor_ranking.csv")
        features = ranking["feature"].head(5).to_list()
    except Exception:
        features = [col for col in df.columns if col not in ["target", "default"]]

    X = df.select(features).to_pandas()
    y = df["target"].to_numpy()

    # 3. Encoding and Modeling
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    X_df = pl.DataFrame(X_encoded, schema=features).to_pandas()

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_df, y)

    # 4. SHAP Analysis
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # 5. Global Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.title("Global Factor Impact (SHAP)", fontsize=14, pad=20)
    plt.savefig("reports/figures/shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()

    # 6. Local Waterfall Plot
    high_risk_idx = model.predict_proba(X_df)[:, 1].argmax()
    exp = shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_df.iloc[high_risk_idx],
        feature_names=features,
    )
    shap.plots.waterfall(exp, show=False)
    plt.savefig(
        "reports/figures/shap_local_high_risk.png", bbox_inches="tight", dpi=150
    )
    plt.close()

    print("\n" + "=" * 50)
    print("XAI SUITE COMPLETE: check reports/figures/")
    print("=" * 50)


def main():
    parser = get_standard_parser("RiskLens: XAI Engine")
    args = parser.parse_args()
    run_explainability_suite(data_path=args.data)


if __name__ == "__main__":
    main()
