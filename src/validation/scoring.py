import sys
from pathlib import Path
import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import setup_logging, init_polars, get_standard_parser, load_and_preprocess


def run_benchmarking(data_path: str = "data/raw/credit_risk.csv", top_n: int = 5):
    """Benchmark a model using only the top discovered factors."""
    setup_logging()
    init_polars()

    # 1. Load Data via shared Utility
    df = load_and_preprocess(data_path)

    # 2. Select Top Features
    try:
        ranking = pl.read_csv("data/processed/factor_ranking.csv")
        features = ranking["feature"].head(5).to_list()
    except Exception:
        features = [col for col in df.columns if col not in ["target", "default"]]

    X = df.select(features).to_pandas()
    y = df["target"].to_numpy()

    logger.info(f"Benchmarking with features: {features}")

    # 3. Encoding Categorical Features
    categorical_cols = X.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        logger.info(f"Encoding categorical features: {list(categorical_cols)}")
        encoder = OrdinalEncoder()
        X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

    # 4. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    logger.info("Training XGBoost benchmark model...")
    model.fit(X_train, y_train)

    # 5. Evaluation
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, probs)

    print("\n" + "=" * 50)
    print("RISKLENS BENCHMARKING REPORT")
    print("=" * 50)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("=" * 50)


def main():
    parser = get_standard_parser("RiskLens: Benchmarking Engine")
    args = parser.parse_args()
    run_benchmarking(data_path=args.data)


if __name__ == "__main__":
    main()
