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

from utils import setup_logging, init_polars


def run_benchmarking(data_path: str = "data/raw/credit_risk.csv", top_n: int = 5):
    """Benchmark a model using only the top discovered factors."""
    setup_logging()
    init_polars()

    logger.info("Initializing RiskLens Benchmarking Sequence.")

    # 1. Load Data
    df = pl.read_csv(data_path)
    # Map target
    df = df.with_columns([((pl.col("default") == 2).cast(pl.Int8)).alias("target")])

    # 2. Select Features (Hardcoded based on discovery results or could be dynamic)
    # We use the top discovered factors
    features = [
        "checking_balance",
        "credit_history",
        "months_loan_duration",
        "amount",
        "savings_balance",
    ]

    X = df.select(features)
    y = df["target"].to_numpy()

    logger.info(f"Benchmarking with features: {features}")

    # 3. Encoding Categorical Features
    # Convert Polars to Pandas temporarily for easier SKLearn transformation
    X_pd = X.to_pandas()
    categorical_cols = X_pd.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:
        logger.info(f"Encoding categorical features: {list(categorical_cols)}")
        encoder = OrdinalEncoder()
        X_pd[categorical_cols] = encoder.fit_transform(X_pd[categorical_cols])

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pd, y, test_size=0.25, random_state=42
    )

    # 5. Train XGBoost
    # Using modern XGBoost parameters
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )

    logger.info("Training XGBoost benchmark model...")
    model.fit(X_train, y_train)

    # 6. Evaluation
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds)

    print("\n" + "=" * 50)
    print("RISKLENS BENCHMARKING REPORT")
    print("=" * 50)
    print("Model: XGBoost Classifier")
    print(f"Features Used: {len(features)}")
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("=" * 50)

    # 7. Feature Importance
    importance = model.feature_importances_
    for f, imp in zip(features, importance):
        logger.debug(f"XGBoost Importance | {f}: {imp:.4f}")


if __name__ == "__main__":
    run_benchmarking()
