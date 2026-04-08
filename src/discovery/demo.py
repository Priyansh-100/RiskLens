import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import polars as pl
import numpy as np
from discovery.engine import FactorDiscoveryEngine
from utils import setup_logging
from loguru import logger


def generate_synthetic_data(n_samples: int = 1000):
    """Generate synthetic credit data for demonstration."""
    np.random.seed(42)

    # Meaningful predictors
    income = np.random.normal(50000, 15000, n_samples)
    credit_score = np.random.normal(650, 50, n_samples)

    # Relationship to default (target)
    # Higher income/score = lower probability of default
    prob = 1 / (1 + np.exp(0.0001 * (income - 50000) + 0.05 * (credit_score - 650)))
    default = np.random.binomial(1, prob)

    # Noise predictor
    random_noise = np.random.normal(0, 1, n_samples)

    return pl.DataFrame(
        {
            "income": income,
            "credit_score": credit_score,
            "random_noise": random_noise,
            "default": default,
        }
    )


def run_demo():
    setup_logging()
    print(">>> Starting Synthetic Factor Discovery Demo.")
    logger.info("Starting Synthetic Factor Discovery Demo.")

    df = generate_synthetic_data()
    print(">>> Synthetic data generated.")
    engine = FactorDiscoveryEngine(target_column="default")

    features = ["income", "credit_score", "random_noise"]
    ranking = engine.batch_discovery(df, features)

    print("\nTarget Factor Ranking (by Information Value):")
    print(ranking)

    logger.info("Demo sequence completed.")


if __name__ == "__main__":
    run_demo()
