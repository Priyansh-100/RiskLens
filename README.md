# RiskLens

## Statistical Factor Discovery and Credit Risk Analysis

RiskLens is an advanced analytical framework designed to identify and validate hidden risk factors within complex credit datasets. Unlike standard predictive models that prioritize accuracy at the expense of transparency, RiskLens focuses on exploratory factor discovery, statistical validation, and post-hoc interpretability.

The primary objective is to provide a structured workflow for risk analysts to isolate meaningful drivers of default risk, quantify their impact, and ensure regulatory compliance through explainable machine learning.

---

## Core Capabilities

- **Factor Discovery**: Identification of latent structures and predictive variables using modern statistical techniques.
- **Dimensionality Engineering**: Reduction of high-dimensional data into high-signal feature sets through collinearity analysis and information value assessment.
- **Predictive Validation**: Benchmarking discovered factors against state-of-the-art machine learning algorithms.
- **Explainable AI (XAI)**: Implementation of global and local explanation layers using SHAP and Explainable Boosting Machines (EBM).

---

## Technical Stack (2026 Standards)

The project leverages a modern Python ecosystem optimized for performance and reliability:

- **Environment & Dependency Management**: [uv](https://github.com/astral-sh/uv)
- **Data Engine**: [Polars](https://pypi.org/project/polars/) (Memory-efficient, multicore execution)
- **Modeling & XAI**: [InterpretML](https://github.com/interpretml/interpret), [SHAP](https://github.com/slundberg/shap), [XGBoost](https://github.com/dmlc/xgboost)
- **Code Quality**: [Ruff](https://github.com/astral-sh/ruff) (Linting and Formatting)
- **Configuration**: Unified `pyproject.toml`

---

## Getting Started

Ensure you have `uv` installed. To set up the development environment, execute:

```bash
uv sync
```

To run the full factor discovery pipeline:

```bash
uv run python src/main.py
```