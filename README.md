# RiskLens

## Predictive Factor Discovery and Credit Risk Analysis

RiskLens is a high-performance analytical framework designed for identifying, validating, and explaining hidden risk factors in credit datasets. Built for the rigorous demands of financial landscape, it prioritizes **factor transparency** and **statistical rigor** over black-box predictive accuracy.

The project provides a structured workflow for risk analysts to isolate meaningful drivers of default risk, quantify their impact, and ensure regulatory compliance through explainable machine learning (XAI).

---

## Core Capabilities

- **Factor Discovery**: Identifying latent structures using Weight of Evidence (WoE) and Information Value (IV) via `OptBinning`.
- **High-Performance EDA**: Leveraging `Polars` and `DuckDB` for multi-core, memory-efficient analysis of large-scale credit records.
- **Explainable AI (XAI)**: Native support for Explainable Boosting Machines (EBM) and SHAP value decomposition.
- **Modern Orchestration**: Managed entirely by `uv` for deterministic builds and ultra-fast execution.

---

## Technical Stack

- **Orchestration**: [uv](https://github.com/astral-sh/uv)
- **Data Engine**: [Polars](https://pypi.org/project/polars/) (Primary Dataframe engine)
- **Statistical Analytics**: [OptBinning](https://github.com/guillermo-navas-palencia/optbinning)
- **Modeling**: [InterpretML](https://github.com/interpretml/interpret), [XGBoost](https://github.com/dmlc/xgboost)
- **Quality Control**: [Ruff](https://github.com/astral-sh/ruff), [MyPy](https://github.com/python/mypy)

---

## Execution Guide

### 1. Environment Setup
Clone the repository and synchronize the environment:
```bash
uv sync
```

### 2. Available Commands
RiskLens uses simplified command aliases for common tasks:

- **Run Main Pipeline**:
  ```bash
  uv run risklens
  ```
- **Run Factor Discovery Demo**:
  ```bash
  uv run discovery-demo
  ```
- **Generate Visual Risk Reports**:
  ```bash
  uv run visualize-factors
  ```
- **Run Benchmark Model**:
  ```bash
  uv run benchmark-model
  ```
- **Generate XAI Explanations**:
  ```bash
  uv run explain-model
  ```

### 3. Development & Quality Assurance
To maintain modern development standards, run linting and type checks:
```bash
# Linting and Formatting
uv run ruff check src
uv run ruff format src

# Type Integrity
uvx mypy src
```
