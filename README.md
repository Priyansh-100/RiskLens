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
RiskLens is now fully dynamic. You can target the internal datasets or pass paths to external files using the `--data` flag.

- **Factor Discovery (Main)**:
  ```bash
  uv run risklens --data path/to/your_data.csv
  ```
- **Visual Risk Reports**:
  ```bash
  uv run visualize-factors --data path/to/your_data.csv
  ```
- **Benchmark Model**:
  ```bash
  uv run benchmark-model --data path/to/your_data.csv
  ```
- **XAI Explanations**:
  ```bash
  uv run explain-model --data path/to/your_data.csv
  ```

### 3. Dataset Compatibility
The engine automatically detects and maps common credit schemas:
- **Standard Format**: Detects `default` (1=Good, 2=Bad) and maps to proper binary targets.
- **Kaggle Format**: Detects `loan_status` (0=Non-Default, 1=Default) and performs auto-renaming.
- **Universal Mode**: For any other schema, simply ensure your CSV has a `target` or `default` column.

### 4. Development & Quality Assurance
To maintain technical integrity, we follow a strict DRY (Don't Repeat Yourself) architecture with shared utilities:
```bash
# Quality Checks
uv run ruff check src --fix
uvx mypy src --ignore-missing-imports
```
