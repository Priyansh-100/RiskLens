# RiskLens: Execution Plan

## 1. Project Objective
RiskLens is designed to move beyond black-box predictive modeling in credit risk. The primary goal is **Factor Discovery**: identifying, validating, and explaining the specific variables that drive default risk in high-dimensional financial datasets.

---

## 2. Modern Tech Stack

Leveraging the highest standards of Python development:

- **Orchestration**: `uv` (Astral) for ultra-fast dependency management and project isolation.
- **Configuration**: centralized `pyproject.toml` for all tools (ruff, ty, build-system).
- **Runtime**: Python 3.14+ (leveraging free-threading where applicable for data ops).
- **Data Processing**: `Polars` (lazy-evaluation, multicore execution) over legacy Pandas.
- **Modeling**: 
  - **Explainable Boosting Machines (EBM)** via `InterpretML` for inherent transparency.
  - **LightGBM/XGBoost** for performance benchmarks.
- **Explainability (XAI)**: `SHAP` and `Lime` for post-hoc analysis.
- **Code Quality**:
  - `ruff`: Unified linter and formatter.
  - `ty`: Next-gen type checking and static analysis.

---

## 3. Implementation Phases

### Phase 1: Environment & Architecture Setup
- [ ] Initialize project with `uv init --lib`.
- [ ] Configure `pyproject.toml` with `ruff` and `ty` rules.
- [ ] Set up a structured directory layout:
  ```text
  RiskLens/
  ├── data/           # Raw and processed credit records
  ├── notebooks/      # Exploratory Factor Analysis
  ├── src/
  │   ├── discovery/  # Factor identifying logic
  │   ├── validation/ # ML model validation
  │   └── explain/    # XAI implementation
  └── pyproject.toml
  ```

### Phase 2: Factor Discovery & EDA
- [ ] **Data Ingestion**: Load raw credit data using Polars for high-speed I/O.
- [ ] **Statistical Discovery**: Use Mutual Information, Information Value (IV), and WOE (Weight of Evidence) to identify high-impact variables.
- [ ] **Redundancy Reduction**: Multi-collinearity analysis (VIF) and feature correlation pruning.

### Phase 3: Latent Structure Identification
- [ ] Apply **Principal Component Analysis (PCA)** or **Factor Analysis (FA)** to find latent financial dimensions (e.g., "Liquidity Stress", "Spending Volatility").
- [ ] Unsupervised clustering to identify distinct "Risk Personas" within the dataset.

### Phase 4: Validating Factors with Predictive ML
- [ ] Train a hybrid model (EBM + Gradient Boosting) on discovered factors.
- [ ] Benchmark "Reduced Feature Set" (Discovered Factors) against "Raw Feature Set" to ensure minimal loss in AUC/Gini.
- [ ] Evaluate model robustness via Cross-Validation.

### Phase 5: XAI & Interpretation
- [ ] Generate **SHAP Summary Plots** to visualize global factor importance.
- [ ] Implement **Waterfall plots** for individual sample explanations (Local interpretability).
- [ ] Compile a "Risk Factor Dictionary" documenting discovered drivers and their causal intuitions.

---

## 4. Success Criteria

1. **Dimensionality Reduction**: Reduce raw features by >70% while maintaining >95% of original model performance.
2. **Transparency**: Every model prediction must be decomposable into constituent factor contributions.
3. **Reproducibility**: Environment and pipeline must be reproducible via a single `uv sync` command.
4. **Code Quality**: Zero linting/type errors as enforced by `ruff` and `ty`.
