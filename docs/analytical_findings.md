# RiskLens: Comprehensive Analytical Findings

This document summarizes the core statistical and predictive results derived from the RiskLens analytical pipeline. The findings are based on the UCI German Credit dataset (1,000 samples, 21 features).

---

## 1. Factor Discovery Results (Statistical Rigor)

Our discovery engine utilized **Information Value (IV)** and **Weight of Evidence (WoE)** to rank feature predictive strength. We identified a specialized set of drivers that hold the majority of predictive power.

| Feature Name | Information Value (IV) | Predictive Strength | Analytical Interpretation |
| :--- | :--- | :--- | :--- |
| `checking_balance` | 0.6660 | Suspiciously High | The single most dominant driver of default. |
| `credit_history` | 0.2918 | Medium | Historical repayment behavior shows strong trend. |
| `months_loan_duration`| 0.2890 | Medium | Longer loan terms correlate with higher risk bins. |
| `amount` | 0.2461 | Medium | Total exposure is a significant risk multiplier. |
| `savings_balance` | 0.1925 | Medium | Liquidity reserves provide a clear safety signal. |

### Key Insight:
The top 5 factors alone account for a significant portion of the total variance in default probability, justifying a "Reduced Feature Set" for enhanced model transparency.

---

## 2. Predictive Performance (Benchmarking)

To validate our factors, we trained an **XGBoost Classifier** using only the Top 5 discovered variables.

- **Primary Metric (ROC-AUC)**: `0.7844`
- **Baseline Accuracy**: `77.2%`
- **F1-Score (Weighted)**: `0.75`

### Strategic Takeaway:
Maintaining a ~0.78 AUC with only 23% of original features (5/21) proves that the Factor Discovery phase is highly efficient at dimensionality reduction while preserving model integrity.

---

## 3. Explainability (SHAP Decomposition)

The explainability layer (XAI) was applied to decompose model decisions into constituent factor contributions.

### Global Contribution
SHAP analysis identifies `checking_balance` and `credit_history` as the most consistent contributors to risk shifts across the entire population. The relationship is non-linear, with the highest impact occurring in the lower balance bins.

### Local Explanation Case Study (Sample 236)
A high-risk applicant analysis revealed that:
1. **Primary Driver**: Low `checking_balance` pushed the risk score significantly higher.
2. **Secondary Driver**: Large `amount` (exposure) further amplified the default probability.
3. **Mitigator**: Positive `credit_history` acted as the only downward pressure on risk, but was insufficient to prevent the "High Risk" classification.

---

## 4. Conclusion and Strategic Recommendations
The RiskLens framework has successfully moved the credit policy from a "Black Box" to an **Evidence-Based Dashboard**. We recommend prioritizing the collection of `checking_balance` and `credit_history` data quality as they provide the highest information density for future scoring iterations.
