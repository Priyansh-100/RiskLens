import polars as pl
from optbinning import OptimalBinning
from loguru import logger
from typing import Dict, Any

class FactorDiscoveryEngine:
    """
    Core engine for identifying hidden risk factors using 2026 analytical standards.
    Focuses on Weight of Evidence (WoE) and Information Value (IV) via Polars & OptBinning.
    """
    
    def __init__(self, target_column: str = "default"):
        self.target_column = target_column
        self.binning_models: Dict[str, Any] = {}
        logger.info(f"Discovery Engine initialized. Target: {target_column}")

    def analyze_feature(self, df: pl.DataFrame, feature_name: str) -> Dict[str, Any]:
        """
        Perform optimal binning and calculate IV for a single feature.
        """
        logger.info(f"Analyzing factor: {feature_name}")
        
        # Convert to numpy for OptBinning compatibility
        x = df[feature_name].to_numpy()
        y = df[self.target_column].to_numpy()
        
        optb = OptimalBinning(name=feature_name, dtype="numerical")
        optb.fit(x, y)
        
        binning_table = optb.binning_table
        # The table must be built to compute IV
        binning_table.build()
        iv = binning_table.iv
        
        # Interpret IV (Standard Rules of Thumb)
        strength = "v.weak"
        if iv > 0.5: strength = "suspiciously high"
        elif iv > 0.3: strength = "strong"
        elif iv > 0.1: strength = "medium"
        elif iv > 0.02: strength = "weak"
        
        logger.info(f"Factor: {feature_name} | IV: {iv:.4f} | Strength: {strength}")
        
        return {
            "feature": feature_name,
            "iv": iv,
            "strength": strength,
            "table": binning_table.build()
        }

    def batch_discovery(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        """
        Execute discovery across multiple potential factors.
        """
        results = []
        for feature in features:
            if feature == self.target_column:
                continue
            res = self.analyze_feature(df, feature)
            results.append({
                "feature": res["feature"],
                "iv": res["iv"],
                "predictive_strength": res["strength"]
            })
            
        return pl.DataFrame(results).sort("iv", descending=True)
