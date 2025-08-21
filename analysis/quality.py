"""
Quality Analysis Implementation.
Track 2: Find key geometric features that differentiate pass/fail layouts.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from features.geometric import extract_all_features
from scipy.stats import ttest_ind
from utils.logger import get_logger

from core.schemas import Layout

logger = get_logger(__name__)


def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Cohen's d measures the standardized difference between two means:
    d = (μ₁ - μ₂) / σ_pooled
    
    Where σ_pooled = √[((n₁-1)σ₁² + (n₂-1)σ₂²) / (n₁+n₂-2)]
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        float: Cohen's d effect size (absolute value)
    """
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
    
    # Calculate pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return abs(np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_feature_significance(feature_data: pd.DataFrame, 
                                quality_labels: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze statistical significance of features for quality differentiation.
    
    Uses independent samples t-test to determine which geometric features
    significantly distinguish between 'pass' and 'fail' layouts.
    
    Args:
        feature_data: DataFrame with geometric features
        quality_labels: List of quality labels ('pass' or 'fail')
        
    Returns:
        List of feature analysis results, sorted by significance
    """
    
    # Split data by quality
    pass_mask = [q == 'pass' for q in quality_labels]
    fail_mask = [q == 'fail' for q in quality_labels]
    
    pass_data = feature_data[pass_mask]
    fail_data = feature_data[fail_mask]
    
    significant_features = []
    
    # Test each feature
    for feature in feature_data.columns:
        if feature == 'id':
            continue
            
        pass_values = pass_data[feature].dropna()
        fail_values = fail_data[feature].dropna()
        
        if len(pass_values) == 0 or len(fail_values) == 0:
            continue
        
        # Perform t-test
        try:
            statistic, p_value = ttest_ind(pass_values, fail_values)
            effect_size = calculate_effect_size(pass_values.values, fail_values.values)
            
            # Calculate means for interpretation
            pass_mean = np.mean(pass_values)
            fail_mean = np.mean(fail_values)
            
            feature_info = {
                'feature': feature,
                'p_value': p_value,
                'effect_size': effect_size,
                'pass_mean': pass_mean,
                'fail_mean': fail_mean,
                'difference': pass_mean - fail_mean,
                'significant': p_value < 0.05,
                'large_effect': effect_size > 0.8
            }
            
            significant_features.append(feature_info)
            
        except Exception as e:
            logger.warning(f"Could not analyze feature '{feature}': {e}")
    
    # Sort by p-value (most significant first)
    significant_features.sort(key=lambda x: x['p_value'])
    
    return significant_features


def run_quality_analysis(layouts: List[Layout]) -> Dict[str, Any]:
    """Find key geometric features that differentiate pass/fail layouts."""
    logger.info("="*60)
    logger.info("QUALITY ANALYSIS - STATISTICAL DIFFERENTIATORS")
    logger.info("="*60)
    
    # Extract comprehensive features
    logger.info(f"Extracting comprehensive geometric features for {len(layouts)} layouts...")
    feature_records = [extract_all_features(layout) for layout in layouts]
    feature_df = pd.DataFrame(feature_records)
    
    # Get quality labels
    quality_labels = [layout.quality for layout in layouts]
    
    # Count pass/fail distribution
    pass_count = sum(1 for q in quality_labels if q == 'pass')
    fail_count = len(quality_labels) - pass_count
    
    logger.info(f"Dataset distribution: {pass_count} pass layouts, {fail_count} fail layouts")
    
    # Analyze feature significance
    logger.info("Analyzing statistical significance of features...")
    feature_analysis = analyze_feature_significance(
        feature_df.drop(columns=['id']), 
        quality_labels
    )
    
    # Report results
    logger.info("--- Key Quality Differentiators (Statistical Analysis) ---")
    
    significant_features = {}
    top_features = []
    
    for i, feature_info in enumerate(feature_analysis):
        feature = feature_info['feature']
        p_value = feature_info['p_value']
        effect_size = feature_info['effect_size']
        pass_mean = feature_info['pass_mean']
        fail_mean = feature_info['fail_mean']
        significant = feature_info['significant']
        large_effect = feature_info['large_effect']
        
        if significant:
            significant_features[feature] = p_value
            
            # Determine direction of effect
            direction = "higher" if pass_mean > fail_mean else "lower"
            
            logger.info(f"✅ {feature}:")
            logger.info(f"   p-value: {p_value:.4f}, effect size: {effect_size:.3f}")
            logger.info(f"   Pass layouts have {direction} values ({pass_mean:.3f} vs {fail_mean:.3f})")
            
            if large_effect:
                logger.info("   🎯 LARGE EFFECT - Strong quality predictor!")
                top_features.append(feature_info)
            
            logger.info("")
        
        # Only show top 10 for readability
        if i >= 9:
            break
    
    if not significant_features:
        logger.warning("⚠️  No statistically significant features found (p < 0.05)")
    else:
        logger.info(f"Summary: Found {len(significant_features)} statistically significant features")
        
        # Identify top predictive features
        large_effect_features = [f for f in feature_analysis if f['large_effect'] and f['significant']]
        
        if large_effect_features:
            logger.info("🎯 TOP QUALITY PREDICTORS (large effect size > 0.8):")
            for feature_info in large_effect_features[:5]:
                feature = feature_info['feature']
                effect_size = feature_info['effect_size']
                direction = "↑" if feature_info['difference'] > 0 else "↓"
                logger.info(f"   {direction} {feature} (effect size: {effect_size:.3f})")
        
        # Build quality score based on top features
        if large_effect_features:
            logger.info("📊 QUALITY SCORE FORMULA:")
            logger.info("Based on the most predictive features, a simple quality score could be:")
            
            score_components = []
            for feature_info in large_effect_features[:3]:  # Top 3 features
                feature = feature_info['feature']
                weight = "+" if feature_info['difference'] > 0 else "-"
                score_components.append(f"{weight} {feature}")
            
            logger.info(f"   Quality Score = {' '.join(score_components)}")
    
    return {
        'significant_features': significant_features,
        'feature_analysis': feature_analysis,
        'top_predictors': large_effect_features,
        'feature_df': feature_df,
        'quality_labels': quality_labels,
        'pass_count': pass_count,
        'fail_count': fail_count
    }


def main() -> Dict[str, Any]:
    """Main function for standalone execution."""
    from pathlib import Path

    from core.schemas import load_layouts_from_json

    # Load data
    data_path = Path(__file__).resolve().parent.parent / "data" / "01_raw" / "assignment_data.json"
    
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        logger.error("Please copy the assignment_data.json file to data/01_raw/")
        return {}
    
    layouts = load_layouts_from_json(str(data_path))
    
    # Run quality analysis
    results = run_quality_analysis(layouts)
    
    logger.info("✅ Quality analysis complete!")
    return results


if __name__ == "__main__":
    main()
