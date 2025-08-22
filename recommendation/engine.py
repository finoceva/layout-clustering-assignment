"""
Layout Recommendation Engine
Combines structural clustering and quality analysis for actionable recommendations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from analysis.quality import run_quality_analysis
from clustering.structural import run_structural_clustering
from config.manager import RecommendationConfigManager
from core.schemas import Layout
from features.geometric import extract_all_features
from prompts.loader import PromptLoader
from utils.logger import get_logger

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI package is required. Install with: uv add openai")

logger = get_logger(__name__)


class LayoutRecommendationEngine:
    """Two-track layout recommendation system with YAML-based prompts."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the recommendation engine.

        Args:
            config_path: Optional path to recommendation configuration file
        """
        self.structural_results: Optional[Dict[str, Any]] = None
        self.quality_results: Optional[Dict[str, Any]] = None
        self.layouts: Optional[List[Layout]] = None
        self.prompt_loader = PromptLoader()
        self.is_trained: bool = False

        # Load configuration
        self.config_manager = RecommendationConfigManager(config_path)
        self.config = self.config_manager.get_effective_config()

        # Validate configuration
        if self.config_manager.validate_config():
            logger.info("✅ Recommendation configuration loaded and validated")
        else:
            logger.warning("⚠️  Configuration validation failed, using defaults")

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'"
            )

        # Get OpenAI configuration from config manager
        openai_config = self.config.get("openai", {})
        self.model_name = openai_config.get("model", "gpt-4.1-mini")
        self.temperature = openai_config.get("temperature", 0.7)
        self.max_tokens = openai_config.get("max_tokens", 300)

        self.openai_client = OpenAI(api_key=api_key)

        logger.info("✅ OpenAI client initialized successfully")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Temperature: {self.temperature}")
        logger.info(f"   Max tokens: {self.max_tokens}")

        # Store other configuration settings
        self.training_config = self.config.get("training", {})
        self.recommendation_config = self.config.get("recommendation", {})
        self.logging_config = self.config.get("logging", {})

    def train(self, layouts: List[Layout]) -> None:
        """Train the recommender on a set of layouts."""
        logger.info("=" * 60)
        logger.info("LAYOUT RECOMMENDER - TRAINING")
        logger.info("=" * 60)

        self.layouts = layouts

        # Run Track 1: Structural clustering
        logger.info("🏗️  TRACK 1: STRUCTURAL ANALYSIS")
        self.structural_results = run_structural_clustering(layouts)

        # Run Track 2: Quality analysis
        logger.info("📊 TRACK 2: QUALITY ANALYSIS")
        self.quality_results = run_quality_analysis(layouts)

        self.is_trained = True
        logger.info("✅ Recommender training complete!")

        # Print summary
        self._print_training_summary()

    def _print_training_summary(self) -> None:
        """Print a summary of training results."""
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)

        # Check if results exist
        if self.structural_results is None or self.quality_results is None:
            logger.error("Training results not available")
            return

        # Structural results
        struct_sil = self.structural_results["silhouette_score"]
        struct_clusters = self.structural_results["n_clusters"]

        logger.info("🏗️  STRUCTURAL CLUSTERING:")
        logger.info("   • Method: LayoutLMv3 + PCA + KMeans")
        logger.info(f"   • Clusters found: {struct_clusters}")
        logger.info(f"   • Silhouette score: {struct_sil:.3f}")
        logger.info("   → Excellent for finding visually similar layouts")

        # Quality results
        sig_features = len(self.quality_results["significant_features"])
        top_predictors = len(self.quality_results["top_predictors"])

        logger.info("📊 QUALITY ANALYSIS:")
        logger.info(f"   • Significant features found: {sig_features}")
        logger.info(f"   • Strong predictors (large effect): {top_predictors}")

        if top_predictors > 0:
            logger.info(f"   • Top quality predictor: {self.quality_results['top_predictors'][0]['feature']}")

        logger.info("   → Excellent for understanding what makes layouts good/bad")

        logger.info("🎯 RECOMMENDATION CAPABILITY:")
        logger.info("   • Can find structurally similar layouts (Track 1)")
        logger.info("   • Can identify quality improvement areas (Track 2)")
        logger.info("   • Combines both for comprehensive recommendations")

    def find_similar_layouts(self, target_layout: Layout, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find structurally similar layouts using Track 1."""
        if not self.is_trained or self.structural_results is None or self.layouts is None:
            raise ValueError("Recommender must be trained first")

        # Use configuration default if not specified
        if top_k is None:
            top_k = self.recommendation_config.get("max_similar_layouts", 3)

        # Find target layout's cluster
        layout_ids = self.structural_results["layout_ids"]
        cluster_labels = self.structural_results["cluster_labels"]

        target_index = None
        for i, layout_id in enumerate(layout_ids):
            if layout_id == target_layout.id:
                target_index = i
                break

        if target_index is None:
            return []

        target_cluster = cluster_labels[target_index]

        # Find other layouts in the same cluster
        similar_layouts = []
        for i, (layout_id, cluster_id) in enumerate(zip(layout_ids, cluster_labels)):
            if cluster_id == target_cluster and layout_id != target_layout.id:
                layout = self.layouts[i]
                similar_layouts.append(
                    {
                        "layout": layout,
                        "similarity_score": 1.0,  # Simplified
                        "cluster_id": cluster_id,
                    }
                )

        return similar_layouts[:top_k]

    def analyze_quality_issues(self, layout: Layout) -> List[Dict[str, Any]]:
        """Analyze quality issues using Track 2."""
        if not self.is_trained or self.quality_results is None:
            raise ValueError("Recommender must be trained first")

        # Get layout features
        layout_features = extract_all_features(layout)

        # Compare against top quality predictors
        issues = []
        top_predictors = self.quality_results["top_predictors"]

        # Use configuration for max quality issues to analyze
        max_quality_issues = self.recommendation_config.get("max_quality_issues", 5)
        min_significance = self.recommendation_config.get("min_significance_level", 0.05)

        for predictor in top_predictors[:max_quality_issues]:
            feature_name = predictor["feature"]

            if feature_name in layout_features:
                current_value = layout_features[feature_name]
                pass_mean = predictor["pass_mean"]
                fail_mean = predictor["fail_mean"]

                # Check if feature meets significance threshold
                if predictor["p_value"] <= min_significance:
                    # Determine if current value is closer to pass or fail
                    dist_to_pass = abs(current_value - pass_mean)
                    dist_to_fail = abs(current_value - fail_mean)

                    if dist_to_fail < dist_to_pass:
                        # Layout value is closer to fail pattern
                        improvement_direction = "increase" if pass_mean > fail_mean else "decrease"
                        target_value = pass_mean

                        issues.append(
                            {
                                "feature": feature_name,
                                "current_value": current_value,
                                "target_value": target_value,
                                "improvement_direction": improvement_direction,
                                "severity": predictor["effect_size"],
                                "p_value": predictor["p_value"],
                            }
                        )

        # Sort by severity (effect size)
        issues.sort(key=lambda x: x["severity"], reverse=True)

        return issues

    def _analyze_layout_structure(self, layout: Layout) -> str:
        """Generate detailed layout structure description."""
        return str(self.prompt_loader.format_layout_structure(layout))

    def _get_improvement_strategies(self, feature_name: str, current_value: float, target_value: float) -> str:
        """Get improvement strategies for a specific feature."""
        return str(self.prompt_loader.format_improvement_strategies(feature_name))

    def _get_design_principles_context(self, feature_name: str) -> str:
        """Get design principles context for a feature."""
        return str(self.prompt_loader.format_design_principles_context(feature_name))

    def _generate_llm_prompt(self, layout_context: Dict[str, Any]) -> str:
        """Generate LLM prompt using YAML template."""
        base_prompt = self.prompt_loader.get_base_prompt("base_recommendation_prompt")
        if not base_prompt:
            return "Please provide layout improvement recommendations."

        layout = layout_context["layout"]
        issue = layout_context["issue"]

        # Get components
        layout_structure = self._analyze_layout_structure(layout)

        # Safely get values for improvement strategies (some methods might expect floats)
        current_val = issue.get("current_value", 0.0)
        target_val = issue.get("target_value", 0.0)
        try:
            current_val = float(current_val) if current_val is not None else 0.0
            target_val = float(target_val) if target_val is not None else 0.0
        except (ValueError, TypeError):
            current_val = 0.0
            target_val = 0.0

        improvement_strategies = self._get_improvement_strategies(
            issue.get("feature", "unknown"), current_val, target_val
        )
        design_principles_context = self._get_design_principles_context(issue.get("feature", "unknown"))

        # Safely convert values to ensure they're numeric before formatting
        def safe_float(value: Any, default: float = 0.0) -> float:
            """Safely convert a value to float, handling None and string cases."""
            try:
                if value is None:
                    return default
                return float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert value '{value}' to float, using default {default}")
                return default

        current_value = safe_float(issue.get("current_value"))
        target_value = safe_float(issue.get("target_value"))
        p_value = safe_float(issue.get("p_value"), 1.0)
        severity = safe_float(issue.get("severity"))

        # Format the main prompt - ensure all values are properly formatted
        try:
            # Convert template components to strings
            layout_str = str(layout_structure)
            strategies_str = str(improvement_strategies)
            principles_str = str(design_principles_context)

            # Format numeric values as nicely formatted strings for template
            formatted_prompt = str(
                base_prompt.format(
                    feature_name=str(issue.get("feature", "unknown")),
                    current_value=f"{current_value:.3f}",  # Format to 3 decimal places
                    target_value=f"{target_value:.3f}",  # Format to 3 decimal places
                    improvement_direction=str(issue.get("improvement_direction", "improve")),
                    p_value=f"{p_value:.4f}",  # Format to 4 decimal places for p-value
                    effect_size=f"{severity:.3f}",  # Format to 3 decimal places
                    sample_size=str(len(self.layouts) if self.layouts else 0),
                    confidence_level="95%" if p_value < 0.05 else "Not significant",
                    layout_structure=layout_str,
                    improvement_strategies=strategies_str,
                    design_principles_context=principles_str,
                )
            )

            return formatted_prompt

        except Exception as e:
            logger.error(f"Error in prompt formatting: {type(e).__name__}: {e}")
            raise

    def _call_llm(self, prompt: str, feature_name: str) -> str:
        """Call OpenAI API for layout recommendations using YAML prompts."""
        try:
            # Get system prompt from YAML
            system_prompt = self.prompt_loader.get_base_prompt("system_prompt")
            if not system_prompt:
                logger.warning("System prompt not found in YAML, using fallback")
                system_prompt = (
                    "You are an expert UI/UX designer providing specific, actionable layout recommendations. "
                    "Give precise, measurable suggestions for improving design layouts. "
                    "Focus on practical changes like spacing, alignment, sizing, and positioning. "
                    "Keep responses concise but detailed enough to implement."
                )

            response = self.openai_client.chat.completions.create(
                model=self.model_name,  # Configurable model
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,  # Configurable max tokens
                temperature=self.temperature,  # Configurable temperature
            )

            recommendation: str = (
                response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            )
            logger.info(f"✅ OpenAI recommendation generated for {feature_name}")

            # Log token usage if configured
            if self.logging_config.get("log_token_usage", True) and response.usage:
                logger.info(
                    f"   Tokens used: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})"
                )

            return recommendation

        except Exception as e:
            logger.error(f"Failed to call OpenAI API for {feature_name}: {e}")
            raise RuntimeError(f"OpenAI API call failed for {feature_name}: {e}") from e

    def generate_recommendation(self, layout: Layout) -> Dict[str, Any]:
        """Generate comprehensive recommendation for a layout."""
        if not self.is_trained:
            raise ValueError("Recommender must be trained first")

        logger.info(f"🎯 GENERATING RECOMMENDATION FOR LAYOUT: {layout.id}")
        logger.info("=" * 50)

        try:
            # Track 1: Find similar layouts
            similar_layouts = self.find_similar_layouts(layout)

            logger.info("🏗️  STRUCTURAL SIMILARITY (Track 1):")
            if similar_layouts:
                logger.info(f"Found {len(similar_layouts)} structurally similar layouts:")
                for sim_layout in similar_layouts:
                    sim_id = sim_layout["layout"].id
                    sim_quality = sim_layout["layout"].quality
                    logger.info(f"   • {sim_id} (quality: {sim_quality})")
            else:
                logger.info("   No similar layouts found in training set")

            # Track 2: Analyze quality issues
            quality_issues = self.analyze_quality_issues(layout)

            logger.info("📊 QUALITY ANALYSIS (Track 2):")
            if quality_issues:
                logger.info(f"Found {len(quality_issues)} potential improvement areas:")

                # Helper function for safe float conversion in logging
                def safe_float_for_logging(value: Any, default: float = 0.0) -> float:
                    try:
                        if value is None:
                            return default
                        return float(value)
                    except (ValueError, TypeError):
                        return default

                for i, issue in enumerate(quality_issues[:3]):  # Top 3 issues
                    feature = issue.get("feature", "unknown")
                    direction = issue.get("improvement_direction", "improve")
                    current = safe_float_for_logging(issue.get("current_value"))
                    target = safe_float_for_logging(issue.get("target_value"))
                    severity = safe_float_for_logging(issue.get("severity"))

                    logger.info(f"   {i + 1}. {feature}:")
                    logger.info(f"      Current: {current:.3f}, Target: {target:.3f}")
                    logger.info(f"      Recommendation: {direction} this value")
                    logger.info(f"      Impact: {'High' if severity > 0.8 else 'Medium' if severity > 0.5 else 'Low'}")
            else:
                logger.info("   No significant quality issues detected")

            # Generate LLM-enhanced recommendations
            llm_recommendations = []
            max_llm_recs = self.training_config.get("max_llm_enhanced_issues", 2)
            min_effect_size = self.training_config.get("min_effect_size_for_llm_enhancement", 0.5)

            # Filter issues by effect size threshold and limit by configuration
            # Safely handle severity comparison
            def get_severity(issue: Dict[str, Any]) -> float:
                try:
                    severity = issue.get("severity", 0.0)
                    return float(severity) if severity is not None else 0.0
                except (ValueError, TypeError):
                    return 0.0

            eligible_issues = [issue for issue in quality_issues if get_severity(issue) >= min_effect_size]

            for issue in eligible_issues[:max_llm_recs]:
                prompt = self._generate_llm_prompt({"layout": layout, "issue": issue})

                feature_name = issue.get("feature", "unknown")
                recommendation = self._call_llm(prompt, feature_name)
                llm_recommendations.append(
                    {
                        "feature": feature_name,
                        "recommendation": recommendation,
                        "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    }
                )

            logger.info("💡 LLM-ENHANCED RECOMMENDATIONS:")
            for i, rec in enumerate(llm_recommendations):
                logger.info(f"   {i + 1}. {rec['feature']}: {rec['recommendation']}")

            return {
                "layout_id": layout.id,
                "similar_layouts": similar_layouts,
                "quality_issues": quality_issues,
                "llm_recommendations": llm_recommendations,
                "overall_assessment": "fail" if quality_issues else "pass",
            }

        except Exception as e:
            logger.error(f"Error in generate_recommendation: {type(e).__name__}: {e}")
            raise


def failed_layout_recommendations(data_path: Union[str, Path], n_layouts: Optional[int] = None) -> None:
    """
    Test the recommendation system on fail layouts.

    Args:
        n_layouts: Number of fail layouts to test. If None, test all fail layouts.
    """
    from pathlib import Path

    from core.schemas import load_layouts_from_json

    # Load data
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")

    layouts = load_layouts_from_json(str(path))

    # Initialize and train recommender
    recommender = LayoutRecommendationEngine()
    recommender.train(layouts)

    # Test recommendation on specified number of layouts
    logger.info("=" * 60)
    logger.info("TESTING RECOMMENDATIONS")
    logger.info("=" * 60)

    # Get fail layouts and apply limit if specified
    fail_layouts = [layout for layout in layouts if layout.quality == "fail"]

    if n_layouts is not None:
        fail_layouts = fail_layouts[:n_layouts]
        logger.info(
            f"Testing on first {len(fail_layouts)} fail layouts (out of {len([layout for layout in layouts if layout.quality == 'fail'])} total)"
        )
    else:
        logger.info(f"Testing on all {len(fail_layouts)} fail layouts")

    for i, layout in enumerate(fail_layouts, 1):
        logger.info(f"\n--- Testing layout {i}/{len(fail_layouts)} ---")
        recommender.generate_recommendation(layout)
        logger.info("-" * 50)

    logger.info("✅ Recommendation system test complete!")


def main() -> None:
    """Main function for standalone execution with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the Layout Recommendation Engine", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d", "--data-path", type=str, default="data/01_raw/assignment_data.json", help="Path to the data file"
    )
    parser.add_argument(
        "-n",
        "--num-layouts",
        type=int,
        default=2,
        help="Number of fail layouts to test (default: 2, use 0 for all layouts)",
    )

    args = parser.parse_args()

    # Convert 0 to None for "all layouts"
    n_layouts = None if args.num_layouts == 0 else args.num_layouts

    failed_layout_recommendations(args.data_path, n_layouts)


if __name__ == "__main__":
    main()
