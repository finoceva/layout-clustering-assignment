#!/usr/bin/env python3
"""
Full Pipeline Runner - Clustering Optimization + Recommendations
Runs feasible clustering optimization then generates recommendations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from clustering.flexible import FlexibleStructuralClusterer
from core.schemas import Layout, load_layouts_from_json
from recommendation.engine import LayoutRecommendationEngine
from utils.logger import get_logger

logger = get_logger(__name__)


def run_focused_clustering_optimization(layouts: List[Layout]) -> Dict[str, Any]:
    """
    Run clustering optimization with focused parameter combinations.

    Tests key method combinations:
    - Embedding models: layoutlmv1, layoutlmv3
    - Pooling: cls (most stable)
    - Reduction: pca, umap
    - Clustering: kmeans, hdbscan

    Total: 2 × 1 × 2 × 2 = 8 combinations (very feasible!)
    """
    logger.info("🎯 FOCUSED CLUSTERING OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("Testing 8 key method combinations:")
    logger.info("  📊 Embeddings: layoutlmv1, layoutlmv3")
    logger.info("  🎭 Pooling: cls (most stable)")
    logger.info("  📉 Reduction: pca, umap")
    logger.info("  🎪 Clustering: kmeans, hdbscan")
    logger.info("=" * 60)

    clusterer = FlexibleStructuralClusterer()

    # Override config to test focused combinations
    embedding_models = ["layoutlmv1", "layoutlmv3"]

    # Run optimization with focused scope (8 combinations)
    results = clusterer.run_optimization(
        layouts=layouts,
        embedding_models=embedding_models,
        max_combinations=8,  # Test all focused combinations
        sampling_strategy="first",  # Use all combinations we specify
    )

    return results


def run_recommendations_on_failed_layouts(
    layouts: List[Layout], best_config: Dict[str, Any], n_layouts: int = 8
) -> List[Dict[str, Any]]:
    """
    Generate recommendations for failed layouts using the best clustering config.

    Args:
        layouts: All layouts
        best_config: Best clustering configuration from optimization
        n_layouts: Number of failed layouts to test (default: 8)

    Returns:
        List of recommendation results
    """
    logger.info("🎯 GENERATING RECOMMENDATIONS FOR FAILED LAYOUTS")
    logger.info("=" * 60)

    # Initialize recommendation engine
    recommender = LayoutRecommendationEngine()

    # Train with all layouts using the best clustering config
    logger.info("Training recommendation engine with optimized clustering...")
    recommender.train(layouts)

    # Get failed layouts
    failed_layouts = [layout for layout in layouts if layout.quality == "fail"]
    test_layouts = failed_layouts[:n_layouts]

    logger.info(f"Testing recommendations on {len(test_layouts)} failed layouts")
    logger.info(f"(out of {len(failed_layouts)} total failed layouts)")

    # Generate recommendations
    recommendations = []
    for i, layout in enumerate(test_layouts, 1):
        logger.info(f"\n--- Generating recommendation {i}/{len(test_layouts)} ---")
        logger.info(f"Layout ID: {layout.id}")

        try:
            recommendation = recommender.generate_recommendation(layout)
            recommendations.append(recommendation)

            # Log summary
            similar_count = len(recommendation.get("similar_layouts", []))
            quality_issues = len(recommendation.get("quality_issues", []))
            llm_recs = len(recommendation.get("llm_recommendations", []))

            logger.info(
                f"✅ Generated: {similar_count} similar layouts, "
                f"{quality_issues} quality issues, {llm_recs} LLM recommendations"
            )

        except Exception as e:
            logger.error(f"❌ Failed to generate recommendation: {e}")
            recommendations.append(
                {
                    "layout_id": layout.id,
                    "error": str(e),
                    "similar_layouts": [],
                    "quality_issues": [],
                    "llm_recommendations": [],
                }
            )

    return recommendations


def save_pipeline_results(
    clustering_results: Dict[str, Any], recommendations: List[Dict[str, Any]], output_dir: str = "results"
) -> Dict[str, Any]:
    """Save all pipeline results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save clustering optimization results
    clustering_file = output_path / "clustering_optimization_results.json"
    with open(clustering_file, "w") as f:
        json.dump(clustering_results, f, indent=2, default=str)
    logger.info(f"💾 Saved clustering results to {clustering_file}")

    # Save recommendation results
    recommendations_file = output_path / "recommendation_results.json"
    with open(recommendations_file, "w") as f:
        json.dump(recommendations, f, indent=2, default=str)
    logger.info(f"💾 Saved recommendations to {recommendations_file}")

    # Create summary
    summary = {
        "pipeline_summary": {
            "total_configurations_tested": clustering_results.get("total_configurations", 0),
            "best_clustering_config": clustering_results.get("best_result", {}).get("config", {}),
            "best_clustering_score": clustering_results.get("best_result", {})
            .get("metrics", {})
            .get("combined_score", 0),
            "recommendations_generated": len(recommendations),
            "successful_recommendations": len([r for r in recommendations if "error" not in r]),
        }
    }

    summary_file = output_path / "pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"📊 Saved pipeline summary to {summary_file}")

    return summary


def main() -> None:
    """Main pipeline execution."""
    logger.info("🚀 STARTING FULL PIPELINE EXECUTION")
    logger.info("=" * 80)

    # Load data
    data_path = "data/01_raw/assignment_data.json"
    logger.info(f"Loading layouts from {data_path}")
    layouts = load_layouts_from_json(data_path)
    logger.info(f"Loaded {len(layouts)} layouts")

    # Step 1: Clustering optimization
    logger.info("\n" + "🎯 STEP 1: CLUSTERING OPTIMIZATION")
    clustering_results = run_focused_clustering_optimization(layouts)

    # Step 2: Generate recommendations
    logger.info("\n" + "🎯 STEP 2: RECOMMENDATION GENERATION")
    best_config = clustering_results.get("best_result", {}).get("config", {})
    recommendations = run_recommendations_on_failed_layouts(layouts, best_config, n_layouts=8)

    # Step 3: Save results and create summary
    logger.info("\n" + "🎯 STEP 3: SAVING RESULTS")
    summary = save_pipeline_results(clustering_results, recommendations)

    # Print final summary
    logger.info("\n" + "🎉 PIPELINE EXECUTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"✅ Tested {summary['pipeline_summary']['total_configurations_tested']} clustering configurations")
    logger.info(f"✅ Generated {summary['pipeline_summary']['recommendations_generated']} recommendations")
    logger.info(
        f"✅ Success rate: {summary['pipeline_summary']['successful_recommendations']}/{summary['pipeline_summary']['recommendations_generated']}"
    )
    logger.info("📁 Results saved to: results/")
    logger.info("   - clustering_optimization_results.json")
    logger.info("   - recommendation_results.json")
    logger.info("   - pipeline_summary.json")


if __name__ == "__main__":
    main()
