"""
Layout Recommendation Engine
Combines structural clustering and quality analysis for actionable recommendations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore
from analysis.quality import run_quality_analysis  # type: ignore
from clustering.structural import run_structural_clustering  # type: ignore
from features.geometric import extract_all_features  # type: ignore
from utils.logger import get_logger  # type: ignore

from core.schemas import Layout  # type: ignore

logger = get_logger(__name__)


class LayoutRecommendationEngine:
    """Two-track layout recommendation system with YAML-based prompts."""
    
    def __init__(self) -> None:
        """Initialize the recommendation engine."""
        self.structural_results: Optional[Dict[str, Any]] = None
        self.quality_results: Optional[Dict[str, Any]] = None
        self.layouts: Optional[List[Layout]] = None
        self.prompts: Optional[Dict[str, Any]] = None
        self.is_trained: bool = False
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load YAML prompt templates."""
        prompts_path = Path(__file__).resolve().parent.parent / "prompts" / "layout_improvement.yaml"
        
        try:
            with open(prompts_path, 'r') as f:
                self.prompts = yaml.safe_load(f)
            logger.info("Loaded prompt templates from YAML")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            self.prompts = {}
    
    def train(self, layouts: List[Layout]) -> None:
        """Train the recommender on a set of layouts."""
        logger.info("="*60)
        logger.info("LAYOUT RECOMMENDER - TRAINING")
        logger.info("="*60)
        
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
        logger.info("="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        
        # Check if results exist
        if self.structural_results is None or self.quality_results is None:
            logger.error("Training results not available")
            return
        
        # Structural results
        struct_sil = self.structural_results['silhouette_score']
        struct_clusters = self.structural_results['n_clusters']
        
        logger.info("🏗️  STRUCTURAL CLUSTERING:")
        logger.info(f"   • Method: LayoutLMv3 + PCA + KMeans")
        logger.info(f"   • Clusters found: {struct_clusters}")
        logger.info(f"   • Silhouette score: {struct_sil:.3f}")
        logger.info(f"   → Excellent for finding visually similar layouts")
        
        # Quality results
        sig_features = len(self.quality_results['significant_features'])
        top_predictors = len(self.quality_results['top_predictors'])
        
        logger.info("📊 QUALITY ANALYSIS:")
        logger.info(f"   • Significant features found: {sig_features}")
        logger.info(f"   • Strong predictors (large effect): {top_predictors}")
        
        if top_predictors > 0:
            logger.info(f"   • Top quality predictor: {self.quality_results['top_predictors'][0]['feature']}")
        
        logger.info(f"   → Excellent for understanding what makes layouts good/bad")
        
        logger.info("🎯 RECOMMENDATION CAPABILITY:")
        logger.info(f"   • Can find structurally similar layouts (Track 1)")
        logger.info(f"   • Can identify quality improvement areas (Track 2)")
        logger.info(f"   • Combines both for comprehensive recommendations")
    
    def find_similar_layouts(self, target_layout: Layout, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find structurally similar layouts using Track 1."""
        if not self.is_trained or self.structural_results is None or self.layouts is None:
            raise ValueError("Recommender must be trained first")
        
        # Find target layout's cluster
        layout_ids = self.structural_results['layout_ids']
        cluster_labels = self.structural_results['cluster_labels']
        
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
                similar_layouts.append({
                    'layout': layout,
                    'similarity_score': 1.0,  # Simplified
                    'cluster_id': cluster_id
                })
        
        return similar_layouts[:top_k]
    
    def analyze_quality_issues(self, layout: Layout) -> List[Dict[str, Any]]:
        """Analyze quality issues using Track 2."""
        if not self.is_trained or self.quality_results is None:
            raise ValueError("Recommender must be trained first")
        
        # Get layout features
        layout_features = extract_all_features(layout)
        
        # Compare against top quality predictors
        issues = []
        top_predictors = self.quality_results['top_predictors']
        
        for predictor in top_predictors[:5]:  # Top 5 predictors
            feature_name = predictor['feature']
            
            if feature_name in layout_features:
                current_value = layout_features[feature_name]
                pass_mean = predictor['pass_mean']
                fail_mean = predictor['fail_mean']
                
                # Determine if current value is closer to pass or fail
                dist_to_pass = abs(current_value - pass_mean)
                dist_to_fail = abs(current_value - fail_mean)
                
                if dist_to_fail < dist_to_pass:
                    # Layout value is closer to fail pattern
                    improvement_direction = "increase" if pass_mean > fail_mean else "decrease"
                    target_value = pass_mean
                    
                    issues.append({
                        'feature': feature_name,
                        'current_value': current_value,
                        'target_value': target_value,
                        'improvement_direction': improvement_direction,
                        'severity': predictor['effect_size'],
                        'p_value': predictor['p_value']
                    })
        
        # Sort by severity (effect size)
        issues.sort(key=lambda x: x['severity'], reverse=True)
        
        return issues
    
    def _analyze_layout_structure(self, layout: Layout) -> str:
        """Generate detailed layout structure description."""
        if not layout.elements:
            return "Layout contains no elements."
        
        # Basic layout info
        element_types: Dict[str, int] = {}
        for elem in layout.elements:
            elem_type = elem.element_class
            element_types[elem_type] = element_types.get(elem_type, 0) + 1
        
        # Content bounds
        min_x = min(elem.x for elem in layout.elements)
        max_x = max(elem.x + elem.width for elem in layout.elements)
        min_y = min(elem.y for elem in layout.elements)
        max_y = max(elem.y + elem.height for elem in layout.elements)
        
        # Margins
        margin_left = min_x
        margin_right = layout.width - max_x
        margin_top = min_y
        margin_bottom = layout.height - max_y
        
        # Unique positions and sizes
        unique_x = len(set(elem.x for elem in layout.elements))
        unique_y = len(set(elem.y for elem in layout.elements))
        unique_sizes = len(set((elem.width, elem.height) for elem in layout.elements))
        
        # Element details (first 8 elements)
        element_details = []
        for i, elem in enumerate(layout.elements[:8]):
            detail = f"  {i+1}. {elem.element_class}: ({elem.x}, {elem.y}) {elem.width}×{elem.height}px"
            element_details.append(detail)
        
        if len(layout.elements) > 8:
            element_details.append(f"  ... and {len(layout.elements) - 8} more elements")
        
        # Format using template
        if self.prompts and 'layout_structure_template' in self.prompts:
            return self.prompts['layout_structure_template'].format(
                canvas_width=layout.width,
                canvas_height=layout.height,
                element_count=len(layout.elements),
                element_types=", ".join([f"{count} {type}" for type, count in element_types.items()]),
                content_left=min_x,
                content_top=min_y,
                content_right=max_x,
                content_bottom=max_y,
                margin_left=margin_left,
                margin_right=margin_right,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
                unique_x_positions=unique_x,
                unique_y_positions=unique_y,
                unique_element_sizes=unique_sizes,
                element_details="\n".join(element_details)
            )
        
        return f"Layout: {layout.width}×{layout.height}, {len(layout.elements)} elements"
    
    def _get_improvement_strategies(self, feature_name: str, current_value: float, target_value: float) -> str:
        """Get improvement strategies for a specific feature."""
        if not self.prompts or 'feature_strategies' not in self.prompts or feature_name not in self.prompts['feature_strategies']:
            return f"Improve {feature_name} from {current_value:.3f} to {target_value:.3f}"
        
        strategies = self.prompts['feature_strategies'][feature_name]
        
        if self.prompts and 'improvement_strategies_template' in self.prompts:
            return self.prompts['improvement_strategies_template'].format(
                feature_name=feature_name,
                common_issues=strategies.get('common_issues', ''),
                spatial_adjustments=strategies.get('spatial_adjustments', ''),
                expected_impact=strategies.get('expected_impact', '')
            )
        
        return str(strategies)
    
    def _get_design_principles_context(self, feature_name: str) -> str:
        """Get design principles context for a feature."""
        if not self.prompts or 'design_principles' not in self.prompts or feature_name not in self.prompts['design_principles']:
            return f"Apply standard design principles for {feature_name}"
        
        principles = self.prompts['design_principles'][feature_name]
        return "\n".join([f"- {principle}" for principle in principles])
    
    def _generate_llm_prompt(self, layout_context: Dict[str, Any]) -> str:
        """Generate LLM prompt using YAML template."""
        if not self.prompts or 'base_recommendation_prompt' not in self.prompts:
            return "Please provide layout improvement recommendations."
        
        layout = layout_context['layout']
        issue = layout_context['issue']
        
        # Get components
        layout_structure = self._analyze_layout_structure(layout)
        improvement_strategies = self._get_improvement_strategies(
            issue['feature'], 
            issue['current_value'], 
            issue['target_value']
        )
        design_principles_context = self._get_design_principles_context(issue['feature'])
        
        # Format the main prompt
        return self.prompts['base_recommendation_prompt'].format(
            feature_name=issue['feature'],
            current_value=issue['current_value'],
            target_value=issue['target_value'],
            improvement_direction=issue['improvement_direction'],
            p_value=issue['p_value'],
            effect_size=issue['severity'],
            sample_size=len(self.layouts) if self.layouts else 0,
            confidence_level="95%" if issue['p_value'] < 0.05 else "Not significant",
            layout_structure=layout_structure,
            improvement_strategies=improvement_strategies,
            design_principles_context=design_principles_context
        )
    
    def _call_llm(self, prompt: str, feature_name: str) -> str:
        """Call LLM API (placeholder implementation)."""
        # This is a placeholder - in real implementation, you would call an actual LLM API
        
        # Feature-specific hardcoded responses for demo
        responses = {
            'edge_alignment_score': 
                "Align the left edges of your text elements to create a strong vertical line. "
                "Move the body text and button to share the same left margin as the headline, "
                "creating a clean column that guides the eye downward.",
            
            'balance_score': 
                "Move the large image element 40px closer to the center to balance the visual weight. "
                "The current layout is left-heavy - shifting the main content block rightward "
                "will create better equilibrium across the canvas.",
            
            'reading_flow_score': 
                "Reposition the headline 30px higher and ensure the body text flows directly below it. "
                "The current jumping pattern disrupts natural reading - create a clear "
                "top-to-bottom progression for better content comprehension.",
            
            'whitespace_ratio': 
                "Increase spacing between elements by 20px minimum and add 40px margins around the content area. "
                "The current cramped layout needs breathing room to improve readability and create "
                "a more premium appearance.",
            
            'size_hierarchy': 
                "Increase the headline size by 50% to establish clear visual hierarchy. "
                "The current similar sizing creates confusion about content importance - "
                "make the primary message unmistakably dominant."
        }
        
        return responses.get(feature_name, 
            f"Focus on improving the {feature_name} through strategic positioning and sizing adjustments.")
    
    def generate_recommendation(self, layout: Layout) -> Dict[str, Any]:
        """Generate comprehensive recommendation for a layout."""
        if not self.is_trained:
            raise ValueError("Recommender must be trained first")
        
        logger.info(f"🎯 GENERATING RECOMMENDATION FOR LAYOUT: {layout.id}")
        logger.info("="*50)
        
        # Track 1: Find similar layouts
        similar_layouts = self.find_similar_layouts(layout)
        
        logger.info("🏗️  STRUCTURAL SIMILARITY (Track 1):")
        if similar_layouts:
            logger.info(f"Found {len(similar_layouts)} structurally similar layouts:")
            for sim_layout in similar_layouts:
                sim_id = sim_layout['layout'].id
                sim_quality = sim_layout['layout'].quality
                logger.info(f"   • {sim_id} (quality: {sim_quality})")
        else:
            logger.info("   No similar layouts found in training set")
        
        # Track 2: Analyze quality issues
        quality_issues = self.analyze_quality_issues(layout)
        
        logger.info("📊 QUALITY ANALYSIS (Track 2):")
        if quality_issues:
            logger.info(f"Found {len(quality_issues)} potential improvement areas:")
            for i, issue in enumerate(quality_issues[:3]):  # Top 3 issues
                feature = issue['feature']
                direction = issue['improvement_direction']
                current = issue['current_value']
                target = issue['target_value']
                severity = issue['severity']
                
                logger.info(f"   {i+1}. {feature}:")
                logger.info(f"      Current: {current:.3f}, Target: {target:.3f}")
                logger.info(f"      Recommendation: {direction} this value")
                logger.info(f"      Impact: {'High' if severity > 0.8 else 'Medium' if severity > 0.5 else 'Low'}")
        else:
            logger.info("   No significant quality issues detected")
        
        # Generate LLM-enhanced recommendations
        llm_recommendations = []
        for issue in quality_issues[:2]:  # Top 2 issues for LLM enhancement
            prompt = self._generate_llm_prompt({
                'layout': layout,
                'issue': issue
            })
            
            recommendation = self._call_llm(prompt, issue['feature'])
            llm_recommendations.append({
                'feature': issue['feature'],
                'recommendation': recommendation,
                'prompt_used': prompt[:200] + "..." if len(prompt) > 200 else prompt
            })
        
        logger.info("💡 LLM-ENHANCED RECOMMENDATIONS:")
        for i, rec in enumerate(llm_recommendations):
            logger.info(f"   {i+1}. {rec['feature']}: {rec['recommendation']}")
        
        return {
            'layout_id': layout.id,
            'similar_layouts': similar_layouts,
            'quality_issues': quality_issues,
            'llm_recommendations': llm_recommendations,
            'overall_assessment': 'fail' if quality_issues else 'pass'
        }


def main() -> None:
    """Main function for standalone execution."""
    from pathlib import Path

    from core.schemas import load_layouts_from_json

    # Load data
    data_path = Path(__file__).resolve().parent.parent / "data" / "01_raw" / "assignment_data.json"
    
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        logger.error("Please copy the assignment_data.json file to data/01_raw/")
        return
    
    layouts = load_layouts_from_json(str(data_path))
    
    # Initialize and train recommender
    recommender = LayoutRecommendationEngine()
    recommender.train(layouts)
    
    # Test recommendation on a few layouts
    logger.info("="*60)
    logger.info("TESTING RECOMMENDATIONS")
    logger.info("="*60)
    
    # Test on first few fail layouts
    fail_layouts = [layout for layout in layouts if layout.quality == 'fail'][:2]
    
    for layout in fail_layouts:
        recommendation = recommender.generate_recommendation(layout)
        logger.info("-"*50)
    
    logger.info("✅ Recommendation system test complete!")


if __name__ == "__main__":
    main()
