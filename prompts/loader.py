"""
Simple YAML prompt loader for layout recommendations.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml


class PromptLoader:
    """Loads and manages YAML prompt templates."""

    def __init__(self) -> None:
        """Initialize the prompt loader."""
        self.prompts_dir = Path(__file__).parent
        self.base_prompts: Dict[str, str] = {}
        self.feature_strategies: Dict[str, Dict[str, str]] = {}
        self.design_principles: Dict[str, List[str]] = {}
        self._load_all_prompts()

    def _load_all_prompts(self) -> None:
        """Load all prompt files."""
        try:
            # Load base prompts
            with open(self.prompts_dir / "base_prompt.yaml") as f:
                self.base_prompts = yaml.safe_load(f)

            # Load feature strategies
            with open(self.prompts_dir / "feature_strategies.yaml") as f:
                self.feature_strategies = yaml.safe_load(f)

            # Load design principles
            with open(self.prompts_dir / "design_principles.yaml") as f:
                self.design_principles = yaml.safe_load(f)

        except Exception as e:
            print(f"Warning: Failed to load some prompts: {e}")

    def get_base_prompt(self, prompt_name: str) -> str:
        """Get a base prompt template."""
        return self.base_prompts.get(prompt_name, "")

    def get_feature_strategy(self, feature_name: str, strategy_type: str) -> str:
        """Get feature-specific strategy text."""
        feature_data = self.feature_strategies.get(feature_name, {})
        return feature_data.get(strategy_type, "")

    def get_design_principles(self, feature_name: str) -> List[str]:
        """Get design principles for a feature."""
        return self.design_principles.get(feature_name, [])

    def format_layout_structure(self, layout: Any) -> str:
        """Format layout structure using template."""
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
            detail = f"  {i + 1}. {elem.element_class}: ({elem.x}, {elem.y}) {elem.width}×{elem.height}px"
            element_details.append(detail)

        if len(layout.elements) > 8:
            element_details.append(f"  ... and {len(layout.elements) - 8} more elements")

        template = self.get_base_prompt("layout_structure_template")
        if template:
            return template.format(
                canvas_width=str(layout.width),
                canvas_height=str(layout.height),
                element_count=str(len(layout.elements)),
                element_types=", ".join([f"{count} {type}" for type, count in element_types.items()]),
                content_left=str(min_x),
                content_top=str(min_y),
                content_right=str(max_x),
                content_bottom=str(max_y),
                margin_left=str(margin_left),
                margin_right=str(margin_right),
                margin_top=str(margin_top),
                margin_bottom=str(margin_bottom),
                unique_x_positions=str(unique_x),
                unique_y_positions=str(unique_y),
                unique_element_sizes=str(unique_sizes),
                element_details="\n".join(element_details),
            )

        return f"Layout: {layout.width}×{layout.height}, {len(layout.elements)} elements"

    def format_improvement_strategies(self, feature_name: str) -> str:
        """Format improvement strategies for a feature."""
        common_issues = self.get_feature_strategy(feature_name, "common_issues")
        spatial_adjustments = self.get_feature_strategy(feature_name, "spatial_adjustments")
        expected_impact = self.get_feature_strategy(feature_name, "expected_impact")

        template = self.get_base_prompt("improvement_strategies_template")
        if template:
            return template.format(
                feature_name=feature_name,
                common_issues=common_issues,
                spatial_adjustments=spatial_adjustments,
                expected_impact=expected_impact,
            )

        return f"Focus on improving {feature_name}"

    def format_design_principles_context(self, feature_name: str) -> str:
        """Format design principles context for a feature."""
        principles = self.get_design_principles(feature_name)
        if principles:
            return "\n".join([f"- {principle}" for principle in principles])

        return f"Apply standard design principles for {feature_name}"
