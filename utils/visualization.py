"""
Layout visualization utilities for displaying layouts with elements and recommendations.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from core.schemas import Layout


def visualize_layout(
    layout: Layout,
    title: str = "Layout",
    ax: Optional[plt.Axes] = None,
    show_issues: Optional[List[Dict[str, Any]]] = None,
    show_recommendations: Optional[List[Dict[str, Any]]] = None,
    highlight_changes: Optional[List[int]] = None,
) -> plt.Axes:
    """
    Visualize a layout with its elements positioned on canvas.

    Args:
        layout: Layout object to visualize
        title: Title for the visualization
        ax: Matplotlib axis to plot on
        show_issues: List of quality issues to highlight
        show_recommendations: List of LLM recommendations to display
        highlight_changes: List of element indices that were modified

    Returns:
        The matplotlib axis used for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Set up canvas
    ax.set_xlim(0, layout.width)
    ax.set_ylim(0, layout.height)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Invert Y-axis to match layout coordinate system

    # Color mapping for element types
    element_colors = {
        "image": "#FF6B6B",  # Red
        "headline": "#4ECDC4",  # Teal
        "text": "#45B7D1",  # Blue
        "button": "#96CEB4",  # Green
        "background": "#FECA57",  # Yellow
        "icon": "#FF9FF3",  # Pink
        "logo": "#54A0FF",  # Light Blue
        "input": "#5F27CD",  # Purple
    }

    # Draw elements
    for i, element in enumerate(layout.elements):
        x, y = element.x, element.y
        w, h = element.width, element.height

        # Get color for element type
        base_color = element_colors.get(element.element_class, "#95A5A6")  # Default gray

        # Highlight changed elements with special border
        edge_color = "red" if highlight_changes and i in highlight_changes else "black"
        edge_width = 3 if highlight_changes and i in highlight_changes else 1

        # Draw rectangle for element
        rect = plt.Rectangle((x, y), w, h, facecolor=base_color, edgecolor=edge_color, alpha=0.7, linewidth=edge_width)
        ax.add_patch(rect)

        # Add element label
        center_x = x + w / 2
        center_y = y + h / 2

        # Choose text color based on element color brightness
        text_color = "white" if sum(int(base_color[i : i + 2], 16) for i in (1, 3, 5)) < 400 else "black"

        # Add element type and index
        label = f"{element.element_class}\n#{i}"
        if highlight_changes and i in highlight_changes:
            label += "\n(FIXED)"

        ax.text(
            center_x,
            center_y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    # Add canvas border
    canvas_rect = plt.Rectangle((0, 0), layout.width, layout.height, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(canvas_rect)

    # Set title with layout info
    full_title = f"{title}\nCanvas: {layout.width}x{layout.height}, Elements: {len(layout.elements)}, Quality: {layout.quality.upper()}"
    ax.set_title(full_title, fontsize=12, fontweight="bold")

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add legend for element types
    unique_types = list(set(elem.element_class for elem in layout.elements))
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=element_colors.get(elem_type, "#95A5A6"), label=elem_type.capitalize())
        for elem_type in unique_types
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    # Add issues and recommendations as text
    info_text = []
    if show_issues:
        info_text.append("🎯 QUALITY ISSUES:")
        for issue in show_issues[:3]:  # Show top 3 issues
            feature = issue.get("feature", "N/A")
            current = float(issue.get("current_value", 0))
            target = float(issue.get("target_value", 0))
            direction = issue.get("improvement_direction", "N/A")
            info_text.append(f"  • {feature}: {current:.2f} → {target:.2f} ({direction})")

    if show_recommendations:
        info_text.append("\n💡 LLM RECOMMENDATIONS:")
        for rec in show_recommendations[:2]:  # Show top 2 LLM recommendations
            feature = rec.get("feature", "N/A")
            suggestion = rec.get("recommendation", "N/A")[:100] + "..."
            info_text.append(f"  • {feature}: {suggestion}")

    if info_text:
        ax.text(
            1.02,
            0.5,
            "\n".join(info_text),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

    return ax


def apply_recommendations_to_layout(
    layout: Layout, quality_issues: List[Dict[str, Any]], llm_recommendations: List[Dict[str, Any]]
) -> Tuple[Layout, List[int]]:
    """
    Apply LLM recommendations to create a "fixed" version of the layout for visualization.

    Args:
        layout: Original layout to fix
        quality_issues: List of quality issues identified
        llm_recommendations: List of LLM recommendations

    Returns:
        Tuple of (fixed_layout, list_of_changed_element_indices)
    """
    # Create a deep copy of the layout to modify
    fixed_layout = copy.deepcopy(layout)
    changed_elements = []

    # Apply fixes based on quality issues
    for issue in quality_issues:
        feature = issue.get("feature", "")
        current_value = float(issue.get("current_value", 0))
        target_value = float(issue.get("target_value", 0))
        direction = issue.get("improvement_direction", "")

        # Apply feature-specific fixes
        if "spacing" in feature.lower() or "margin" in feature.lower():
            changed_elements.extend(_fix_spacing_issues(fixed_layout, feature, current_value, target_value, direction))

        elif "alignment" in feature.lower():
            changed_elements.extend(
                _fix_alignment_issues(fixed_layout, feature, current_value, target_value, direction)
            )

        elif "size" in feature.lower() or "width" in feature.lower() or "height" in feature.lower():
            changed_elements.extend(_fix_size_issues(fixed_layout, feature, current_value, target_value, direction))

        elif "position" in feature.lower() or "center" in feature.lower():
            changed_elements.extend(_fix_position_issues(fixed_layout, feature, current_value, target_value, direction))

        elif "aspect_ratio" in feature.lower():
            changed_elements.extend(
                _fix_aspect_ratio_issues(fixed_layout, feature, current_value, target_value, direction)
            )

    # Update layout quality to "pass" for demonstration
    fixed_layout.quality = "pass"

    return fixed_layout, list(set(changed_elements))


def _fix_spacing_issues(layout: Layout, feature: str, current: float, target: float, direction: str) -> List[int]:
    """Fix spacing-related issues in the layout."""
    changed = []

    # Simple spacing fix: adjust element positions to improve spacing
    if "increase" in direction.lower() and len(layout.elements) > 1:
        # Increase spacing between elements
        for i in range(1, len(layout.elements)):
            layout.elements[i].y += 10  # Move elements down to increase vertical spacing
            changed.append(i)

    return changed


def _fix_alignment_issues(layout: Layout, feature: str, current: float, target: float, direction: str) -> List[int]:
    """Fix alignment-related issues in the layout."""
    changed = []

    # Smart alignment fix: only align text-based elements, preserve visual elements
    if len(layout.elements) > 1:
        # Define which element types should NOT be auto-aligned (preserve visual impact)
        preserve_types = {"image", "background", "graphicshape"}

        # Get text-based elements that should be aligned
        text_elements = [
            (i, elem) for i, elem in enumerate(layout.elements) if elem.element_class.lower() not in preserve_types
        ]

        if len(text_elements) > 1:
            # Find the most common x-coordinate among text elements only
            text_x_positions = [elem.x for _, elem in text_elements]
            target_x = max(set(text_x_positions), key=text_x_positions.count)

            # Only align text elements to create consistent text column
            for i, elem in text_elements:
                if abs(elem.x - target_x) > 5:  # Only fix significantly misaligned elements
                    elem.x = target_x
                    changed.append(i)

    return changed


def _fix_size_issues(layout: Layout, feature: str, current: float, target: float, direction: str) -> List[int]:
    """Fix size-related issues in the layout."""
    changed = []

    # Define which element types should NOT be auto-resized (preserve visual impact)
    preserve_types = {"image", "background", "graphicshape"}

    # Simple size fix: adjust element sizes (but preserve important visual elements)
    change_factor = target / current if current > 0 else 1.2

    for i, elem in enumerate(layout.elements):
        # Skip resizing of visual elements that should maintain their impact
        if elem.element_class.lower() in preserve_types:
            continue

        if "width" in feature.lower():
            new_width = elem.width * change_factor
            # Ensure element fits within canvas
            if elem.x + new_width <= layout.width:
                elem.width = new_width
                changed.append(i)
        elif "height" in feature.lower():
            new_height = elem.height * change_factor
            # Ensure element fits within canvas
            if elem.y + new_height <= layout.height:
                elem.height = new_height
                changed.append(i)

    return changed


def _fix_position_issues(layout: Layout, feature: str, current: float, target: float, direction: str) -> List[int]:
    """Fix position-related issues in the layout."""
    changed = []

    # Define which element types should have limited position changes (preserve visual impact)
    preserve_types = {"image", "background", "graphicshape"}

    # Simple position fix: center elements or adjust positions (but preserve important visual elements)
    if "center" in feature.lower():
        for i, elem in enumerate(layout.elements):
            # Skip major position changes for visual elements that should maintain their impact
            if elem.element_class.lower() in preserve_types:
                continue

            # Center element horizontally
            new_x = (layout.width - elem.width) / 2
            if new_x != elem.x:
                elem.x = new_x
                changed.append(i)

    return changed


def _fix_aspect_ratio_issues(layout: Layout, feature: str, current: float, target: float, direction: str) -> List[int]:
    """Fix aspect ratio-related issues in the layout."""
    changed = []

    # Simple aspect ratio fix: adjust element proportions
    for i, elem in enumerate(layout.elements):
        current_ratio = elem.width / elem.height if elem.height > 0 else 1
        if abs(current_ratio - target) > 0.1:  # Only fix if significantly off
            if target > current_ratio:  # Need wider element
                elem.width = elem.height * target
            else:  # Need taller element
                elem.height = elem.width / target

            # Ensure element fits within canvas
            if elem.x + elem.width <= layout.width and elem.y + elem.height <= layout.height:
                changed.append(i)
            else:
                # Revert if doesn't fit
                if target > current_ratio:
                    elem.width = elem.height * current_ratio
                else:
                    elem.height = elem.width / current_ratio

    return changed


def visualize_before_after_layouts(
    original_layout: Layout,
    quality_issues: List[Dict[str, Any]],
    llm_recommendations: List[Dict[str, Any]],
    figsize: Tuple[int, int] = (20, 10),
) -> plt.Figure:
    """
    Create a before/after visualization showing original and fixed layouts side by side.

    Args:
        original_layout: The original failed layout
        quality_issues: List of quality issues identified
        llm_recommendations: List of LLM recommendations
        figsize: Figure size for the visualization

    Returns:
        The matplotlib figure containing the visualization
    """
    # Apply recommendations to create fixed layout
    fixed_layout, changed_elements = apply_recommendations_to_layout(
        original_layout, quality_issues, llm_recommendations
    )

    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Visualize original layout
    visualize_layout(
        original_layout,
        title=f"BEFORE: Failed Layout {original_layout.id[:8]}...",
        ax=ax1,
        show_issues=quality_issues,
        show_recommendations=llm_recommendations,
    )

    # Visualize fixed layout
    visualize_layout(
        fixed_layout, title=f"AFTER: Fixed Layout {fixed_layout.id[:8]}...", ax=ax2, highlight_changes=changed_elements
    )

    # Add overall title
    fig.suptitle("Layout Improvement: Before & After LLM Recommendations", fontsize=16, fontweight="bold", y=0.95)

    plt.tight_layout()
    return fig


def analyze_layout_improvements(
    original_layout: Layout, fixed_layout: Layout, changed_elements: List[int]
) -> Dict[str, Any]:
    """
    Analyze the improvements made to a layout.

    Args:
        original_layout: Original layout
        fixed_layout: Fixed layout
        changed_elements: List of element indices that were changed

    Returns:
        Dictionary containing improvement analysis
    """
    changes: List[Dict[str, Any]] = []
    improvements = {
        "total_elements": len(original_layout.elements),
        "elements_modified": len(changed_elements),
        "modification_percentage": len(changed_elements) / len(original_layout.elements) * 100,
        "changes": changes,
    }

    for elem_idx in changed_elements:
        if elem_idx < len(original_layout.elements) and elem_idx < len(fixed_layout.elements):
            orig_elem = original_layout.elements[elem_idx]
            fixed_elem = fixed_layout.elements[elem_idx]

            change_info = {
                "element_index": elem_idx,
                "element_type": orig_elem.element_class,
                "position_change": {"x": fixed_elem.x - orig_elem.x, "y": fixed_elem.y - orig_elem.y},
                "size_change": {
                    "width": fixed_elem.width - orig_elem.width,
                    "height": fixed_elem.height - orig_elem.height,
                },
            }

            changes.append(change_info)

    return improvements
