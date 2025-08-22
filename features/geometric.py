"""
Geometric Feature Extraction for Layout Analysis

This module provides comprehensive geometric feature extraction capabilities for layout clustering.
Features are organized into categories based on their mathematical foundation and design purpose.

Feature Categories:
1. Basic Features: Element counts, areas, and densities
2. Spatial Features: Position distributions and spreads
3. Alignment Features: Edge alignment and grid adherence
4. Balance Features: Visual weight distribution and symmetry
5. Spacing Features: Inter-element distances and whitespace
6. Hierarchy Features: Size relationships and visual importance
7. Flow Features: Reading patterns and scanning behavior

Mathematical Foundations:
- Center of Mass: Weighted average position based on element areas
- Standard Deviation: Measure of spatial spread and consistency
- Euclidean Distance: Inter-element spacing calculations
- Ratio Analysis: Proportional relationships and hierarchies
"""

from typing import Dict, Union

import numpy as np
from loguru import logger

from core.schemas import Layout


def calculate_basic_features(layout: Layout) -> Dict[str, float]:
    """
    Calculate fundamental geometric properties of the layout.

    Mathematical basis:
    - Content Density = Σ(element_areas) / canvas_area
    - Average Element Size = Σ(element_areas) / element_count

    Args:
        layout: Layout object containing elements and dimensions

    Returns:
        Dict containing basic geometric features
    """
    elements = layout.elements

    if not elements:
        return {
            "element_count": 0,
            "content_density": 0.0,
            "avg_element_size": 0.0,
        }

    total_area = sum(elem.area for elem in elements)
    canvas_area = layout.width * layout.height

    return {
        "element_count": len(elements),
        "content_density": total_area / canvas_area if canvas_area > 0 else 0.0,
        "avg_element_size": total_area / len(elements),
    }


def calculate_spatial_features(layout: Layout) -> Dict[str, float]:
    """
    Calculate spatial distribution and position-based features.

    Mathematical basis:
    - Center of Mass: COM_x = Σ(x_i * area_i) / Σ(area_i)
    - Spatial Spread: σ = √(Σ(x_i - μ)² / N)
    - Asymmetry: |COM - canvas_center| / canvas_dimension

    Args:
        layout: Layout object

    Returns:
        Dict containing spatial distribution features
    """
    elements = layout.elements

    if not elements:
        return {
            "spatial_spread_x": 0.0,
            "spatial_spread_y": 0.0,
            "center_of_mass_x": 0.5,
            "center_of_mass_y": 0.5,
            "asymmetry_x": 0.0,
            "asymmetry_y": 0.0,
        }

    # Element centroids and areas
    centroids_x = [elem.x + elem.width / 2 for elem in elements]
    centroids_y = [elem.y + elem.height / 2 for elem in elements]
    areas = [elem.area for elem in elements]

    # Weighted center of mass calculation
    total_area = sum(areas) if areas else 1.0
    if total_area > 0:
        com_x = sum(x * area for x, area in zip(centroids_x, areas)) / total_area
        com_y = float(sum(y * area for y, area in zip(centroids_y, areas)) / total_area)
    else:
        com_x = float(np.mean(centroids_x))
        com_y = float(np.mean(centroids_y))

    # Normalize to layout dimensions
    com_x_norm = com_x / layout.width if layout.width > 0 else 0.5
    com_y_norm = com_y / layout.height if layout.height > 0 else 0.5

    # Spatial spread (standard deviation)
    spread_x = float(np.std(centroids_x)) / layout.width if len(centroids_x) > 1 and layout.width > 0 else 0.0
    spread_y = float(np.std(centroids_y)) / layout.height if len(centroids_y) > 1 and layout.height > 0 else 0.0

    # Asymmetry (deviation from center)
    asymmetry_x = abs(com_x_norm - 0.5)
    asymmetry_y = abs(com_y_norm - 0.5)

    return {
        "spatial_spread_x": spread_x,
        "spatial_spread_y": spread_y,
        "center_of_mass_x": com_x_norm,
        "center_of_mass_y": com_y_norm,
        "asymmetry_x": asymmetry_x,
        "asymmetry_y": asymmetry_y,
    }


def calculate_alignment_features(layout: Layout) -> Dict[str, float]:
    """
    Calculate edge alignment and grid adherence features.

    Mathematical basis:
    - Edge Alignment Score = aligned_edges / total_edge_pairs
    - Grid Adherence = Σ(grid_aligned_properties) / total_properties
    - Unique Position Penalty = unique_positions / total_elements

    Args:
        layout: Layout object

    Returns:
        Dict containing alignment and grid-based features
    """
    elements = layout.elements

    if len(elements) < 2:
        return {
            "edge_alignment_score": 1.0,
            "unique_x_positions": len(elements),
            "unique_y_positions": len(elements),
            "grid_adherence_score": 1.0,
        }

    # Collect edge positions
    left_edges = [elem.x for elem in elements]
    right_edges = [elem.x + elem.width for elem in elements]
    top_edges = [elem.y for elem in elements]
    bottom_edges = [elem.y + elem.height for elem in elements]

    all_edges = left_edges + right_edges + top_edges + bottom_edges

    # Count aligned edges (within tolerance)
    tolerance = 5  # pixels
    aligned_count = 0
    total_pairs = 0

    for i, edge1 in enumerate(all_edges):
        for edge2 in all_edges[i + 1 :]:
            total_pairs += 1
            if abs(edge1 - edge2) <= tolerance:
                aligned_count += 1

    edge_alignment_score = aligned_count / total_pairs if total_pairs > 0 else 0.0

    # Count unique positions
    unique_x = len(set(round(x / tolerance) * tolerance for x in left_edges))
    unique_y = len(set(round(y / tolerance) * tolerance for y in top_edges))

    # Grid adherence calculation
    grid_scores = []
    for grid_size in [20, 30, 40, 50]:  # Common grid sizes
        aligned_elements = 0.0
        for elem in elements:
            # Check alignment to grid
            x_aligned = (elem.x % grid_size <= tolerance) or (elem.x % grid_size >= grid_size - tolerance)
            y_aligned = (elem.y % grid_size <= tolerance) or (elem.y % grid_size >= grid_size - tolerance)
            w_aligned = (elem.width % grid_size <= tolerance) or (elem.width % grid_size >= grid_size - tolerance)
            h_aligned = (elem.height % grid_size <= tolerance) or (elem.height % grid_size >= grid_size - tolerance)

            alignment_score = sum([x_aligned, y_aligned, w_aligned, h_aligned]) / 4
            aligned_elements += alignment_score

        grid_scores.append(aligned_elements / len(elements))

    grid_adherence_score = max(grid_scores) if grid_scores else 0.0

    return {
        "edge_alignment_score": edge_alignment_score,
        "unique_x_positions": unique_x,
        "unique_y_positions": unique_y,
        "grid_adherence_score": grid_adherence_score,
    }


def calculate_balance_features(layout: Layout) -> Dict[str, float]:
    """
    Calculate visual balance and weight distribution features.

    Mathematical basis:
    - Balance Score = 1 - |weighted_center - canvas_center| / max_distance
    - Horizontal Balance = 1 - |COM_x - 0.5|
    - Vertical Balance = 1 - |COM_y - 0.5|
    - Weight Distribution = σ(weighted_positions)

    Args:
        layout: Layout object

    Returns:
        Dict containing balance and symmetry features
    """
    elements = layout.elements

    if not elements:
        return {
            "balance_score": 1.0,
            "horizontal_balance": 1.0,
            "vertical_balance": 1.0,
            "weight_center_x": 0.5,
            "weight_center_y": 0.5,
            "weight_distribution_x": 0.0,
            "weight_distribution_y": 0.0,
        }

    # Calculate weighted center of mass
    total_area = sum(elem.area for elem in elements)
    if total_area == 0:
        return {
            "balance_score": 1.0,
            "horizontal_balance": 1.0,
            "vertical_balance": 1.0,
            "weight_center_x": 0.5,
            "weight_center_y": 0.5,
            "weight_distribution_x": 0.0,
            "weight_distribution_y": 0.0,
        }

    # Weighted center calculation
    weighted_x = sum((elem.x + elem.width / 2) * elem.area for elem in elements) / total_area
    weighted_y = sum((elem.y + elem.height / 2) * elem.area for elem in elements) / total_area

    # Normalize to canvas dimensions
    weight_center_x = weighted_x / layout.width if layout.width > 0 else 0.5
    weight_center_y = weighted_y / layout.height if layout.height > 0 else 0.5

    # Balance calculations
    horizontal_balance = 1.0 - abs(weight_center_x - 0.5) * 2  # Scale to [0,1]
    vertical_balance = 1.0 - abs(weight_center_y - 0.5) * 2

    # Overall balance score (distance from center)
    center_distance = np.sqrt((weight_center_x - 0.5) ** 2 + (weight_center_y - 0.5) ** 2)
    balance_score = max(0.0, 1.0 - center_distance * 2)  # Scale to [0,1]

    # Weight distribution (spread of weighted positions)
    weighted_positions_x = [(elem.x + elem.width / 2) * elem.area / total_area for elem in elements]
    weighted_positions_y = [(elem.y + elem.height / 2) * elem.area / total_area for elem in elements]

    weight_distribution_x = float(np.std(weighted_positions_x)) if len(weighted_positions_x) > 1 else 0.0
    weight_distribution_y = float(np.std(weighted_positions_y)) if len(weighted_positions_y) > 1 else 0.0

    return {
        "balance_score": max(0.0, balance_score),
        "horizontal_balance": max(0.0, horizontal_balance),
        "vertical_balance": max(0.0, vertical_balance),
        "weight_center_x": weight_center_x,
        "weight_center_y": weight_center_y,
        "weight_distribution_x": weight_distribution_x,
        "weight_distribution_y": weight_distribution_y,
    }


def calculate_spacing_features(layout: Layout) -> Dict[str, float]:
    """
    Calculate inter-element spacing and whitespace features.

    Mathematical basis:
    - Minimum Distance = min(horizontal_gap, vertical_gap) between rectangles
    - Spacing Consistency = 1 / (1 + σ_spacing / μ_spacing)
    - Whitespace Ratio = (canvas_area - Σ(element_areas)) / canvas_area

    Args:
        layout: Layout object

    Returns:
        Dict containing spacing and whitespace features
    """
    elements = layout.elements

    if len(elements) < 2:
        return {
            "avg_spacing": 0.0,
            "min_spacing": 0.0,
            "spacing_consistency": 1.0,
            "whitespace_ratio": 1.0,
        }

    # Calculate pairwise distances
    distances = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            elem1, elem2 = elements[i], elements[j]

            # Rectangle boundaries
            left1, right1 = elem1.x, elem1.x + elem1.width
            top1, bottom1 = elem1.y, elem1.y + elem1.height
            left2, right2 = elem2.x, elem2.x + elem2.width
            top2, bottom2 = elem2.y, elem2.y + elem2.height

            # Calculate gaps
            h_gap = max(0, max(left1, left2) - min(right1, right2))
            v_gap = max(0, max(top1, top2) - min(bottom1, bottom2))

            # Minimum gap (closest approach)
            min_gap = min(h_gap, v_gap) if h_gap > 0 or v_gap > 0 else 0
            distances.append(min_gap)

    if not distances:
        return {
            "avg_spacing": 0.0,
            "min_spacing": 0.0,
            "spacing_consistency": 1.0,
            "whitespace_ratio": 1.0,
        }

    # Spacing metrics
    avg_spacing = float(np.mean(distances))
    min_spacing = float(np.min(distances))
    spacing_std = float(np.std(distances))

    # Consistency score (lower variance = higher consistency)
    spacing_consistency = 1.0 / (1.0 + spacing_std / (avg_spacing + 1e-6))

    # Whitespace ratio
    used_area = sum(elem.area for elem in elements)
    canvas_area = layout.width * layout.height
    whitespace_ratio = (canvas_area - used_area) / canvas_area if canvas_area > 0 else 0.0

    return {
        "avg_spacing": avg_spacing,
        "min_spacing": min_spacing,
        "spacing_consistency": spacing_consistency,
        "whitespace_ratio": max(0.0, whitespace_ratio),
    }


def calculate_hierarchy_features(layout: Layout) -> Dict[str, float]:
    """
    Calculate visual hierarchy and size relationship features.

    Mathematical basis:
    - Size Hierarchy = largest_area / second_largest_area (normalized)
    - Element Ratio = largest_area / total_area
    - Size Variance = σ(areas) / μ(areas) (coefficient of variation)

    Args:
        layout: Layout object

    Returns:
        Dict containing hierarchy and size-based features
    """
    elements = layout.elements

    if not elements:
        return {
            "size_hierarchy": 0.0,
            "largest_element_ratio": 0.0,
            "size_variance": 0.0,
        }

    areas = [elem.area for elem in elements]

    if len(areas) == 1:
        return {
            "size_hierarchy": 1.0,
            "largest_element_ratio": 1.0,
            "size_variance": 0.0,
        }

    # Size hierarchy calculation
    sorted_areas = sorted(areas, reverse=True)
    if len(sorted_areas) > 1 and sorted_areas[1] > 0:
        size_hierarchy = sorted_areas[0] / sorted_areas[1]
        # Normalize to [0,1] where 3x difference = perfect hierarchy
        size_hierarchy = min(size_hierarchy / 3.0, 1.0)
    else:
        size_hierarchy = 1.0

    # Largest element ratio
    total_area = sum(areas)
    largest_element_ratio = sorted_areas[0] / total_area if total_area > 0 else 0.0

    # Size variance (coefficient of variation)
    area_mean = float(np.mean(areas))
    size_variance = float(np.std(areas)) / (area_mean + 1e-6)

    return {
        "size_hierarchy": size_hierarchy,
        "largest_element_ratio": largest_element_ratio,
        "size_variance": size_variance,
    }


def calculate_flow_features(layout: Layout) -> Dict[str, float]:
    """
    Calculate reading flow and scanning pattern features.

    Mathematical basis:
    - Reading Flow = 1 - (order_violations / total_transitions)
    - Top-left Weight = top_left_area / total_area
    - Scanning Pattern = 1 - σ(zone_areas) / μ(zone_areas)

    Args:
        layout: Layout object

    Returns:
        Dict containing flow and scanning features
    """
    elements = layout.elements

    if not elements:
        return {
            "reading_flow_score": 1.0,
            "top_left_weight": 0.0,
            "scanning_pattern": 0.0,
        }

    canvas_width, canvas_height = layout.width, layout.height

    # Top-left weight (F-pattern reading)
    top_left_area = 0.0
    total_area = sum(elem.area for elem in elements)

    for elem in elements:
        elem_center_x = elem.x + elem.width / 2
        elem_center_y = elem.y + elem.height / 2

        # Check if element is in top-left region (30% of canvas)
        if elem_center_x < canvas_width * 0.3 and elem_center_y < canvas_height * 0.3:
            top_left_area += elem.area

    top_left_weight = top_left_area / total_area if total_area > 0 else 0.0

    # Reading flow analysis
    if len(elements) < 2:
        reading_flow_score = 1.0
    else:
        # Sort elements by natural reading order (top-to-bottom, left-to-right)
        element_positions = [(elem.x + elem.width / 2, elem.y + elem.height / 2, elem.area) for elem in elements]
        sorted_by_reading = sorted(element_positions, key=lambda x: (x[1], x[0]))

        # Count reading order violations
        order_violations = 0
        for i in range(len(sorted_by_reading) - 1):
            curr_x, curr_y, _ = sorted_by_reading[i]
            next_x, next_y, _ = sorted_by_reading[i + 1]

            # Violation: next element is significantly left and not much below
            if next_x < curr_x - 50 and next_y < curr_y + 50:
                order_violations += 1

        reading_flow_score = 1.0 - (order_violations / (len(sorted_by_reading) - 1))

    # Scanning pattern (distribution across zones)
    zones = {"top": 0.0, "middle": 0.0, "bottom": 0.0}
    for elem in elements:
        elem_center_y = elem.y + elem.height / 2
        rel_y = elem_center_y / canvas_height if canvas_height > 0 else 0.5

        if rel_y < 0.33:
            zones["top"] += elem.area
        elif rel_y < 0.67:
            zones["middle"] += elem.area
        else:
            zones["bottom"] += elem.area

    zone_areas = list(zones.values())
    if len(zone_areas) > 0 and sum(zone_areas) > 0:
        zone_mean = np.mean(zone_areas)
        scanning_pattern = float(1.0 - (np.std(zone_areas) / (zone_mean + 1e-6)))
    else:
        scanning_pattern = 0.0

    return {
        "reading_flow_score": max(0.0, reading_flow_score),
        "top_left_weight": top_left_weight,
        "scanning_pattern": max(0.0, float(scanning_pattern)),
    }


def extract_all_features(layout: Layout) -> Dict[str, Union[str, float]]:
    """
    Extract comprehensive geometric features for a layout.

    This function combines all feature categories into a single feature vector
    suitable for machine learning algorithms.

    Args:
        layout: Layout object to analyze

    Returns:
        Dict containing all geometric features with descriptive names
    """
    try:
        features: Dict[str, Union[str, float]] = {"id": layout.id}

        # Combine all feature categories
        features.update(calculate_basic_features(layout))
        features.update(calculate_spatial_features(layout))
        features.update(calculate_alignment_features(layout))
        features.update(calculate_balance_features(layout))
        features.update(calculate_spacing_features(layout))
        features.update(calculate_hierarchy_features(layout))
        features.update(calculate_flow_features(layout))

        # Derived composite features
        features["elements_per_area"] = float(features["element_count"]) / (layout.width * layout.height / 10000)
        features["visual_complexity"] = float(features["element_count"]) * float(features["size_variance"])

        logger.debug(f"Extracted {len(features)} features for layout {layout.id}")

        return features

    except Exception as e:
        logger.error(f"Error extracting features for layout {layout.id}: {e}")
        raise
