"""
Tests for geometric feature extraction functionality.
"""

import numpy as np
from features.geometric import (
    calculate_alignment_features,
    calculate_balance_features,
    calculate_basic_features,
    calculate_flow_features,
    calculate_hierarchy_features,
    calculate_spacing_features,
    calculate_spatial_features,
    extract_all_features,
)

from core.schemas import Element, Layout


class TestBasicFeatures:
    """Test basic geometric feature calculations."""
    
    def test_basic_features_normal_layout(self, sample_layout: Layout):
        """Test basic features with normal layout."""
        features = calculate_basic_features(sample_layout)
        
        assert 'element_count' in features
        assert 'content_density' in features
        assert 'avg_element_size' in features
        
        assert features['element_count'] == 4
        assert 0 < features['content_density'] < 1
        assert features['avg_element_size'] > 0
    
    def test_basic_features_empty_layout(self, empty_layout: Layout):
        """Test basic features with empty layout."""
        features = calculate_basic_features(empty_layout)
        
        assert features['element_count'] == 0
        assert features['content_density'] == 0.0
        assert features['avg_element_size'] == 0.0


class TestSpatialFeatures:
    """Test spatial distribution feature calculations."""
    
    def test_spatial_features_normal_layout(self, sample_layout: Layout):
        """Test spatial features with normal layout."""
        features = calculate_spatial_features(sample_layout)
        
        required_keys = [
            'spatial_spread_x', 'spatial_spread_y',
            'center_of_mass_x', 'center_of_mass_y',
            'asymmetry_x', 'asymmetry_y'
        ]
        
        for key in required_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
        
        # Center of mass should be normalized [0, 1]
        assert 0 <= features['center_of_mass_x'] <= 1
        assert 0 <= features['center_of_mass_y'] <= 1
    
    def test_spatial_features_empty_layout(self, empty_layout: Layout):
        """Test spatial features with empty layout."""
        features = calculate_spatial_features(empty_layout)
        
        assert features['spatial_spread_x'] == 0.0
        assert features['spatial_spread_y'] == 0.0
        assert features['center_of_mass_x'] == 0.5
        assert features['center_of_mass_y'] == 0.5


class TestAlignmentFeatures:
    """Test alignment and grid feature calculations."""
    
    def test_alignment_features_normal_layout(self, sample_layout: Layout):
        """Test alignment features with normal layout."""
        features = calculate_alignment_features(sample_layout)
        
        required_keys = [
            'edge_alignment_score', 'unique_x_positions',
            'unique_y_positions', 'grid_adherence_score'
        ]
        
        for key in required_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
        
        # Scores should be in valid ranges
        assert 0 <= features['edge_alignment_score'] <= 1
        assert 0 <= features['grid_adherence_score'] <= 1
        assert features['unique_x_positions'] >= 1
        assert features['unique_y_positions'] >= 1
    
    def test_alignment_features_single_element(self):
        """Test alignment features with single element."""
        element = Element(element_class="test", x=100, y=100, width=200, height=100)
        layout = Layout(
            id="single_element",
            width=400,
            height=300,
            group_id="test",
            elements=[element],
            quality="pass"
        )
        
        features = calculate_alignment_features(layout)
        
        # Single element should have perfect alignment
        assert features['edge_alignment_score'] == 1.0
        assert features['unique_x_positions'] == 1
        assert features['unique_y_positions'] == 1


class TestBalanceFeatures:
    """Test balance and weight distribution calculations."""
    
    def test_balance_features_normal_layout(self, sample_layout: Layout):
        """Test balance features with normal layout."""
        features = calculate_balance_features(sample_layout)
        
        required_keys = [
            'balance_score', 'horizontal_balance', 'vertical_balance',
            'weight_center_x', 'weight_center_y',
            'weight_distribution_x', 'weight_distribution_y'
        ]
        
        for key in required_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
        
        # Scores should be in [0, 1] range
        assert 0 <= features['balance_score'] <= 1
        assert 0 <= features['horizontal_balance'] <= 1
        assert 0 <= features['vertical_balance'] <= 1
        
        # Weight centers should be normalized
        assert 0 <= features['weight_center_x'] <= 1
        assert 0 <= features['weight_center_y'] <= 1
    
    def test_balance_features_centered_layout(self):
        """Test balance features with perfectly centered layout."""
        # Create layout with element at center
        element = Element(element_class="center", x=200, y=150, width=200, height=100)
        layout = Layout(
            id="centered",
            width=600,
            height=400,
            group_id="test",
            elements=[element],
            quality="pass"
        )
        
        features = calculate_balance_features(layout)
        
        # Should have good balance scores
        assert features['balance_score'] > 0.8
        assert abs(features['weight_center_x'] - 0.5) < 0.1
        assert abs(features['weight_center_y'] - 0.5) < 0.1


class TestSpacingFeatures:
    """Test spacing and whitespace calculations."""
    
    def test_spacing_features_normal_layout(self, sample_layout: Layout):
        """Test spacing features with normal layout."""
        features = calculate_spacing_features(sample_layout)
        
        required_keys = [
            'avg_spacing', 'min_spacing',
            'spacing_consistency', 'whitespace_ratio'
        ]
        
        for key in required_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
        
        # Spacing should be non-negative
        assert features['avg_spacing'] >= 0
        assert features['min_spacing'] >= 0
        
        # Ratios should be in valid ranges
        assert 0 <= features['spacing_consistency'] <= 1
        assert 0 <= features['whitespace_ratio'] <= 1
    
    def test_spacing_features_single_element(self):
        """Test spacing features with single element."""
        element = Element(element_class="single", x=100, y=100, width=200, height=100)
        layout = Layout(
            id="single",
            width=400,
            height=300,
            group_id="test",
            elements=[element],
            quality="pass"
        )
        
        features = calculate_spacing_features(layout)
        
        # Single element should have default spacing values
        assert features['avg_spacing'] == 0.0
        assert features['min_spacing'] == 0.0
        assert features['spacing_consistency'] == 1.0


class TestHierarchyFeatures:
    """Test hierarchy and size relationship calculations."""
    
    def test_hierarchy_features_normal_layout(self, sample_layout: Layout):
        """Test hierarchy features with normal layout."""
        features = calculate_hierarchy_features(sample_layout)
        
        required_keys = [
            'size_hierarchy', 'largest_element_ratio', 'size_variance'
        ]
        
        for key in required_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
        
        # Scores should be in valid ranges
        assert 0 <= features['size_hierarchy'] <= 1
        assert 0 <= features['largest_element_ratio'] <= 1
        assert features['size_variance'] >= 0
    
    def test_hierarchy_features_uniform_sizes(self):
        """Test hierarchy features with uniform element sizes."""
        elements = [
            Element(element_class=f"elem_{i}", x=i*100, y=100, width=80, height=80)
            for i in range(3)
        ]
        
        layout = Layout(
            id="uniform",
            width=400,
            height=300,
            group_id="test",
            elements=elements,
            quality="pass"
        )
        
        features = calculate_hierarchy_features(layout)
        
        # Uniform sizes should have low hierarchy
        assert features['size_hierarchy'] < 0.5
        assert features['size_variance'] == 0.0


class TestFlowFeatures:
    """Test reading flow and scanning pattern calculations."""
    
    def test_flow_features_normal_layout(self, sample_layout: Layout):
        """Test flow features with normal layout."""
        features = calculate_flow_features(sample_layout)
        
        required_keys = [
            'reading_flow_score', 'top_left_weight', 'scanning_pattern'
        ]
        
        for key in required_keys:
            assert key in features
            assert isinstance(features[key], (int, float))
        
        # Scores should be in valid ranges
        assert 0 <= features['reading_flow_score'] <= 1
        assert 0 <= features['top_left_weight'] <= 1
        assert 0 <= features['scanning_pattern'] <= 1
    
    def test_flow_features_top_left_heavy(self):
        """Test flow features with top-left heavy layout."""
        # Create elements in top-left area
        elements = [
            Element(element_class="header", x=50, y=30, width=200, height=50),
            Element(element_class="subheader", x=50, y=90, width=150, height=30),
        ]
        
        layout = Layout(
            id="top_left",
            width=600,
            height=400,
            group_id="test",
            elements=elements,
            quality="pass"
        )
        
        features = calculate_flow_features(layout)
        
        # Should have high top-left weight
        assert features['top_left_weight'] > 0.5


class TestFeatureIntegration:
    """Test the complete feature extraction pipeline."""
    
    def test_extract_all_features_normal_layout(self, sample_layout: Layout):
        """Test complete feature extraction."""
        features = extract_all_features(sample_layout)
        
        # Should have ID
        assert 'id' in features
        assert features['id'] == sample_layout.id
        
        # Should have features from all categories
        feature_categories = [
            'element_count', 'content_density',  # Basic
            'spatial_spread_x', 'center_of_mass_x',  # Spatial
            'edge_alignment_score', 'grid_adherence_score',  # Alignment
            'balance_score', 'horizontal_balance',  # Balance
            'avg_spacing', 'whitespace_ratio',  # Spacing
            'size_hierarchy', 'largest_element_ratio',  # Hierarchy
            'reading_flow_score', 'top_left_weight',  # Flow
        ]
        
        for category in feature_categories:
            assert category in features
            assert isinstance(features[category], (int, float))
        
        # Should have derived features
        assert 'elements_per_area' in features
        assert 'visual_complexity' in features
    
    def test_extract_all_features_empty_layout(self, empty_layout: Layout):
        """Test feature extraction with empty layout."""
        features = extract_all_features(empty_layout)
        
        # Should handle empty layout gracefully
        assert features['id'] == empty_layout.id
        assert features['element_count'] == 0
        assert features['content_density'] == 0.0
        
        # All features should be numeric
        for key, value in features.items():
            if key != 'id':
                assert isinstance(value, (int, float))
                assert not np.isnan(value), f"Feature {key} is NaN"
    
    def test_feature_extraction_robustness(self):
        """Test feature extraction with various edge cases."""
        # Test with very small layout
        small_layout = Layout(
            id="small",
            width=10,
            height=10,
            group_id="test",
            elements=[Element(element_class="tiny", x=0, y=0, width=5, height=5)],
            quality="pass"
        )
        
        features = extract_all_features(small_layout)
        assert len(features) > 10  # Should extract meaningful features
        
        # Test with zero-area elements
        zero_area_layout = Layout(
            id="zero_area",
            width=100,
            height=100,
            group_id="test",
            elements=[Element(element_class="zero", x=50, y=50, width=0, height=0)],
            quality="pass"
        )
        
        features = extract_all_features(zero_area_layout)
        assert features['element_count'] == 1
        assert features['content_density'] == 0.0
