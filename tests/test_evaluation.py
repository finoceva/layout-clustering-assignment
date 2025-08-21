"""
Tests for clustering evaluation functionality.
"""

from typing import List

import numpy as np
import pytest
from utils.evaluation import (
    calculate_combined_score,
    calculate_quality_purity,
    calculate_silhouette_score,
    evaluate_clustering,
)


class TestSilhouetteScore:
    """Test silhouette score calculations."""
    
    def test_silhouette_score_valid_clustering(self):
        """Test silhouette score with valid clustering."""
        # Create simple 2D data with clear clusters
        X = np.array([
            [0, 0], [1, 1], [0, 1], [1, 0],  # Cluster 0
            [5, 5], [6, 6], [5, 6], [6, 5],  # Cluster 1
        ])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        score = calculate_silhouette_score(X, labels)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Should be good clustering
        assert score > 0.5  # Clear separation should give good score
    
    def test_silhouette_score_single_cluster(self):
        """Test silhouette score with single cluster."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])
        
        score = calculate_silhouette_score(X, labels)
        
        assert score == 0.0  # Single cluster should return 0
    
    def test_silhouette_score_with_noise(self):
        """Test silhouette score with noise points."""
        X = np.array([
            [0, 0], [1, 1],  # Cluster 0
            [5, 5], [6, 6],  # Cluster 1
            [10, 10]         # Noise
        ])
        labels = np.array([0, 0, 1, 1, -1])
        
        score = calculate_silhouette_score(X, labels)
        
        assert isinstance(score, float)
        assert score >= 0  # Should handle noise gracefully
    
    def test_silhouette_score_empty_input(self):
        """Test silhouette score with empty input."""
        X = np.array([]).reshape(0, 2)
        labels = np.array([])
        
        score = calculate_silhouette_score(X, labels)
        
        assert score == 0.0


class TestQualityPurity:
    """Test quality purity calculations."""
    
    def test_quality_purity_perfect_clustering(self):
        """Test quality purity with perfect clustering."""
        cluster_labels = np.array([0, 0, 1, 1, 2, 2])
        quality_labels = ['pass', 'pass', 'fail', 'fail', 'pass', 'pass']
        
        purity = calculate_quality_purity(cluster_labels, quality_labels)
        
        assert isinstance(purity, float)
        assert purity == 1.0  # Perfect separation
    
    def test_quality_purity_random_clustering(self):
        """Test quality purity with random clustering."""
        cluster_labels = np.array([0, 1, 0, 1, 0, 1])
        quality_labels = ['pass', 'pass', 'fail', 'fail', 'pass', 'fail']
        
        purity = calculate_quality_purity(cluster_labels, quality_labels)
        
        assert isinstance(purity, float)
        assert 0 <= purity <= 1
        assert purity < 1.0  # Should not be perfect
    
    def test_quality_purity_single_cluster(self):
        """Test quality purity with single cluster."""
        cluster_labels = np.array([0, 0, 0, 0])
        quality_labels = ['pass', 'pass', 'fail', 'fail']
        
        purity = calculate_quality_purity(cluster_labels, quality_labels)
        
        assert purity == 0.5  # 50% purity (2 pass, 2 fail)
    
    def test_quality_purity_with_noise(self):
        """Test quality purity with noise cluster."""
        cluster_labels = np.array([0, 0, 1, 1, -1, -1])
        quality_labels = ['pass', 'pass', 'fail', 'fail', 'pass', 'fail']
        
        purity = calculate_quality_purity(cluster_labels, quality_labels)
        
        assert isinstance(purity, float)
        assert 0 <= purity <= 1
        # Noise points should be ignored, so purity should be 1.0 
        # (perfect clusters + ignored noise)
        assert purity == 1.0
    
    def test_quality_purity_empty_input(self):
        """Test quality purity with empty input."""
        cluster_labels = np.array([])
        quality_labels = []
        
        purity = calculate_quality_purity(cluster_labels, quality_labels)
        
        assert purity == 0.0
    
    def test_quality_purity_mismatched_lengths(self):
        """Test quality purity with mismatched input lengths."""
        cluster_labels = np.array([0, 1, 2])
        quality_labels = ['pass', 'fail']  # Wrong length
        
        with pytest.raises(ValueError):
            calculate_quality_purity(cluster_labels, quality_labels)


class TestCombinedScore:
    """Test combined score calculations."""
    
    def test_combined_score_equal_weights(self):
        """Test combined score with equal weights."""
        silhouette = 0.8
        quality_purity = 0.6
        
        combined = calculate_combined_score(silhouette, quality_purity)
        
        expected = (0.8 + 0.6) / 2
        assert combined == expected
    
    def test_combined_score_custom_weights(self):
        """Test combined score with custom weights."""
        silhouette = 0.8
        quality_purity = 0.6
        silhouette_weight = 0.7
        
        combined = calculate_combined_score(silhouette, quality_purity, silhouette_weight)
        
        expected = 0.8 * 0.7 + 0.6 * 0.3
        assert combined == expected
    
    def test_combined_score_boundary_values(self):
        """Test combined score with boundary values."""
        # Test with perfect scores
        combined_perfect = calculate_combined_score(1.0, 1.0)
        assert combined_perfect == 1.0
        
        # Test with zero scores
        combined_zero = calculate_combined_score(0.0, 0.0)
        assert combined_zero == 0.0
        
        # Test with mixed scores
        combined_mixed = calculate_combined_score(1.0, 0.0)
        assert combined_mixed == 0.5


class TestEvaluateClustering:
    """Test comprehensive clustering evaluation."""
    
    def test_evaluate_clustering_complete(self):
        """Test complete clustering evaluation."""
        # Create test data
        X = np.array([
            [0, 0], [1, 1], [0, 1], [1, 0],  # Cluster 0
            [5, 5], [6, 6], [5, 6], [6, 5],  # Cluster 1
        ])
        cluster_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        quality_labels = ['pass', 'pass', 'pass', 'pass', 'fail', 'fail', 'fail', 'fail']
        
        results = evaluate_clustering(X, cluster_labels, quality_labels)
        
        # Check all required metrics are present
        required_keys = ['silhouette_score', 'quality_purity', 'combined_score', 'n_clusters']
        for key in required_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
        
        # Check metric ranges
        assert 0 <= results['silhouette_score'] <= 1
        assert 0 <= results['quality_purity'] <= 1
        assert 0 <= results['combined_score'] <= 1
        assert results['n_clusters'] == 2
        
        # Should have good scores for this clear separation
        assert results['silhouette_score'] > 0.5
        assert results['quality_purity'] == 1.0  # Perfect quality separation
    
    def test_evaluate_clustering_poor_clustering(self):
        """Test evaluation with poor clustering."""
        # Create overlapping clusters
        X = np.array([
            [0, 0], [1, 1], [2, 2], [3, 3],
            [0.5, 0.5], [1.5, 1.5], [2.5, 2.5], [3.5, 3.5]
        ])
        cluster_labels = np.array([0, 1, 0, 1, 1, 0, 1, 0])  # Mixed clustering
        quality_labels = ['pass', 'pass', 'pass', 'pass', 'fail', 'fail', 'fail', 'fail']
        
        results = evaluate_clustering(X, cluster_labels, quality_labels)
        
        # Should have lower scores for poor clustering
        assert results['silhouette_score'] < 0.5
        assert results['quality_purity'] < 1.0
        assert results['combined_score'] < 0.7
    
    def test_evaluate_clustering_with_noise(self):
        """Test evaluation with noise cluster."""
        X = np.array([
            [0, 0], [1, 1],  # Cluster 0
            [5, 5], [6, 6],  # Cluster 1
            [10, 10], [20, 20]  # Noise
        ])
        cluster_labels = np.array([0, 0, 1, 1, -1, -1])
        quality_labels = ['pass', 'pass', 'fail', 'fail', 'pass', 'fail']
        
        results = evaluate_clustering(X, cluster_labels, quality_labels)
        
        # Should handle noise appropriately
        assert results['n_clusters'] == 2  # Should not count noise cluster
        assert isinstance(results['silhouette_score'], float)
        assert isinstance(results['quality_purity'], float)
    
    def test_evaluate_clustering_edge_cases(self):
        """Test evaluation edge cases."""
        # Single point
        X = np.array([[0, 0]])
        cluster_labels = np.array([0])
        quality_labels = ['pass']
        
        results = evaluate_clustering(X, cluster_labels, quality_labels)
        
        assert results['silhouette_score'] == 0.0
        assert results['quality_purity'] == 1.0  # Single point is pure
        assert results['n_clusters'] == 1
        
        # Empty clustering
        X = np.array([]).reshape(0, 2)
        cluster_labels = np.array([])
        quality_labels = []
        
        results = evaluate_clustering(X, cluster_labels, quality_labels)
        
        assert results['silhouette_score'] == 0.0
        assert results['quality_purity'] == 0.0
        assert results['n_clusters'] == 0
