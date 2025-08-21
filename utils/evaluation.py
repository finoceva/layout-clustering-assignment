"""
Clustering Evaluation Utilities

This module provides standardized evaluation metrics for clustering results,
including both geometric coherence (silhouette) and domain-specific quality metrics.
"""

from typing import List

import numpy as np
from loguru import logger
from sklearn.metrics import silhouette_score


def calculate_silhouette_score(X: np.ndarray, cluster_labels: np.ndarray) -> float:
    """
    Calculate silhouette score for clustering quality.
    
    The silhouette score measures how similar an object is to its own cluster
    compared to other clusters. Score ranges from -1 to +1:
    - +1: Perfect clustering (well-matched to cluster, poorly-matched to neighbors)
    - 0: Overlapping clusters  
    - -1: Wrong cluster assignment
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        cluster_labels: Cluster assignments for each sample
        
    Returns:
        float: Average silhouette score across all samples
        
    Raises:
        ValueError: If fewer than 2 clusters or insufficient samples
    """
    try:
        unique_labels = np.unique(cluster_labels)
        # Remove noise cluster (-1) if present
        valid_labels = unique_labels[unique_labels != -1]
        
        if len(valid_labels) < 2:
            logger.warning("Silhouette score requires at least 2 clusters")
            return 0.0
            
        # Filter out noise points for silhouette calculation
        mask = cluster_labels != -1
        if np.sum(mask) < 2:
            logger.warning("Insufficient non-noise points for silhouette calculation")
            return 0.0
            
        return silhouette_score(X[mask], cluster_labels[mask])
        
    except Exception as e:
        logger.error(f"Error calculating silhouette score: {e}")
        return 0.0


def calculate_quality_purity(cluster_labels: np.ndarray, quality_labels: List[str]) -> float:
    """
    Calculate quality purity score for clustering evaluation.
    
    Quality purity measures how well clusters align with ground truth quality labels.
    It calculates the proportion of samples that belong to the majority quality class
    within their assigned cluster.
    
    Mathematical definition:
    Purity = (1/N) * Σ(max_j |C_i ∩ Q_j|) for all clusters i
    
    Where:
    - N = total number of samples
    - C_i = set of samples in cluster i  
    - Q_j = set of samples with quality label j
    - |C_i ∩ Q_j| = number of samples in cluster i with quality j
    
    Args:
        cluster_labels: Cluster assignments (shape: n_samples)
        quality_labels: Ground truth quality labels ('pass' or 'fail')
        
    Returns:
        float: Purity score between 0.0 and 1.0
               1.0 = perfect alignment (each cluster is pure)
               0.5 = random alignment  
               0.0 = worst possible alignment
    """
    if len(cluster_labels) != len(quality_labels):
        raise ValueError("cluster_labels and quality_labels must have same length")
    
    if len(cluster_labels) == 0:
        return 0.0
    
    # Convert quality labels to binary (1=pass, 0=fail)
    quality_binary = np.array([1 if q == 'pass' else 0 for q in quality_labels])
    
    total_samples = len(cluster_labels)
    correctly_grouped = 0
    
    # Calculate purity for each cluster
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        # Skip noise cluster if using HDBSCAN
        if cluster_id == -1:
            continue
            
        # Get quality labels for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_qualities = quality_binary[cluster_mask]
        
        if len(cluster_qualities) == 0:
            continue
            
        # Count majority class in this cluster
        pass_count = np.sum(cluster_qualities)
        fail_count = len(cluster_qualities) - pass_count
        majority_count = max(pass_count, fail_count)
        
        correctly_grouped += majority_count
    
    purity = correctly_grouped / total_samples
    
    logger.debug(f"Quality purity calculation: {correctly_grouped}/{total_samples} = {purity:.3f}")
    
    return purity


def calculate_combined_score(silhouette: float, quality_purity: float, 
                           silhouette_weight: float = 0.5) -> float:
    """
    Calculate combined clustering evaluation score.
    
    Combines geometric coherence (silhouette) with domain-specific quality alignment.
    
    Args:
        silhouette: Silhouette score (geometric coherence)
        quality_purity: Quality purity score (quality alignment)
        silhouette_weight: Weight for silhouette score (default: 0.5)
        
    Returns:
        float: Combined score between 0.0 and 1.0
    """
    quality_weight = 1.0 - silhouette_weight
    return silhouette * silhouette_weight + quality_purity * quality_weight


def evaluate_clustering(X: np.ndarray, cluster_labels: np.ndarray, 
                       quality_labels: List[str]) -> dict:
    """
    Comprehensive clustering evaluation.
    
    Args:
        X: Feature matrix
        cluster_labels: Cluster assignments  
        quality_labels: Ground truth quality labels
        
    Returns:
        dict: Evaluation metrics including silhouette, quality_purity, combined_score
    """
    silhouette = calculate_silhouette_score(X, cluster_labels)
    quality_purity = calculate_quality_purity(cluster_labels, quality_labels)
    combined = calculate_combined_score(silhouette, quality_purity)
    
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    
    return {
        'silhouette_score': silhouette,
        'quality_purity': quality_purity,
        'combined_score': combined,
        'n_clusters': n_clusters
    }
