import numpy as np
from scipy.stats import wilcoxon

def compute_cohens_d(x, y):
    """Computes effect size using Cohen's d statistic."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0
    return (np.mean(x) - np.mean(y)) / pooled_std

def validate_statistical_significance(global_best_errors, alternative_errors):
    """Runs a paired Wilcoxon signed-rank test and calculates effect sizing."""
    global_best_errors = np.array(global_best_errors)
    alternative_errors = np.array(alternative_errors)
    
    # Check if the distributions are completely identical
    if np.array_equal(global_best_errors, alternative_errors):
        return 1.0, 0.0
        
    try:
        stat, p_val = wilcoxon(global_best_errors, alternative_errors)
    except ValueError:
        # Fallback if zero differences break the ranks
        p_val = 1.0
        
    effect_size = compute_cohens_d(global_best_errors, alternative_errors)
    return p_val, effect_size