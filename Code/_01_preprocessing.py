import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample

# Configuration & Constants
SEED = 42
TARGET_COLUMN = 'Summary Work Effort'

NUMERICAL_COLUMNS = [
    "Functional Size", "Adjusted Function Points", "Value Adjustment Factor",
    "Defect Density", "Speed of Delivery", "Project Elapsed Time",
    "Project Inactive Time", "Total Defects Delivered", "Resource Level",
    "Max Team Size", "Ratio of Project Work Effort to Non-Project Activity",
    "Lines of Code"
]

CATEGORICAL_COLUMNS = [
    "Industry Sector", "Organisation Type", "Application Group", "Application Type",
    "Development Type", "Development Platform", "Language Type",
    "Primary Programming Language", "Count Approach", "Relative Size",
    "Project Activity Scope", "Software Process CMM", "Software Process CMMI",
    "Software Process ISO", "Software Process Other", "Architecture",
    "Development Methodologies", "Development Techniques", "Planning Documents",
    "Specification Documents", "Specification Techniques", "Design Documents",
    "Design Techniques", "Build Products", "Build Activity", "Test Documents",
    "Test Activity", "Implement Documents", "Functional Sizing Technique",
    "FP Standards All", "Percentage of Uncollected Work Effort"
]

def load_and_split_data(filepath='mean.csv'):
    """Loads dataset and performs clean split to prevent data leakage."""
    data = pd.read_csv(filepath)
    X = data[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = data[TARGET_COLUMN]
    
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype(str)
        
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def calculate_metrics(y_true, y_pred):
    """Calculates all domain-required Software Engineering estimation metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mre = np.abs((y_true - y_pred) / (y_true + 1e-10))
    mmre = np.mean(mre)
    mdmre = np.median(mre)
    pred_025 = np.sum(mre <= 0.25) / len(mre) * 100
    
    return {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'MMRE': mmre, 'MdMRE': mdmre, 'Pred_025': pred_025
    }

def get_bootstrap_ci(y_true, y_pred, metric_name='MAE', n_iterations=1000):
    """Calculates 95% Confidence Intervals via bootstrapping for a metric."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    stats = []
    
    for _ in range(n_iterations):
        idx = resample(np.arange(len(y_true)), random_state=None)
        sample_true = y_true[idx]
        sample_pred = y_pred[idx]
        
        metrics = calculate_metrics(sample_true, sample_pred)
        stats.append(metrics[metric_name])
        
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return lower, upper