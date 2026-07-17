import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================================
# Load Dataset
# ============================================================================
data = pd.read_csv("cart.csv")

# ==============================cart==============================================
# Basic COCOMO 81 (Embedded Mode)
# ============================================================================
a = 3.6
b = 1.20

# Lines of Code (LOC)
loc = data["Lines of Code"]

# Actual effort
actual_effort = data["Summary Work Effort"]

# Predicted effort using Basic COCOMO 81
predicted_effort = a * (loc / 1000) ** b

# ============================================================================
# Evaluation Metrics
# ============================================================================

# Mean Absolute Error
mae = mean_absolute_error(actual_effort, predicted_effort)

# Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(actual_effort, predicted_effort))

# Coefficient of Determination (R²)
r2 = r2_score(actual_effort, predicted_effort)

# ============================================================================
# Results
# ============================================================================

print("="*60)
print("Basic COCOMO 81 Performance")
print("="*60)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")