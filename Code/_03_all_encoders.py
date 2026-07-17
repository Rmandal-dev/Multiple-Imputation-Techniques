import category_encoders as ce

def get_encoders(categorical_columns, seed=42):
    """Instantiates the six specified categorical-to-numeric mapping pipelines."""
    return {
        "OrdinalEncoder": ce.OrdinalEncoder(cols=categorical_columns),
        "TargetEncoder": ce.TargetEncoder(cols=categorical_columns),
        "MEstimateEncoder": ce.MEstimateEncoder(cols=categorical_columns),
        "JamesSteinEncoder": ce.JamesSteinEncoder(cols=categorical_columns),
        "LeaveOneOutEncoder": ce.LeaveOneOutEncoder(cols=categorical_columns, random_state=seed),
        "CatBoostEncoder": ce.CatBoostEncoder(cols=categorical_columns, random_state=seed)
    }