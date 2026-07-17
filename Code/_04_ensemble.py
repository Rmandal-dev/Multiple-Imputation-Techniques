import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

class PairwiseOrTripleEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, fitted_pipelines_dict, model_keys):
        self.fitted_pipelines_dict = fitted_pipelines_dict
        self.model_keys = model_keys
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        preds = np.column_stack([self.fitted_pipelines_dict[key].predict(X) for key in self.model_keys])
        return np.mean(preds, axis=1)

class StackingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_models_dict, meta_estimator=Ridge()):
        self.base_models_dict = base_models_dict
        self.meta_estimator = meta_estimator
        self.base_keys = ["Random Forest", "XGBoost", "LightGBM"]
        
    def fit(self, X, y):
        self.fitted_base_models_ = []
        meta_features = []
        
        for key in self.base_keys:
            base_model = clone(self.base_models_dict[key])
            oof_preds = cross_val_predict(base_model, X, y, cv=5, n_jobs=-1)
            meta_features.append(oof_preds)
            
            base_model.fit(X, y)
            self.fitted_base_models_.append(base_model)
            
        meta_matrix = np.column_stack(meta_features)
        self.meta_estimator_ = clone(self.meta_estimator).fit(meta_matrix, y)
        return self
        
    def predict(self, X):
        meta_features = np.column_stack([model.predict(X) for model in self.fitted_base_models_])
        return self.meta_estimator_.predict(meta_features)