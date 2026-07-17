import optuna
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_model_hyperparameters(model_name, X_train, y_train, seed=42):
    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    
    if model_name == "Linear Regression":
        return LinearRegression(), True

    def objective(trial):
        if model_name == "Random Forest":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
                max_depth=trial.suggest_int("max_depth", 5, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                random_state=seed, n_jobs=-1
            )
            scaled = False
            
        elif model_name == "Extra Trees":
            model = ExtraTreesRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
                max_depth=trial.suggest_int("max_depth", 5, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                random_state=seed, n_jobs=-1
            )
            scaled = False

        elif model_name == "XGBoost":
            model = xgb.XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                random_state=seed, n_jobs=-1
            )
            scaled = False

        elif model_name == "LightGBM":
            model = lgb.LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 10, 100),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),  # FIX: changed parameter label string 'reg_float' to 'reg_alpha'
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                random_state=seed, n_jobs=-1, verbose=-1
            )
            scaled = False

        elif model_name == "CatBoost":
            model = cb.CatBoostRegressor(
                iterations=trial.suggest_int("iterations", 100, 500, step=50),
                depth=trial.suggest_int("depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                random_state=seed, thread_count=-1, verbose=0
            )
            scaled = False

        elif model_name == "SVR":
            model = SVR(
                kernel='rbf',
                C=trial.suggest_float("C", 0.1, 100.0, log=True),
                gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                epsilon=trial.suggest_float("epsilon", 0.001, 1.0, log=True)
            )
            scaled = True
            
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    
    if model_name == "Random Forest":
        return RandomForestRegressor(**best_params, random_state=seed, n_jobs=-1), False
    elif model_name == "Extra Trees":
        return ExtraTreesRegressor(**best_params, random_state=seed, n_jobs=-1), False
    elif model_name == "XGBoost":
        return xgb.XGBRegressor(**best_params, random_state=seed, n_jobs=-1), False
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(**best_params, random_state=seed, n_jobs=-1, verbose=-1), False
    elif model_name == "CatBoost":
        return cb.CatBoostRegressor(**best_params, random_state=seed, thread_count=-1, verbose=0), False
    elif model_name == "SVR":
        return SVR(kernel='rbf', **best_params), True