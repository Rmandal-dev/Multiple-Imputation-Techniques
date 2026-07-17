import pandas as pd
import numpy as np
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from _01_preprocessing import load_and_split_data, calculate_metrics, get_bootstrap_ci, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, SEED
from _02_models import optimize_model_hyperparameters
from _03_all_encoders import get_encoders
from _04_ensemble import PairwiseOrTripleEnsemble, StackingEnsemble
from _05_statistics import validate_statistical_significance

warnings.filterwarnings('ignore')

def main():
    print("🚀 Initializing Live Optuna Tuning Experiment Aligned with Methodology...")
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    encoders = get_encoders(CATEGORICAL_COLUMNS, SEED)
    model_names = ["Linear Regression", "SVR", "Random Forest", "Extra Trees", "XGBoost", "LightGBM", "CatBoost"]
    
    all_results = []
    absolute_errors_store = {}
    
    for enc_name, encoder_template in encoders.items():
        print(f"\nEvaluating Category Encoder: {enc_name}")
        print("-" * 75)
        
        fitted_base_pipelines = {}
        active_tuned_models = {}
        
        encoder = clone(encoder_template)
        X_train_cat = encoder.fit_transform(X_train[CATEGORICAL_COLUMNS], y_train)
        X_test_cat = encoder.transform(X_test[CATEGORICAL_COLUMNS])
        
        X_train_processed = pd.concat([X_train[NUMERICAL_COLUMNS].reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
        X_test_processed = pd.concat([X_test[NUMERICAL_COLUMNS].reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)
        
        # 1. LIVE OPTUNA HYPERPARAMETER TUNING LOOP
        for model_name in model_names:
            print(f"  -> Tuning & Training Model via Optuna (10-Fold CV): {model_name}")
            
            tuned_model, requires_scaling = optimize_model_hyperparameters(model_name, X_train_processed, y_train, SEED)
            active_tuned_models[model_name] = tuned_model
            
            steps = []
            if requires_scaling:
                steps.append(('scaler', StandardScaler()))
            steps.append(('model', tuned_model))
            
            pipeline = Pipeline(steps)
            pipeline.fit(X_train_processed, y_train)
            fitted_base_pipelines[model_name] = pipeline
            
            preds = pipeline.predict(X_test_processed)
            
            # Create a robust unique tracking identifier key
            unique_key = f"{enc_name}_{model_name}"
            absolute_errors_store[unique_key] = np.abs(y_test - preds)
            
            metrics = calculate_metrics(y_test, preds)
            mae_low, mae_high = get_bootstrap_ci(y_test, preds, 'MAE')
            
            all_results.append({
                'Encoder': enc_name, 'Model': model_name, 'Model_Key': unique_key,
                'MAE': metrics['MAE'], 'MAE_CI_Lower': mae_low, 'MAE_CI_Upper': mae_high,
                'RMSE': metrics['RMSE'], 'R2': metrics['R2'],
                'MMRE': metrics['MMRE'], 'MdMRE': metrics['MdMRE'], 'Pred_025': metrics['Pred_025'],
                'Architecture_Type': 'Live Optuna Tuned'
            })
            
        # 2. ENSEMBLE SPECIFICATIONS MATRIX
        ensemble_specs = {
            "Pairwise Ensemble: LR+RFR": ["Linear Regression", "Random Forest"],
            "Pairwise Ensemble: LR+SVR": ["Linear Regression", "SVR"],
            "Pairwise Ensemble: RFR+SVR": ["Random Forest", "SVR"],
            "Triple Ensemble: LR+RFR+SVR": ["Linear Regression", "Random Forest", "SVR"],
            "Pairwise Ensemble: XGBoost+LightGBM": ["XGBoost", "LightGBM"],
            "Pairwise Ensemble: XGBoost+CatBoost": ["XGBoost", "CatBoost"],
            "Pairwise Ensemble: LightGBM+CatBoost": ["LightGBM", "CatBoost"],
            "Pairwise Ensemble: Random Forest+Extra Trees": ["Random Forest", "Extra Trees"],
            "Pairwise Ensemble: RFR+XGBoost": ["Random Forest", "XGBoost"],
            "Pairwise Ensemble: Extra Trees+XGBoost": ["Extra Trees", "XGBoost"],
            "Triple Ensemble: XGBoost+LightGBM+CatBoost": ["XGBoost", "LightGBM", "CatBoost"],
            "Triple Ensemble: RFR+XGBoost+LightGBM": ["Random Forest", "XGBoost", "LightGBM"],
            "Triple Ensemble: ETR+XGBoost+CatBoost": ["Extra Trees", "XGBoost", "CatBoost"],
            "Triple Ensemble: RFR+ETR+XGBoost": ["Random Forest", "Extra Trees", "XGBoost"]
        }
        
        for ens_label, model_keys in ensemble_specs.items():
            avg_ens = PairwiseOrTripleEnsemble(fitted_base_pipelines, model_keys)
            avg_preds = avg_ens.predict(X_test_processed)
            
            unique_key = f"{enc_name}_{ens_label}"
            absolute_errors_store[unique_key] = np.abs(y_test - avg_preds)
            
            ens_metrics = calculate_metrics(y_test, avg_preds)
            ens_low, ens_high = get_bootstrap_ci(y_test, avg_preds, 'MAE')
            
            all_results.append({
                'Encoder': enc_name, 'Model': ens_label, 'Model_Key': unique_key,
                'MAE': ens_metrics['MAE'], 'MAE_CI_Lower': ens_low, 'MAE_CI_Upper': ens_high,
                'RMSE': ens_metrics['RMSE'], 'R2': ens_metrics['R2'],
                'MMRE': ens_metrics['MMRE'], 'MdMRE': ens_metrics['MdMRE'], 'Pred_025': ens_metrics['Pred_025'],
                'Architecture_Type': 'Averaged Ensemble'
            })

        # 3. STACKING ENSEMBLE ARCHITECTURE
        print(f"  -> Building 5-Fold Stacking Architecture using Best Models...")
        stack_model_name = 'Stacking Ensemble (RFR+XGB+LGB)'
        stack_ens = StackingEnsemble(active_tuned_models).fit(X_train_processed, y_train)
        stack_preds = stack_ens.predict(X_test_processed)
        
        unique_key = f"{enc_name}_{stack_model_name}"
        absolute_errors_store[unique_key] = np.abs(y_test - stack_preds)
        
        stack_metrics = calculate_metrics(y_test, stack_preds)
        stack_low, stack_high = get_bootstrap_ci(y_test, stack_preds, 'MAE')
        
        all_results.append({
            'Encoder': enc_name, 'Model': stack_model_name, 'Model_Key': unique_key,
            'MAE': stack_metrics['MAE'], 'MAE_CI_Lower': stack_low, 'MAE_CI_Upper': stack_high,
            'RMSE': stack_metrics['RMSE'], 'R2': stack_metrics['R2'],
            'MMRE': stack_metrics['MMRE'], 'MdMRE': stack_metrics['MdMRE'], 'Pred_025': stack_metrics['Pred_025'],
            'Architecture_Type': 'Meta-Learning Stacking'
        })

    # Convert results array to DataFrame and sort by global best performance metric
    results_df = pd.DataFrame(all_results).sort_values(by='MAE', ascending=True).reset_index(drop=True)

    # 4. ROBUST STATISTICAL SIGNIFICANCE LOOP (Fixes KeyErrors permanently)
    print("\nMapping statistical significance benchmarks against overall best configuration...")
    best_model_key = results_df.iloc[0]['Model_Key']
    global_best_errors = absolute_errors_store[best_model_key]

    p_values, cohens_d_stats = [], []
    for _, row in results_df.iterrows():
        # Pulling directly from saved explicit row value key avoids naming mismatches entirely
        curr_key = row['Model_Key']
        p_val, effect_size = validate_statistical_significance(global_best_errors, absolute_errors_store[curr_key])
        p_values.append(p_val)
        cohens_d_stats.append(effect_size)
        
    results_df['Wilcoxon_p_value'] = p_values
    results_df['Cohens_d'] = cohens_d_stats

    # Clean up tracking column prior to dashboard generation and file saving
    results_df.drop(columns=['Model_Key'], inplace=True)
    results_df.to_csv('live_optuna_optimized_results.csv', index=False)
    
    print("\n" + "=" * 115)
    print("🏆 FINAL PERFORMANCE SCORES WITH LIVE OPTUNA DEPLOYED (Sorted by MAE)")
    print("=" * 115)
    print(results_df[['Encoder', 'Model', 'MAE', 'MAE_CI_Lower', 'MAE_CI_Upper', 'RMSE', 'Wilcoxon_p_value']].to_string(index=False))

if __name__ == '__main__':
    main()