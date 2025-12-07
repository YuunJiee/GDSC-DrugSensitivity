import os
import argparse
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from optuna.integration import TFKerasPruningCallback

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.data.preprocessing import load_and_preprocess_dl, load_and_preprocess_baseline
from src.models.deep_learning.deep_learning_model import GDSCNeuralNetwork

def objective(trial, X_cell, X_drug, y):
    """
    Optuna objective function for optimizing Deep Learning hyperparameters.
    """
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = 30 # Reduced epochs for faster tuning, rely on early stopping
    
    # Model Architecture Hyperparameters
    hyperparams = {
        'cell_layers': trial.suggest_int('cell_layers', 1, 3),
        'cell_units': [
            trial.suggest_categorical(f'cell_units_l{i}', [64, 128, 256, 512]) 
            for i in range(3) 
        ],
        'cell_dropout': trial.suggest_float('cell_dropout', 0.1, 0.5),
        
        'drug_layers': trial.suggest_int('drug_layers', 1, 3),
        'drug_units': [
            trial.suggest_categorical(f'drug_units_l{i}', [32, 64, 128, 256])
            for i in range(3)
        ],
        'drug_dropout': trial.suggest_float('drug_dropout', 0.1, 0.5),
        
        'fusion_layers': trial.suggest_int('fusion_layers', 1, 3),
        'fusion_units': [
            trial.suggest_categorical(f'fusion_units_l{i}', [64, 128, 256])
            for i in range(3)
        ],
        'fusion_dropout': trial.suggest_float('fusion_dropout', 0.1, 0.5),
    }
    
    # Slice the unit lists to match the number of layers
    hyperparams['cell_units'] = hyperparams['cell_units'][:hyperparams['cell_layers']]
    hyperparams['drug_units'] = hyperparams['drug_units'][:hyperparams['drug_layers']]
    hyperparams['fusion_units'] = hyperparams['fusion_units'][:hyperparams['fusion_layers']]

    # 2. Split Data for Validation (Train/Val split separate from Test)
    # We split 20% from the provided training set for validation during tuning
    X_cell_train, X_cell_val, X_drug_train, X_drug_val, y_train, y_val = train_test_split(
        X_cell, X_drug, y, test_size=0.2, random_state=42
    )

    # 3. Model Building
    model = GDSCNeuralNetwork(
        cell_input_dim=X_cell.shape[1],
        drug_input_dim=X_drug.shape[1],
        learning_rate=learning_rate,
        hyperparams=hyperparams
    )
    
    # 4. Training (GDSCNeuralNetwork handles scaling internally)
    history = model.fit(
        [X_cell_train, X_drug_train], y_train,
        X_val_list=[X_cell_val, X_drug_val],
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Get best validation loss (min val_loss)
    val_loss = min(history.history['val_loss'])
    
    return val_loss

def objective_rf(trial, X, y):
    """
    Optuna objective for Random Forest.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'n_jobs': -1,
        'random_state': 42
    }
    
    # 3-Fold CV for speed
    from sklearn.model_selection import KFold, cross_val_score
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    model = RandomForestRegressor(**param)
    
    # Use negative MSE as score (maximize it -> minimize MSE)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse = np.sqrt(-scores.mean())
    
    return rmse

def objective_xgb(trial, X, y):
    """
    Optuna objective for XGBoost.
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'tree_method': 'hist',
        'device': 'cuda', # Try GPU if available
        'n_jobs': 1,
        'random_state': 42
    }
    
    from sklearn.model_selection import KFold, cross_val_score
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    model = XGBRegressor(**param)
    
    try:
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=1)
    except Exception:
        # Fallback to CPU if GPU fails
        param['device'] = 'cpu'
        model = XGBRegressor(**param)
        scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

    rmse = np.sqrt(-scores.mean())
    return rmse

def run_optimization(file_path, include_ids=True, n_trials=50, study_name='gdsc_optimization', model_type='dl'):
    """
    Run Optuna optimization.
    :param model_type: 'dl', 'rf', 'xgb'
    """
    print(f"🚀 Starting Optuna Optimization for {model_type.upper()}...")
    print(f"   include_ids: {include_ids}")
    
    # 1. Load Data
    X, y = None, None
    
    if model_type == 'dl':
         X_cell_train, X_drug_train, _, _, y_train, _, _, _ = load_and_preprocess_dl(
            file_path, include_ids=include_ids
         )
         # For DL objective, we pass individual inputs
         if hasattr(X_cell_train, 'values'): X_cell_train = X_cell_train.values
         if hasattr(X_drug_train, 'values'): X_drug_train = X_drug_train.values
         if hasattr(y_train, 'values'): y_train = y_train.values
         
         # DL Objective is special as it takes multiple inputs
         func = lambda trial: objective(trial, X_cell_train, X_drug_train, y_train)
         
    else:
        # For ML models, use the baseline preprocessing (creates a single matrix)
        # Note: load_and_preprocess_baseline performs split. We want full training set for CV.
        # But for consistency with load_and_preprocess_baseline, let's use it and combine train set?
        # Actually load_and_preprocess_baseline does VarianceThreshold which is important.
        X_train, _, y_train, _, _ = load_and_preprocess_baseline(
            file_path, include_ids=include_ids
        )
        X = X_train
        y = y_train
        
        if model_type == 'rf':
            func = lambda trial: objective_rf(trial, X, y)
        elif model_type == 'xgb':
            func = lambda trial: objective_xgb(trial, X, y)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # 2. Define Study
    study = optuna.create_study(
        direction='minimize', 
        study_name=f"{study_name}_{model_type}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # 3. Optimize
    print(f"   Running {n_trials} trials for {model_type}...")
    study.optimize(func, n_trials=n_trials)
    
    print("\n✅ Optimization Completed!")
    print(f"   Best Trial Value (RMSE): {study.best_value:.4f}")
    print("   Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")
        
    # Save best params
    os.makedirs('results/optimization', exist_ok=True)
    df_results = study.trials_dataframe()
    df_results.to_csv(f'results/optimization/{study_name}_{model_type}_results.csv')
    print(f"   Results saved to results/optimization/{study_name}_{model_type}_results.csv")
    
    # --- Generate Plots for Report ---
    try:
        import matplotlib.pyplot as plt
        import optuna.visualization.matplotlib as optuna_vis
        
        # 1. Optimization History
        plt.figure(figsize=(10, 6))
        optuna_vis.plot_optimization_history(study)
        plt.title(f'Optimization History ({model_type.upper()})')
        plt.tight_layout()
        plt.savefig(f'results/optimization/{study_name}_{model_type}_history.png')
        print(f"   📊 Optimization History saved: results/optimization/{study_name}_{model_type}_history.png")
        
        # 2. Parameter Importance
        if len(study.trials) > 1:
            plt.figure(figsize=(10, 6))
            try:
                optuna_vis.plot_param_importances(study)
                plt.title(f'Hyperparameter Importance ({model_type.upper()})')
                plt.tight_layout()
                plt.savefig(f'results/optimization/{study_name}_{model_type}_importance.png')
                print(f"   📊 Parameter Importance saved: results/optimization/{study_name}_{model_type}_importance.png")
            except Exception as e:
                print(f"   ⚠️ Skipping parameter importance plot (not enough data or error): {e}")

    except ImportError:
        print("   ⚠️ Matplotlib or optuna-visualization not found, skipping plots.")
    except Exception as e:
        print(f"   ⚠️ Error creating plots: {e}")
    
    return study.best_params

if __name__ == "__main__":
    # Example usage
    file_path = 'data/processed/Data_imputed.csv'
    
    # Check if data exists
    if not os.path.exists(file_path):
        # Fallback to absolute path or assume running from project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_path = os.path.join(project_root, 'data/processed/Data_imputed.csv')
        
    # Create required directories
    os.makedirs('results/optimization', exist_ok=False if os.path.exists('results/optimization') else True)

    # Note: User requested NOT to run it effectively, but gave permission to "myself execute it".
    # So we provide the script ready to run.
    print("Optimization script ready. Run this script to start tuning.")
    
    # Uncomment to run immediately:
    # run_optimization(file_path, include_ids=True, n_trials=20)
