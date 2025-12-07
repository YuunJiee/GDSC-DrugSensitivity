import os

class PathConfig:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA = os.path.join(DATA_DIR, 'processed')
    PROCESSED_DATA_DIR = PROCESSED_DATA
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
    TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
    REPORTS_DIR = os.path.join(RESULTS_DIR, 'reports')

class ModelConfig:
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    
    # Random Forest
    RF_PARAMS = {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }
    
    # XGBoost
    XGB_PARAMS = {
        'objective': 'reg:squarederror',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'device': 'cuda', # Will fallback to cpu if needed
        'random_state': 42,
        'n_jobs': 1
    }
    
    # Deep Learning
    DL_PARAMS = {
        'epochs': 100,
        'batch_size': 128, # Optimized from 64
        'learning_rate': 0.0016 # Optimized from 0.001
    }
    
    # Optuna Optimized Hyperparameters
    BEST_DL_PARAMS = {
        'cell_layers': 3,
        'cell_units': [256, 64, 64],
        'cell_dropout': 0.127,
        'drug_layers': 3,
        'drug_units': [128, 256, 256],
        'drug_dropout': 0.207,
        'fusion_layers': 1,
        'fusion_units': [256],
        'fusion_dropout': 0.301
    }

class DataConfig:
    TARGET = 'LN_IC50'
    VARIANCE_THRESHOLD = 0.01
