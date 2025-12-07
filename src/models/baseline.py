import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.config import ModelConfig

def train_rf_model(X_train, y_train, X_test, **kwargs):
    """
    Train Random Forest model with fixed parameters (Fast Mode).
    Skipping RandomizedSearchCV to save time.
    """
    print("   -> Training Random Forest (Fixed Parameters)...")
    
    # 1. 基礎參數 (從 Config 讀取)
    params = ModelConfig.RF_PARAMS.copy()
    params.update(kwargs)
    
    # 手動設定一些合理的固定參數 (若 Config 沒給)
    if 'max_depth' not in params:
         params['max_depth'] = 25 # 限制深度以加快速度並防止過擬合
    
    model = RandomForestRegressor(**params)
    
    # 2. 直接訓練 (不進行 CV 搜尋)
    model.fit(X_train, y_train)
    
    print(f"      RF 訓練完成 (n_estimators={params.get('n_estimators')}, max_depth={params.get('max_depth')})")
    
    # 3. 預測
    y_pred = model.predict(X_test)
    return model, y_pred

def train_xgb_model(X_train, y_train, X_test, **kwargs):
    """
    Train XGBoost model with randomized search.
    Accepts kwargs to override default config.
    """
    print("   -> Training XGBoost...")
    
    params = ModelConfig.XGB_PARAMS.copy()
    params.update(kwargs)
    
    # Separate fit params or search params if needed, but for now we stick to the original logic
    # The original logic created an XGBRegressor and then ran RandomizedSearchCV
    
    xgb = XGBRegressor(**params)
    
    # 使用 RandomizedSearchCV 尋找最佳參數
    param_dist = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    search = RandomizedSearchCV(
        xgb, 
        param_distributions=param_dist, 
        n_iter=10, 
        scoring='neg_mean_squared_error', 
        cv=3, 
        random_state=ModelConfig.RANDOM_SEED, 
        n_jobs=-1, # Windows 上避免 joblib 崩潰
        verbose=1
    )
    
    try:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"      最佳參數: {search.best_params_}")
    except Exception as e:
        print(f"      ⚠️ GPU 訓練失敗，切換回 CPU: {e}")
        xgb.set_params(device='cpu', tree_method='auto')
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    return best_model, y_pred
