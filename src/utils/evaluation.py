import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    計算並列印模型評估指標
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"   [{model_name}] RMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
    return rmse, r2, mae

def evaluate_dl_model(y_test, y_pred, model_name="Neural Network"):
    """
    深度學習模型的詳細評估
    """
    print(f"\n📊 評估 {model_name} 效能...")
    
    # 計算指標
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Spearman 相關係數
    spearman_corr, spearman_pval = spearmanr(y_test, y_pred)
    
    # 顯示結果
    print("\n" + "="*60)
    print(f"{model_name} 評估結果")
    print("="*60)
    print(f"  R² Score:              {r2:.4f}")
    print(f"  RMSE:                  {rmse:.4f}")
    print(f"  MAE:                   {mae:.4f}")
    print(f"  MSE:                   {mse:.4f}")
    print(f"  Spearman Correlation:  {spearman_corr:.4f} (p={spearman_pval:.4e})")
    print("="*60)
    
    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'Spearman_Correlation': spearman_corr,
        'Spearman_PValue': spearman_pval
    }
    
    return metrics
