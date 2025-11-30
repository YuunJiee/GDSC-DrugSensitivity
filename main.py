import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler  # ⭐ 新增：用於特徵標準化

# 導入深度學習模型
from DL.deep_learning_model import preprocess_data_standalone
from DL.deep_learning_model import run_deep_learning_pipeline

# 設定繪圖風格
plt.style.use('ggplot')

def preprocess_data(file_path):
    """
    階段 1: 資料前處理
    負責讀取、清洗、特徵工程與分割
    """
    print(f"Step 1: 正在讀取資料並進行前處理: {file_path}...")

    df = pd.read_csv(file_path)
    target = 'LN_IC50'

    # 移除 ID 與 Data Leakage 欄位
    drop_cols = [
        'COSMIC_ID', 'CELL_LINE_NAME', 'DRUG_ID',
        'AUC', 'Z_SCORE', 'RMSE',
        'NLME_RESULT_ID', 'NLME_CURVE_ID', 'SANGER_MODEL_ID'
    ]

    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df_cleaned = df.drop(columns=existing_drop_cols)
    print(f"   -> 已移除欄位: {existing_drop_cols}")

    # One-Hot Encoding
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

    # 分割 X, y
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    feature_names = X.columns

    # 分割訓練/測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  
    print(f"   -> 資料形狀: X_train={X_train.shape}, X_test={X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_names

def train_rf_model(X_train, y_train, X_test):
    """
    階段 2-A: 訓練 Random Forest 模型
    獨立的 RF 訓練邏輯
    """
    print("\nStep 2-A: 正在訓練 Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 進行預測
    y_pred = model.predict(X_test)
    return model, y_pred

def train_xgb_model(X_train, y_train, X_test):
    """
    階段 2-B: 訓練 XGBoost 模型
    獨立的 XGBoost 訓練邏輯
    """
    print("Step 2-B: 正在訓練 XGBoost...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 進行預測
    y_pred = model.predict(X_test)
    return model, y_pred

def evaluate_and_plot_comparison(y_test, rf_pred, xgb_pred, xgb_model, feature_names):
    """
    階段 3: 評估與視覺化比較
    接收兩個模型的預測結果進行統一評估
    """
    print("\nStep 3: 比較模型結果...")

    # --- 內部評估函式 ---
    def calculate_metrics(y_true, y_pred, name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"   [{name}] RMSE: {rmse:.4f}, R2: {r2:.4f}")
        return rmse, r2

    print("\n" + "="*30)
    print("模型效能指標")
    print("="*30)
    rf_rmse, rf_r2 = calculate_metrics(y_test, rf_pred, "Random Forest")
    xgb_rmse, xgb_r2 = calculate_metrics(y_test, xgb_pred, "XGBoost")

    # --- 視覺化 ---
    plt.figure(figsize=(14, 6))

    # 左圖: RF
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, rf_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'Random Forest\nRMSE: {rf_rmse:.3f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # 右圖: XGB
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, xgb_pred, alpha=0.5, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'XGBoost\nRMSE: {xgb_rmse:.3f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\n圖表已保存為 'model_comparison.png'")

    # --- 特徵重要性 (XGB) ---
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    print("\n" + "="*30)
    print("XGBoost 前 10 重要特徵")
    print("="*30)
    print(feat_imp)


# --- 主程式執行區 ---
if __name__ == "__main__":
    file_path = 'Preprocessing/Data_imputed.csv' # Mac 檔案路徑
    # file_path = 'Preprocessing\Data_imputed.csv' # Windows 檔案路徑
    # 設定：是否執行深度學習模型
    RUN_DEEP_LEARNING = True  # 設為 False 只執行基線模型
    
    try:
        print("\n" + "="*30)
        print("GDSC 藥物敏感性預測專案")
        print("="*30 + "\n")
        
        # 1. 資料處理
        X_train, X_test, y_train, y_test, features = preprocess_data(file_path)
        
        # 轉換為 numpy array（深度學習需要）
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
        
        print("\n" + "="*60)
        print("開始訓練基線模型（機器學習）")
        print("="*60)
        
        # 2. 分別訓練基線模型
        rf_model, rf_pred = train_rf_model(X_train, y_train, X_test)
        xgb_model, xgb_pred = train_xgb_model(X_train, y_train, X_test)
        
        # 3. 統一評估基線模型
        evaluate_and_plot_comparison(y_test, rf_pred, xgb_pred, xgb_model, features)
        
        # 4. 深度學習模型（選擇性執行）
        if RUN_DEEP_LEARNING:
            print("\n\n" + "="*60)
            print("訓練深度學習模型功能啟用")
            print("="*60)
            
            # ⭐ 深度學習使用獨立的資料預處理（Label Encoding）
            # print("\n💡 注意：深度學習模型使用獨立的資料預處理流程（Label Encoding）")
            # print("   這與基線模型的 One-Hot Encoding 不同，可能產生更好的結果\n")
            
            X_train_dl, X_test_dl, y_train_dl, y_test_dl, features_dl, encoders = \
                preprocess_data_standalone(file_path)
            
            encoder, mlp_model, dl_pred, dl_metrics = run_deep_learning_pipeline(
                X_train_dl, X_test_dl, y_train_dl, y_test_dl, features_dl
            )
            
            print("\n" + "="*30)
            print("所有模型訓練完成！")
            print("="*30)
            
            # 最終比較
            print("\n" + "="*60)
            print("最終模型效能比較")
            print("="*60)
            print("\n說明：")
            print("   - Random Forest & XGBoost: 使用 One-Hot Encoding")
            print("   - Deep Learning: 使用 Label Encoding（獨立資料處理）")
            print("   - 由於資料處理方式不同，測試集可能略有差異\n")
            
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_r2 = r2_score(y_test, rf_pred)
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            xgb_r2 = r2_score(y_test, xgb_pred)
            
            print(f"\n{'模型':<30} {'RMSE':<12} {'R²':<12}")
            print("-"*60)
            print(f"{'Random Forest':<30} {rf_rmse:<12.4f} {rf_r2:<12.4f}")
            print(f"{'XGBoost':<30} {xgb_rmse:<12.4f} {xgb_r2:<12.4f}")
            print(f"{'Deep Learning (Neural Net)':<30} {dl_metrics['RMSE']:<12.4f} {dl_metrics['R2']:<12.4f}")
            print("="*60)
            
            # 判斷最佳模型
            best_model = max(
                [('Random Forest', rf_r2), 
                 ('XGBoost', xgb_r2), 
                 ('Deep Learning', dl_metrics['R2'])],
                key=lambda x: x[1]
            )
            print(f"\n最佳模型: {best_model[0]} (R² = {best_model[1]:.4f})")
            
            print("\n生成的檔案:")
            print("   基線模型:")
            print("     - model_comparison.png")
            print("   深度學習模型:")
            print("     - dl_learning_curves.png")
            print("     - dl_predictions_vs_actual.png")
            print("     - dl_feature_importance.png")
        else:
            print("\n✓ 基線模型訓練完成（深度學習模型已跳過）")
            print("   如需執行深度學習模型，請設定 RUN_DEEP_LEARNING = True")

    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()