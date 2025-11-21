import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
    file_path = 'Preprocessing\Data_imputed.csv'

    try:
        # 1. 資料處理
        X_train, X_test, y_train, y_test, features = preprocess_data(file_path)

        # 2. 分別訓練模型 (現在是獨立的函式呼叫)
        rf_model, rf_pred = train_rf_model(X_train, y_train, X_test)
        xgb_model, xgb_pred = train_xgb_model(X_train, y_train, X_test)

        # 3. 統一評估
        evaluate_and_plot_comparison(y_test, rf_pred, xgb_pred, xgb_model, features)

    except Exception as e:
        print(f"發生錯誤: {e}")