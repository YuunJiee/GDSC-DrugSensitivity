"""
深度學習模型 - Neural Network Regression
基於 Kaggle Notebook 改進的方案B（完整版）

功能：
1. 可獨立執行進行測試
2. 可被 main.py 調用（提供 run_deep_learning_pipeline 函數）
3. 包含類別型特徵編碼、標準化、深層網路、callbacks 等完整功能

作者：改編自 Kaggle - Drug Sensitivity Notebook
日期：2025-11-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# ==================== GPU 檢測 ====================
def check_gpu_availability():
    """檢查並顯示 GPU 狀態"""
    print("\n" + "="*60)
    print("🔍 硬體設備檢測")
    print("="*60)
    
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # 檢查可用設備
    physical_devices = tf.config.list_physical_devices()
    print(f"\n所有可用設備:")
    for device in physical_devices:
        print(f"  - {device.device_type}: {device.name}")
    
    # 檢查 GPU
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    
    print(f"\n🖥️  GPU 設備:")
    if gpu_available:
        print(f"  ✅ 找到 {len(gpus)} 個 GPU: {[gpu.name for gpu in gpus]}")
        
        # Apple Silicon 特殊提示
        import platform
        if platform.processor() == 'arm':
            print(f"\n  🍎 Apple Silicon 偵測:")
            print(f"     處理器: {platform.processor()}")
            print(f"     系統: {platform.system()} {platform.release()}")
            print(f"     ✅ 使用 Metal 後端進行 GPU 加速")
    else:
        print("  ⚠️  未找到 GPU，將使用 CPU 訓練")
    
    print("="*60 + "\n")
    
    return gpu_available

# ==================== 資料預處理（獨立執行模式用） ====================
def preprocess_data_standalone(file_path):
    """
    獨立的資料預處理函數
    專門用於獨立執行模式，不依賴 main.py
    
    處理流程：
    1. 讀取資料
    2. 移除缺失值
    3. 類別型特徵編碼
    4. 分割訓練/測試集
    """
    print(f"\n📂 正在讀取資料: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"   原始資料形狀: {df.shape}")
    print(f"   欄位數量: {len(df.columns)}")
    
    # 移除目標變數缺失值
    df_clean = df.dropna(subset=['LN_IC50'])
    print(f"   移除缺失值後: {df_clean.shape}")
    
    # 處理類別型特徵
    print("\n🔄 處理類別型特徵...")
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    print(f"   發現 {len(categorical_cols)} 個類別型欄位")
    
    # 移除不需要的欄位（ID、名稱等）
    exclude_cols = ['COSMIC_ID', 'CELL_LINE_NAME', 'DRUG_NAME', 'DRUG_ID']
    
    # 對類別型特徵進行 Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        if col not in exclude_cols and col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
            print(f"     ✓ {col}: {len(le.classes_)} 個類別")
    
    # 選擇特徵（排除ID、名稱和目標變數）
    drop_cols = [col for col in exclude_cols if col in df_clean.columns] + ['LN_IC50']
    features = df_clean.drop(columns=drop_cols)
    target = df_clean['LN_IC50']
    
    feature_names = features.columns.tolist()
    
    print(f"\n📊 最終特徵數量: {len(feature_names)}")
    print(f"   數值型特徵: {len(features.select_dtypes(include=[np.number]).columns)}")
    
    # 分割訓練/測試集
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    print(f"\n✅ 資料分割完成:")
    print(f"   訓練集: {X_train.shape}")
    print(f"   測試集: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_names, label_encoders

# ==================== 建立深度學習模型 ====================
def build_neural_network(input_dim):
    """
    建立深度神經網路模型（方案B：完整版）
    
    架構：
    - 輸入層
    - Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    - Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    - Dense(64) + ReLU + Dropout(0.2)
    - 輸出層 (1個神經元，線性激活)
    
    參數：
        input_dim: 輸入特徵維度
    
    返回：
        編譯好的 Keras Sequential 模型
    """
    model = Sequential([
        # 第一層：256 神經元
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # 第二層：128 神經元
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # 第三層：64 神經元
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        # 輸出層：回歸任務
        Dense(1, activation='linear')
    ], name='Neural_Network_Regression')
    
    # 編譯模型
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mse']
    )
    
    return model

# ==================== 訓練模型 ====================
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, verbose=1):
    """
    訓練深度學習模型
    
    參數：
        model: Keras 模型
        X_train, y_train: 訓練資料
        X_val, y_val: 驗證資料
        epochs: 訓練輪數
        batch_size: 批次大小
        verbose: 訓練過程輸出詳細程度
    
    返回：
        history: 訓練歷史記錄
    """
    print(f"\n🚀 開始訓練模型...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   訓練樣本: {len(y_train)}")
    print(f"   驗證樣本: {len(y_val)}")
    
    # 設定 Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # 訓練模型
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    print("\n✅ 訓練完成！")
    
    return history

# ==================== 評估模型 ====================
def evaluate_model(model, X_test, y_test, model_name="Neural Network"):
    """
    評估模型效能
    
    參數：
        model: 訓練好的模型
        X_test, y_test: 測試資料
        model_name: 模型名稱（用於顯示）
    
    返回：
        y_pred: 預測結果
        metrics: 評估指標字典
    """
    print(f"\n📊 評估 {model_name} 效能...")
    
    # 預測
    y_pred = model.predict(X_test, verbose=0).flatten()
    
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
    
    return y_pred, metrics

# ==================== 視覺化結果 ====================
def plot_training_history(history, save_path='dl_learning_curves.png'):
    """繪製訓練過程曲線"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # MAE 曲線
    ax1.plot(history.history['mean_absolute_error'], label='Train MAE', linewidth=2)
    ax1.plot(history.history['val_mean_absolute_error'], label='Val MAE', linewidth=2)
    ax1.set_title('Model MAE over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss 曲線
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss (MSE) over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 訓練曲線已保存至: {save_path}")
    plt.close()

def plot_predictions(y_test, y_pred, metrics, save_path='dl_predictions_vs_actual.png'):
    """繪製預測 vs 實際值散點圖"""
    plt.figure(figsize=(10, 8))
    
    # 散點圖
    plt.scatter(y_test, y_pred, alpha=0.5, s=20, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # 完美預測線
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # 標題和標籤
    plt.title(f'Predictions vs Actual IC50 Values\nR² = {metrics["R2"]:.4f}, RMSE = {metrics["RMSE"]:.4f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Actual LN_IC50', fontsize=12)
    plt.ylabel('Predicted LN_IC50', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加統計資訊
    textstr = f'MAE = {metrics["MAE"]:.4f}\nSpearman ρ = {metrics["Spearman_Correlation"]:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 預測散點圖已保存至: {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20, save_path='dl_feature_importance.png'):
    """
    計算並繪製特徵重要性（基於權重）
    
    注意：深度學習的特徵重要性計算較為複雜，
    這裡使用第一層權重的絕對值平均作為近似值
    """
    print(f"\n📈 計算特徵重要性（Top {top_n}）...")
    
    try:
        # 獲取第一層的權重
        first_layer_weights = model.layers[0].get_weights()[0]  # shape: (n_features, n_neurons)
        
        # 計算每個特徵的重要性（權重絕對值的平均）
        feature_importance = np.abs(first_layer_weights).mean(axis=1)
        
        # 創建 DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # 繪圖
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'], color='skyblue', edgecolor='navy')
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance (Absolute Weight Mean)', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 特徵重要性圖已保存至: {save_path}")
        plt.close()
        
        # 顯示 Top 10
        print("\n" + "="*60)
        print(f"Top 10 Most Important Features")
        print("="*60)
        print(importance_df.head(10).to_string(index=False))
        print("="*60)
        
    except Exception as e:
        print(f"⚠️  特徵重要性計算失敗: {e}")

# ==================== 主要 Pipeline 函數（供 main.py 調用） ====================
def run_deep_learning_pipeline(X_train, X_test, y_train, y_test, feature_names):
    """
    執行完整的深度學習 Pipeline
    
    這是供 main.py 調用的主要接口函數
    
    參數：
        X_train, X_test: 訓練/測試特徵（DataFrame 或 Array）
        y_train, y_test: 訓練/測試目標變數（Series 或 Array）
        feature_names: 特徵名稱列表
    
    返回：
        encoder: 編碼器（這裡返回 scaler）
        model: 訓練好的深度學習模型
        y_pred: 測試集預測結果
        metrics: 評估指標字典
    """
    print("\n" + "="*60)
    print("深度學習模型訓練開始")
    print("="*60)
    
    # 檢查 GPU
    check_gpu_availability()
    
    # 1. 確保資料格式正確（轉換為 numpy array）
    print("\n[1/7] 資料格式轉換...")
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    print(f"   訓練集形狀: {X_train.shape}")
    print(f"   測試集形狀: {X_test.shape}")
    
    # 2. 特徵標準化
    print("\n[2/7] 特徵標準化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   標準化後範圍: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    
    # 3. 驗證集切分
    print("\n[3/7] 切分驗證集...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    print(f"   最終訓練集: {X_train_final.shape}")
    print(f"   驗證集:     {X_val.shape}")
    
    # 4. 建立模型
    print("\n[4/7] 建立神經網路模型...")
    input_dim = X_train_scaled.shape[1]
    model = build_neural_network(input_dim)
    
    print("\n模型架構:")
    model.summary()
    
    # 5. 訓練模型
    print("\n[5/7] 訓練模型...")
    history = train_model(
        model,
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=100,
        batch_size=64,
        verbose=1
    )
    
    # 6. 評估模型
    print("\n[6/7] 評估模型...")
    y_pred, metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # 7. 視覺化結果
    print("\n[7/7] 生成視覺化圖表...")
    plot_training_history(history)
    plot_predictions(y_test, y_pred, metrics)
    plot_feature_importance(model, feature_names)
    
    print("\n" + "="*60)
    print("✅ 深度學習模型訓練完成！")
    print("="*60)
    print("\n📊 生成的檔案:")
    print("   - dl_learning_curves.png")
    print("   - dl_predictions_vs_actual.png")
    print("   - dl_feature_importance.png")
    
    # 返回結果（符合 main.py 的接口）
    return scaler, model, y_pred, metrics

# ==================== 獨立執行模式 ====================
if __name__ == "__main__":
    """
    獨立執行此檔案進行測試
    
    用法：
        python DL/deep_learning_model.py
    """
    print("\n" + "="*30)
    print("深度學習模型 - 獨立測試模式")
    print(""*30)
    
    file_path = '../Preprocessing/Data_imputed.csv'  # 相對於 DL 資料夾的路徑
    
    try:
        # 1. 資料預處理
        X_train, X_test, y_train, y_test, features, encoders = preprocess_data_standalone(file_path)
        
        # 2. 執行深度學習 Pipeline
        scaler, model, y_pred, metrics = run_deep_learning_pipeline(
            X_train, X_test, y_train, y_test, features
        )
        
        # 3. 完成提示
        print("\n" + "="*30)
        print("所有任務完成！")
        print("="*30)
        
        print("\n模型效能總結:")
        print(f"   R² Score: {metrics['R2']:.4f}")
        print(f"   RMSE:     {metrics['RMSE']:.4f}")
        print(f"   MAE:      {metrics['MAE']:.4f}")
        
    except FileNotFoundError:
        print(f"\n❌ 錯誤: 找不到檔案 '{file_path}'")
        print("   請確認檔案路徑是否正確")
        print("   如果從專案根目錄執行，請使用: python DL/deep_learning_model.py")
        
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()


# 模型效能總結:
#    R² Score: 0.6189
#    RMSE:     1.7052
#    MAE:      1.1486