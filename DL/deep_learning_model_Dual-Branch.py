import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks, Input
from tensorflow.keras.models import Model
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

def check_gpu_availability():
    """檢查 GPU 狀態"""
    print("\n" + "="*60)
    print("🔍 硬體設備檢測")
    print("="*60)
    
    # TensorFlow 版本
    print(f"TensorFlow 版本: {tf.__version__}")
    
    # 檢查可用的物理設備
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
        
        # 對於 Apple Silicon，顯示額外資訊
        import platform
        if platform.processor() == 'arm':
            print(f"\n  🍎 Apple Silicon 偵測:")
            print(f"     處理器: {platform.processor()}")
            print(f"     系統: {platform.system()} {platform.release()}")
            print(f"     ✅ 使用 Metal 後端進行 GPU 加速")
    else:
        print("  ⚠️  未找到 GPU，將使用 CPU 訓練")
    
    print("="*60 + "\n")
    
    return len(gpus) > 0

def split_features_by_type(X, feature_names):
    """
    ⭐ 關鍵函數：將特徵區分為「基因」與「藥物」
    
    邏輯：
    GDSC 資料集中，基因通常是大寫字母 (如 BRAF, TP53)，
    而藥物特徵通常包含小寫、數字或特定關鍵字 (如 PubChem, drug, descriptors)。
    """
    feature_names = list(feature_names)
    
    # 定義藥物特徵的關鍵字 (根據你的資料集調整)
    # 如果你的藥物特徵是 One-Hot (如 'DRUG_Name'), 或者是指紋 (Fingerprint)
    drug_keywords = ['drug', 'Drug', 'PubChem', 'fingerprint', 'descriptor']
    
    drug_indices = []
    gene_indices = []
    
    for i, col in enumerate(feature_names):
        # 判斷邏輯：如果欄位名稱包含藥物關鍵字，或是看起來不像基因 (基因通常全是英文大寫)
        is_drug = any(k in col for k in drug_keywords)
        
        # 備用邏輯：如果沒有明確關鍵字，通常非基因欄位比較少，可以用排除法
        # 假設基因欄位大寫比例很高
        if is_drug:
            drug_indices.append(i)
        else:
            gene_indices.append(i)
            
    # 如果自動偵測失敗 (例如全部都被歸類為基因)，則強制使用簡單分割
    # 這裡假設後面的欄位通常是藥物 (若資料有經過特定排序)
    if len(drug_indices) == 0:
        print("⚠️ 警告：無法自動偵測藥物欄位，將嘗試使用啟發式分割...")
        # 假設特徵數量少於 5000 的類別可能是藥物，或者直接取後 20%
        # 這裡僅作示範，建議使用者確認欄位名稱
        split_point = int(len(feature_names) * 0.9) 
        gene_indices = list(range(split_point))
        drug_indices = list(range(split_point, len(feature_names)))

    print(f"  特徵分離結果: 基因特徵 {len(gene_indices)} 個, 藥物特徵 {len(drug_indices)} 個")
    
    # 轉換為 Numpy 並分割
    X_genes = X[:, gene_indices]
    X_drugs = X[:, drug_indices]
    
    return X_genes, X_drugs, gene_indices, drug_indices

"""
def build_dual_branch_model(gene_dim, drug_dim, learning_rate=0.0001):
    
    # ⭐ 建立雙分支模型 (Dual-Branch Architecture)
    
    # Branch 1: 處理基因 (Gene Expression)
    # Branch 2: 處理藥物 (Drug Descriptors/Fingerprints)
    # Fusion:   結合兩者進行預測
    
    
    # --- Branch 1: Gene Tower (基因塔) ---
    gene_input = Input(shape=(gene_dim,), name='gene_input')
    
    # 使用深層網路壓縮基因資訊 (類似 Autoencoder 的 Encoder 部分)
    x = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-4))(gene_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    gene_features = layers.Dense(256, activation='relu', name='gene_latent')(x)
    
    # --- Branch 2: Drug Tower (藥物塔) ---
    drug_input = Input(shape=(drug_dim,), name='drug_input')
    
    y = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(drug_input)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Dropout(0.2)(y)
    
    y = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    
    drug_features = layers.Dense(128, activation='relu', name='drug_latent')(y)
    
    # --- Fusion Layer (融合層) ---
    # 將基因特徵與藥物特徵拼接
    combined = layers.Concatenate()([gene_features, drug_features])
    
    # --- Prediction Head (預測層) ---
    z = layers.Dense(512, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    
    z = layers.Dense(256, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    
    z = layers.Dense(64, activation='relu')(z)
    
    # 輸出層 (回歸預測 IC50)
    output = layers.Dense(1, activation='linear', name='output')(z)
    
    model = Model(inputs=[gene_input, drug_input], outputs=output, name='Dual_Branch_Network')
    
    # 編譯模型
    optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    
    return model
"""

def build_dual_branch_model(gene_dim, drug_dim, learning_rate=0.000001):  # ⭐⭐ 極低學習率防止 Metal 爆炸
    """
    ⭐ 穩定版：專為 Mac Metal 優化，防止梯度爆炸
    
    修改策略：
    1. 移除所有 BatchNormalization（Metal 上數值不穩定）
    2. 使用 He initialization 初始化權重
    3. 極強梯度裁剪 (clipnorm=0.5)
    4. 降低 L2 正則化強度
    5. 使用極低的學習率
    """
    
    # He initialization for ReLU activations
    initializer = keras.initializers.HeNormal(seed=42)
    
    # --- Branch 1: Gene Tower (簡化版) ---
    gene_input = Input(shape=(gene_dim,), name='gene_input')
    
    x = layers.Dense(512, 
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-5),  # ⭐ 降低正則化
                     activation='relu')(gene_input)
    x = layers.Dropout(0.4)(x)  # ⭐ 增加 dropout 替代 BN
    
    x = layers.Dense(256, 
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-5),
                     activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    gene_features = layers.Dense(128, 
                                 kernel_initializer=initializer,
                                 activation='relu', 
                                 name='gene_latent')(x)
    
    # --- Branch 2: Drug Tower (簡化版) ---
    drug_input = Input(shape=(drug_dim,), name='drug_input')
    
    y = layers.Dense(256, 
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-5),
                     activation='relu')(drug_input)
    y = layers.Dropout(0.3)(y)
    
    y = layers.Dense(128, 
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizers.l2(1e-5),
                     activation='relu')(y)
    
    drug_features = layers.Dense(64, 
                                 kernel_initializer=initializer,
                                 activation='relu', 
                                 name='drug_latent')(y)
    
    # --- Fusion Layer ---
    combined = layers.Concatenate()([gene_features, drug_features])
    
    # --- Prediction Head (簡化版) ---
    z = layers.Dense(128, 
                     kernel_initializer=initializer,
                     activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    
    z = layers.Dense(64, 
                     kernel_initializer=initializer,
                     activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    
    # 輸出層
    output = layers.Dense(1, 
                          kernel_initializer=initializer,
                          activation='linear', 
                          name='output')(z)
    
    model = Model(inputs=[gene_input, drug_input], outputs=output, name='Dual_Branch_Network_Stable')
    
    # ⭐⭐⭐ 關鍵修正：極強的梯度裁剪 + 極低學習率
    optimizer = keras.optimizers.legacy.Adam(
        learning_rate=learning_rate,  # 0.000001
        clipnorm=0.5,      # ⭐ 強力梯度裁剪
        clipvalue=0.5,     # ⭐ 雙重保險：同時限制梯度絕對值
        epsilon=1e-7       # ⭐ 數值穩定性
    )
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    
    return model

def train_dual_model(X_train_g, X_train_d, y_train, X_val_g, X_val_d, y_val, epochs=150, batch_size=256):  # ⭐ 增大 batch_size
    """訓練雙分支模型 - 穩定版"""
    print("\n" + "="*50)
    print("🚀 開始訓練雙分支深度學習模型 (Dual-Branch DL)")
    print("="*50)
    
    gene_dim = X_train_g.shape[1]
    drug_dim = X_train_d.shape[1]
    
    print(f"  Gene features: {gene_dim}")
    print(f"  Drug features: {drug_dim}")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    
    model = build_dual_branch_model(gene_dim, drug_dim)
    model.summary()
    
    # Callbacks - 更保守的設定
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=30,  # ⭐ 增加耐心，因為學習率很低
        restore_best_weights=True, 
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=12,  # ⭐ 增加耐心
        min_lr=1e-8,  # ⭐ 允許更低的學習率
        verbose=1
    )
    
    # ⭐ 新增：NaN 檢測，如果 loss 變成 NaN 立即停止
    terminate_on_nan = callbacks.TerminateOnNaN()
    
    history = model.fit(
        x=[X_train_g, X_train_d],
        y=y_train,
        validation_data=([X_val_g, X_val_d], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr, terminate_on_nan],
        verbose=1
    )
    
    return model, history

def evaluate_model(y_true, y_pred, model_name="Dual-Branch DL"):
    """評估模型效能"""
    y_pred = y_pred.flatten()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Spearman 相關係數
    spearman_corr, spearman_pval = spearmanr(y_true, y_pred)
    
    print("\n" + "="*50)
    print(f"{model_name} 評估結果")
    print("="*50)
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  Spearman Correlation: {spearman_corr:.4f} (p={spearman_pval:.4e})")
    print("="*50)
    
    return {
        'R2': r2, 
        'RMSE': rmse, 
        'MAE': mae,
        'MSE': mse,
        'Spearman_Correlation': spearman_corr,
        'Spearman_PValue': spearman_pval
    }

def plot_results(history, y_test, y_pred, metrics):
    """繪製 Loss 曲線與預測散點圖"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss Curve
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Learning Curve (Loss)')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Prediction Scatter
    y_pred_flat = y_pred.flatten()
    axes[1].scatter(y_test, y_pred_flat, alpha=0.5, s=20)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes[1].set_title(f'Actual vs Predicted (R²={metrics["R2"]:.3f})')
    axes[1].set_xlabel('Actual IC50')
    axes[1].set_ylabel('Predicted IC50')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dl_dual_branch_results.png')
    print("✓ 結果圖表已儲存至 dl_dual_branch_results.png")
    plt.close()
"""
def run_deep_learning_pipeline(X_train, X_test, y_train, y_test, feature_names):
    
    # 執行完整的雙分支深度學習流程
    
    check_gpu_availability()
    
    # 1. 確保資料格式正確
    if hasattr(X_train, 'values'): 
        X_train = X_train.values
    if hasattr(X_test, 'values'): 
        X_test = X_test.values
    if hasattr(y_train, 'values'): 
        y_train = y_train.values
    if hasattr(y_test, 'values'): 
        y_test = y_test.values
    
    # 2. 特徵標準化 (Standardization) - 對神經網路極為重要！
    print("\n進行特徵標準化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. 分離基因與藥物特徵
    print("\n正在分離基因與藥物特徵...")
    X_train_g, X_train_d, g_idx, d_idx = split_features_by_type(X_train_scaled, feature_names)
    X_test_g, X_test_d, _, _ = split_features_by_type(X_test_scaled, feature_names)
    
    # 4. 驗證集切分
    X_tr_g, X_val_g, X_tr_d, X_val_d, y_tr, y_val = train_test_split(
        X_train_g, X_train_d, y_train, test_size=0.15, random_state=42
    )
    
    # 5. 訓練模型
    model, history = train_dual_model(
        X_tr_g, X_tr_d, y_tr, 
        X_val_g, X_val_d, y_val, 
        epochs=150, batch_size=128
    )
    
    # 6. 預測與評估
    y_pred = model.predict([X_test_g, X_test_d])
    metrics = evaluate_model(y_test, y_pred)
    
    # 7. 繪圖
    plot_results(history, y_test, y_pred, metrics)
    
    # 為了保持與 main.py 接口一致，返回部分物件
    return None, model, y_pred, metrics
"""

def run_deep_learning_pipeline(X_train, X_test, y_train, y_test, feature_names):
    """執行完整的雙分支深度學習流程 - 穩定版"""
    check_gpu_availability()
    
    # ⭐ 修正 1: 強制轉型 float32 防止 Mac Metal 數值溢出
    print("\n[System] Converting data to float32...")
    if hasattr(X_train, 'values'): X_train = X_train.values
    if hasattr(X_test, 'values'): X_test = X_test.values
    if hasattr(y_train, 'values'): y_train = y_train.values
    if hasattr(y_test, 'values'): y_test = y_test.values
    
    X_train = np.asarray(X_train).astype('float32')
    X_test = np.asarray(X_test).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    
    # ⭐ 修正 2: 檢查 NaN 和 Inf
    print("\n[Validation] Checking for NaN and Inf values...")
    def check_data(data, name):
        if np.any(np.isnan(data)):
            print(f"  ⚠️  WARNING: {name} contains NaN values!")
            return False
        if np.any(np.isinf(data)):
            print(f"  ⚠️  WARNING: {name} contains Inf values!")
            return False
        print(f"  ✅ {name} is clean")
        return True
    
    check_data(X_train, "X_train")
    check_data(X_test, "X_test")
    check_data(y_train, "y_train")
    check_data(y_test, "y_test")
    
    # 2. 特徵標準化
    print("\n[Preprocessing] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ⭐ 修正 3: 限制標準化後的極端值（防止爆炸）
    print("\n[Safety] Clipping extreme values after standardization...")
    X_train_scaled = np.clip(X_train_scaled, -10, 10)  # 限制在 ±10 標準差
    X_test_scaled = np.clip(X_test_scaled, -10, 10)
    
    print(f"  Data range after clipping: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    
    # 3. 分離特徵
    print("\n[Preprocessing] Splitting Gene and Drug features...")
    X_train_g, X_train_d, g_idx, d_idx = split_features_by_type(X_train_scaled, feature_names)
    X_test_g, X_test_d, _, _ = split_features_by_type(X_test_scaled, feature_names)
    
    # 4. 驗證集切分
    print("\n[Splitting] Creating validation set...")
    X_tr_g, X_val_g, X_tr_d, X_val_d, y_tr, y_val = train_test_split(
        X_train_g, X_train_d, y_train, test_size=0.15, random_state=42
    )
    
    print(f"  Training set size: {len(y_tr)}")
    print(f"  Validation set size: {len(y_val)}")
    print(f"  Test set size: {len(y_test)}")
    
    # 5. 訓練模型
    print("\n" + "="*60)
    print("開始訓練模型（預計需要數分鐘，請耐心等待...）")
    print("="*60)
    
    model, history = train_dual_model(
        X_tr_g, X_tr_d, y_tr, 
        X_val_g, X_val_d, y_val, 
        epochs=150, batch_size=256  # ⭐ 使用較大的 batch size
    )
    
    # 6. 預測與評估
    print("\n[Evaluation] Making predictions on test set...")
    y_pred = model.predict([X_test_g, X_test_d], verbose=0)
    metrics = evaluate_model(y_test, y_pred)
    
    # 7. 繪圖
    print("\n[Visualization] Creating plots...")
    plot_results(history, y_test, y_pred, metrics)
    
    print("\n" + "="*60)
    print("✅ 訓練完成！")
    print("="*60)
    
    return None, model, y_pred, metrics

# ==================== 主程式執行區 ====================
if __name__ == "__main__":
    """
    獨立執行此檔案進行測試
    """
    print("深度學習模型 (雙分支架構) - 獨立執行模式")
    print("="*60)
    
    # 載入與前處理資料
    from main import preprocess_data
    
    file_path = 'Preprocessing/Data_imputed.csv'
    
    try:
        # 資料處理
        X_train, X_test, y_train, y_test, features = preprocess_data(file_path)
        
        # 執行深度學習 Pipeline
        encoder, model, y_pred, metrics = run_deep_learning_pipeline(
            X_train, X_test, y_train, y_test, features
        )
        
        print("\n模型訓練與評估完成！")
        print("生成的檔案:")
        print("  - dl_dual_branch_results.png")
        
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()

# result
# ==================================================
# Dual-Branch DL 評估結果
# ==================================================
#   R² Score: 0.4079
#   RMSE:     2.1253
#   MAE:      1.6857
#   Spearman Correlation: 0.5992 (p=0.0000e+00)
# ==================================================