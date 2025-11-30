import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler  # ⭐ 用於特徵標準化
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

# 設定繪圖風格
plt.style.use('ggplot')
sns.set_palette("husl")


def check_gpu_availability():
    """
    檢查並顯示可用的硬體設備（CPU/GPU）
    
    對於 Apple Silicon (M1/M2/M3)，TensorFlow 使用 Metal 後端進行 GPU 加速
    
    返回:
        device_info: 包含設備資訊的字典
    """
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
        print(f"  ✅ 找到 {len(gpus)} 個 GPU")
        for i, gpu in enumerate(gpus):
            print(f"     [{i}] {gpu.name}")
        
        # 對於 Apple Silicon，顯示額外資訊
        import platform
        if platform.processor() == 'arm':
            print(f"\n  🍎 Apple Silicon 偵測:")
            print(f"     處理器: {platform.processor()}")
            print(f"     系統: {platform.system()} {platform.release()}")
            print(f"     ✅ 使用 Metal 後端進行 GPU 加速")
    else:
        print(f"  ⚠️  未找到 GPU，將使用 CPU 訓練")
    
    # 檢查是否啟用混合精度
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.global_policy()
        print(f"\n⚡ 混合精度策略: {policy.name}")
    except:
        pass
    
    # 建議設定
    print(f"\n💡 訓練加速建議:")
    if gpu_available:
        print(f"  ✅ GPU 已啟用，訓練速度將大幅提升")
        print(f"  💡 如需進一步加速，可嘗試:")
        print(f"     - 增加 batch_size（如果記憶體足夠）")
        print(f"     - 啟用混合精度訓練（float16）")
    else:
        print(f"  💡 若要使用 GPU 加速:")
        print(f"     - 確認已安裝 tensorflow-metal（Apple Silicon）")
        print(f"     - 安裝指令: pip install tensorflow-metal")
    
    print("="*60 + "\n")
    
    device_info = {
        'gpu_available': gpu_available,
        'num_gpus': len(gpus),
        'gpu_names': [gpu.name for gpu in gpus],
        'tf_version': tf.__version__
    }
    
    return device_info



def build_autoencoder(input_dim, encoding_dim=128, l2_reg=0.001):
    """
    建立 Autoencoder 用於特徵降維
    
    架構: input -> 512 -> 256 -> encoding_dim -> 256 -> 512 -> output
    
    參數:
        input_dim: 輸入特徵維度
        encoding_dim: 編碼層維度（壓縮後的特徵數）
        l2_reg: L2 正則化強度
    
    返回:
        encoder: 編碼器模型
        autoencoder: 完整的自編碼器模型
    """
    # 輸入層
    input_layer = layers.Input(shape=(input_dim,))
    
    # 編碼器
    encoded = layers.Dense(512, activation='relu', 
                          kernel_regularizer=regularizers.l2(l2_reg))(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.3)(encoded)
    
    encoded = layers.Dense(256, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    
    # 壓縮層（瓶頸層）
    encoded = layers.Dense(encoding_dim, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name='encoded_features')(encoded)
    
    # 解碼器
    decoded = layers.Dense(256, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    
    decoded = layers.Dense(512, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg))(decoded)
    decoded = layers.BatchNormalization()(decoded)
    
    # 輸出層
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    # 建立模型
    autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')
    encoder = Model(inputs=input_layer, outputs=encoded, name='encoder')
    
    return encoder, autoencoder


def build_mlp_model(input_dim, l2_reg=0.0001, dropout_rate=0.3):  # ⭐ 降低 L2 正則化
    """
    建立 MLP 回歸模型（增強版）
    
    架構: input -> 256 -> 128 -> 64 -> 32 -> 16 -> 1
    
    參數:
        input_dim: 輸入特徵維度（編碼後的特徵數）
        l2_reg: L2 正則化強度
        dropout_rate: Dropout 比率
    
    返回:
        model: MLP 模型
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # ⭐ 新增：第一層增加神經元數量
        layers.Dense(256, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # ⭐ 新增：第二層
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.8),
        
        layers.Dense(64, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.6),
        
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate * 0.4),
        
        layers.Dense(16, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        
        layers.Dense(1, activation='linear')  # 回歸輸出
    ], name='mlp_regressor')
    
    return model


def train_autoencoder(X_train, X_val, epochs=100, batch_size=128, verbose=1):
    """
    訓練 Autoencoder
    
    參數:
        X_train: 訓練資料
        X_val: 驗證資料
        epochs: 訓練輪數
        batch_size: 批次大小
        verbose: 顯示詳細程度
    
    返回:
        encoder: 訓練好的編碼器
        autoencoder: 訓練好的自編碼器
        history: 訓練歷史
    """
    print("\n" + "="*50)
    print("階段 1: 訓練 Autoencoder 進行特徵降維")
    print("="*50)
    
    input_dim = X_train.shape[1]
    encoding_dim = 256  # ⭐ 從 128 提升到 256，保留更多特徵資訊
    
    # 建立模型
    encoder, autoencoder = build_autoencoder(input_dim, encoding_dim)
    
    # 編譯模型
    # 使用 legacy Adam optimizer 以兼容 Apple Silicon (M1/M2/M3)
    autoencoder.compile(
        # optimizer=keras.optimizers.Adam(learning_rate=0.001), # 為使用
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0005),  # ⭐ 降低學習率提升穩定性
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    # 訓練
    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    print(f"\n✓ Autoencoder 訓練完成")
    print(f"  - 原始特徵數: {input_dim}")
    print(f"  - 壓縮特徵數: {encoding_dim}")
    print(f"  - 壓縮比率: {encoding_dim/input_dim*100:.1f}%")
    
    return encoder, autoencoder, history


def train_mlp_model(X_train, y_train, X_val, y_val, epochs=200, batch_size=64, verbose=1):
    """
    訓練 MLP 回歸模型
    
    參數:
        X_train: 訓練特徵（編碼後）
        y_train: 訓練標籤
        X_val: 驗證特徵（編碼後）
        y_val: 驗證標籤
        epochs: 訓練輪數
        batch_size: 批次大小
        verbose: 顯示詳細程度
    
    返回:
        model: 訓練好的 MLP 模型
        history: 訓練歷史
    """
    print("\n" + "="*50)
    print("階段 2: 訓練 MLP 回歸模型")
    print("="*50)
    
    input_dim = X_train.shape[1]
    
    # 建立模型
    model = build_mlp_model(input_dim)
    
    # 編譯模型
    # 使用 legacy Adam optimizer 以兼容 Apple Silicon (M1/M2/M3)
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0005),  # ⭐ 降低學習率
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    # 訓練
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    print(f"\n✓ MLP 模型訓練完成")
    
    return model, history


def evaluate_dl_model(y_true, y_pred, model_name="Deep Learning"):
    """
    評估深度學習模型
    
    計算多種評估指標: MAE, MSE, RMSE, R², Spearman Correlation
    
    參數:
        y_true: 真實值
        y_pred: 預測值
        model_name: 模型名稱
    
    返回:
        metrics: 包含所有指標的字典
    """
    # 展平預測值（確保是一維）
    y_pred = y_pred.flatten()
    
    # 計算指標
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Spearman 相關係數
    spearman_corr, spearman_pval = spearmanr(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Spearman_Correlation': spearman_corr,
        'Spearman_PValue': spearman_pval
    }
    
    # 輸出結果
    print("\n" + "="*50)
    print(f"{model_name} 模型評估結果")
    print("="*50)
    print(f"  MAE  (Mean Absolute Error):     {mae:.4f}")
    print(f"  MSE  (Mean Squared Error):      {mse:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  R²   (R-squared):               {r2:.4f}")
    print(f"  Spearman Correlation:           {spearman_corr:.4f} (p={spearman_pval:.4e})")
    print("="*50)
    
    return metrics


def calculate_feature_importance(model, encoder, X_test, y_test, 
                                 feature_names, n_repeats=10, random_state=42):
    """
    使用 Permutation Importance 計算特徵重要性
    
    參數:
        model: MLP 模型
        encoder: Autoencoder 編碼器
        X_test: 測試資料（原始特徵）- numpy array 或 DataFrame
        y_test: 測試標籤 - numpy array 或 Series
        feature_names: 特徵名稱列表或 Index
        n_repeats: 排列重複次數
        random_state: 隨機種子
    
    返回:
        importance_df: 特徵重要性 DataFrame
    """
    print("\n計算特徵重要性（Permutation Importance）...")
    
    # 確保 X_test 是 numpy array
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    X_test = np.asarray(X_test, dtype=np.float32)
    
    # 確保 y_test 是 numpy array
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    y_test = np.asarray(y_test, dtype=np.float32).flatten()
    
    # 確保 feature_names 是列表
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    feature_names = list(feature_names)
    
    # 檢查維度匹配
    if X_test.shape[1] != len(feature_names):
        raise ValueError(
            f"特徵數量不匹配: X_test 有 {X_test.shape[1]} 個特徵，"
            f"但 feature_names 有 {len(feature_names)} 個名稱"
        )
    
    # 設定隨機種子
    np.random.seed(random_state)
    
    # 定義完整的預測流程
    def predict_pipeline(X):
        X_encoded = encoder.predict(X, verbose=0)
        return model.predict(X_encoded, verbose=0).flatten()
    
    # 計算基準分數
    y_pred_baseline = predict_pipeline(X_test)
    baseline_score = r2_score(y_test, y_pred_baseline)
    
    print(f"  基準 R² 分數: {baseline_score:.4f}")
    print(f"  計算 {len(feature_names)} 個特徵的重要性...")
    
    # 計算每個特徵的重要性
    importances = []
    for i, feature_name in enumerate(feature_names):
        if (i + 1) % 50 == 0:  # 每 50 個特徵顯示進度
            print(f"    進度: {i+1}/{len(feature_names)}")
        
        scores = []
        for _ in range(n_repeats):
            # 複製測試資料 - 確保是 numpy array
            X_permuted = np.copy(X_test)  # 使用 np.copy 而不是 .copy()
            # 隨機排列該特徵
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            # 預測
            y_pred_permuted = predict_pipeline(X_permuted)
            # 計算分數下降
            score = r2_score(y_test, y_pred_permuted)
            scores.append(baseline_score - score)
        
        importances.append({
            'feature': feature_name,
            'importance': np.mean(scores),
            'std': np.std(scores)
        })
    
    # 轉換為 DataFrame 並排序
    importance_df = pd.DataFrame(importances)
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print(f"✓ 特徵重要性計算完成")
    
    return importance_df



def plot_learning_curves(ae_history, mlp_history, save_path='dl_learning_curves.png'):
    """
    繪製學習曲線
    
    參數:
        ae_history: Autoencoder 訓練歷史
        mlp_history: MLP 訓練歷史
        save_path: 儲存路徑
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Autoencoder 學習曲線
    ax1 = axes[0]
    ax1.plot(ae_history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(ae_history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Autoencoder Learning Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # MLP 學習曲線
    ax2 = axes[1]
    ax2.plot(mlp_history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(mlp_history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('MLP Regressor Learning Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 學習曲線已儲存: {save_path}")
    plt.close()


def plot_predictions_vs_actual(y_test, y_pred, metrics, save_path='dl_predictions_vs_actual.png'):
    """
    繪製預測值 vs 實際值散點圖
    
    參數:
        y_test: 真實值
        y_pred: 預測值
        metrics: 評估指標字典
        save_path: 儲存路徑
    """
    y_pred = y_pred.flatten()
    
    plt.figure(figsize=(10, 8))
    
    # 散點圖
    plt.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    
    # 對角線（完美預測）
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # 標題和標籤
    plt.xlabel('Actual LN_IC50', fontsize=13, fontweight='bold')
    plt.ylabel('Predicted LN_IC50', fontsize=13, fontweight='bold')
    plt.title('Deep Learning Model: Predictions vs Actual', fontsize=15, fontweight='bold')
    
    # 添加評估指標文字
    textstr = f"R² = {metrics['R2']:.4f}\n"
    textstr += f"RMSE = {metrics['RMSE']:.4f}\n"
    textstr += f"MAE = {metrics['MAE']:.4f}\n"
    textstr += f"Spearman ρ = {metrics['Spearman_Correlation']:.4f}"
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 預測圖已儲存: {save_path}")
    plt.close()


def plot_feature_importance(importance_df, top_n=20, save_path='dl_feature_importance.png'):
    """
    繪製特徵重要性圖
    
    參數:
        importance_df: 特徵重要性 DataFrame
        top_n: 顯示前 N 個重要特徵
        save_path: 儲存路徑
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    # 橫向條形圖
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features['importance'], xerr=top_features['std'],
            alpha=0.8, edgecolor='black', linewidth=1.2)
    
    plt.yticks(y_pos, top_features['feature'], fontsize=10)
    plt.xlabel('Importance (R² decrease)', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importance (Permutation)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 特徵重要性圖已儲存: {save_path}")
    plt.close()


def run_deep_learning_pipeline(X_train, X_test, y_train, y_test, feature_names):
    """
    執行完整的深度學習 Pipeline
    
    流程:
        1. 檢測硬體設備（GPU/CPU）
        2. 資料分割（訓練/驗證）
        3. 訓練 Autoencoder 降維
        4. 使用 Encoder 轉換特徵
        5. 訓練 MLP 回歸模型
        6. 評估模型
        7. 繪製視覺化圖表
    
    參數:
        X_train: 訓練特徵
        X_test: 測試特徵
        y_train: 訓練標籤
        y_test: 測試標籤
        feature_names: 特徵名稱
    
    返回:
        encoder: 訓練好的編碼器
        mlp_model: 訓練好的 MLP 模型
        y_pred: 測試集預測值
        metrics: 評估指標
    """
    print("\n" + "🚀 "*20)
    print("開始深度學習模型訓練 Pipeline")
    print("🚀 "*20)
    
    # Step 0: 檢測硬體設備
    device_info = check_gpu_availability()
    
    # ⭐ Step 0.5: 特徵標準化（確保數據已標準化）
    print("\n" + "="*60)
    print("特徵標準化檢查與處理")
    print("="*60)
    
    # 轉換為 numpy array（如果是 DataFrame）
    if hasattr(X_train, 'values'):
        X_train_np = X_train.values
        X_test_np = X_test.values
        feature_cols = X_train.columns if hasattr(X_train, 'columns') else feature_names
    else:
        X_train_np = np.asarray(X_train)
        X_test_np = np.asarray(X_test)
        feature_cols = feature_names
    
    # 檢查是否已經標準化（檢查均值和標準差）
    train_mean = np.mean(X_train_np)
    train_std = np.std(X_train_np)
    
    print(f"  原始特徵統計:")
    print(f"    均值: {train_mean:.4f}")
    print(f"    標準差: {train_std:.4f}")
    print(f"    範圍: [{np.min(X_train_np):.2f}, {np.max(X_train_np):.2f}]")
    
    # 如果數據看起來未標準化（均值不接近0或標準差不接近1），則進行標準化
    if abs(train_mean) > 0.1 or abs(train_std - 1.0) > 0.2:
        print(f"\n  ⚠️  檢測到特徵未標準化，正在進行標準化...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_np)
        X_test_scaled = scaler.transform(X_test_np)
        
        print(f"  ✅ 標準化完成！")
        print(f"    新均值: {np.mean(X_train_scaled):.4f}")
        print(f"    新標準差: {np.std(X_train_scaled):.4f}")
        print(f"    新範圍: [{np.min(X_train_scaled):.2f}, {np.max(X_train_scaled):.2f}]")
        
        # 使用標準化後的數據
        X_train_np = X_train_scaled
        X_test_np = X_test_scaled
    else:
        print(f"  ✅ 特徵已標準化，跳過標準化步驟")
    
    print("="*60)
    
    # 從訓練集中分出驗證集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_np, y_train, test_size=0.15, random_state=42
    )
    
    print(f"\n資料集大小:")
    print(f"  訓練集: {X_train_split.shape}")
    print(f"  驗證集: {X_val.shape}")
    print(f"  測試集: {X_test.shape}")
    
    # Step 1: 訓練 Autoencoder
    encoder, autoencoder, ae_history = train_autoencoder(
        X_train_split, X_val, 
        epochs=70, 
        batch_size=128,
        verbose=1
    )
    
    # Step 2: 使用 Encoder 轉換資料
    print("\n轉換資料至低維空間...")
    X_train_encoded = encoder.predict(X_train_split, verbose=0)
    X_val_encoded = encoder.predict(X_val, verbose=0)
    X_test_encoded = encoder.predict(X_test_np, verbose=0)  # ⭐ 使用標準化後的數據
    print(f"✓ 特徵維度: {X_train_np.shape[1]} → {X_train_encoded.shape[1]}")
    
    # Step 3: 訓練 MLP 模型
    mlp_model, mlp_history = train_mlp_model(
        X_train_encoded, y_train_split,
        X_val_encoded, y_val,
        epochs=200,
        batch_size=64,
        verbose=1
    )
    
    # Step 4: 預測
    print("\n進行預測...")
    y_pred = mlp_model.predict(X_test_encoded, verbose=0)
    
    # Step 5: 評估
    metrics = evaluate_dl_model(y_test, y_pred, model_name="Deep Learning (Autoencoder + MLP)")
    
    # Step 6: 視覺化
    print("\n" + "="*50)
    print("生成視覺化圖表")
    print("="*50)
    
    plot_learning_curves(ae_history, mlp_history)
    plot_predictions_vs_actual(y_test, y_pred, metrics)
    
    # # Step 7: 特徵重要性（可選，較耗時）
    # print("\n是否計算特徵重要性? （這可能需要幾分鐘）")
    # importance_df = calculate_feature_importance(
    #     mlp_model, encoder, X_test_np, y_test,  # ⭐ 使用標準化後的數據
    #     feature_names, n_repeats=5
    # )
    # plot_feature_importance(importance_df, top_n=20)
    
    print("\n" + "✅ "*20)
    print("深度學習 Pipeline 執行完成！")
    print("✅ "*20 + "\n")
    
    return encoder, mlp_model, y_pred, metrics


# ==================== 主程式執行區 ====================
if __name__ == "__main__":
    """
    獨立執行此檔案進行測試
    """
    print("深度學習模型 - 獨立執行模式")
    print("="*60)
    
    # 載入與前處理資料
    from main import preprocess_data
    
    file_path = 'Preprocessing/Data_imputed.csv'
    
    try:
        # 資料處理
        X_train, X_test, y_train, y_test, features = preprocess_data(file_path)
        
        # 執行深度學習 Pipeline
        encoder, mlp_model, y_pred, metrics = run_deep_learning_pipeline(
            X_train, X_test, y_train, y_test, features
        )
        
        print("\n模型訓練與評估完成！")
        print("生成的檔案:")
        print("  - dl_learning_curves.png")
        print("  - dl_predictions_vs_actual.png")
        print("  - dl_feature_importance.png")
        
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()

# result
# ==================================================
# Deep Learning (Autoencoder + MLP) 模型評估結果
# ==================================================
#   MAE  (Mean Absolute Error):     1.8520
#   MSE  (Mean Squared Error):      5.6691
#   RMSE (Root Mean Squared Error): 2.3810
#   R²   (R-squared):               0.2569
#   Spearman Correlation:           0.4237 (p=0.0000e+00)
# ==================================================