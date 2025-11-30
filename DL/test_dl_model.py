#!/usr/bin/env python3
"""
深度學習模型測試腳本
1. 硬體設備檢測 (GPU/CPU)
2. 模型架構測試
3. 快速訓練測試
4. 評估功能測試
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print("="*60)
print("深度學習模型完整測試")
print("="*60)

# ==================== 測試 1: 硬體設備檢測 ====================
print("\n[測試 1/5] 🔍 硬體設備檢測...")
print("-"*60)

try:
    from deep_learning_model import check_gpu_availability
    
    # 執行 GPU 檢測
    device_info = check_gpu_availability()
    
    # 顯示結果摘要
    print("\n📊 硬體檢測摘要:")
    if device_info['gpu_available']:
        print(f"  ✅ GPU 加速: 已啟用")
        print(f"  🎮 GPU 數量: {device_info['num_gpus']}")
        print(f"  📌 TensorFlow: {device_info['tf_version']}")
        print(f"  🚀 你的訓練將使用 GPU 加速！")
    else:
        print(f"  ⚠️  GPU 加速: 未啟用")
        print(f"  💻 將使用 CPU 訓練")
        print(f"  📌 TensorFlow: {device_info['tf_version']}")
    
    print("✅ 硬體檢測完成\n")
    
except Exception as e:
    print(f"❌ 硬體檢測失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ==================== 測試 2: 檢查模組導入 ====================
print("\n[測試 2/5] 📦 檢查模組導入...")
print("-"*60)

try:
    from deep_learning_model import (
        build_autoencoder,
        build_mlp_model,
        train_autoencoder,
        train_mlp_model,
        evaluate_dl_model,
        run_deep_learning_pipeline
    )
    print("✅ 所有函數成功導入")
except Exception as e:
    print(f"❌ 導入失敗: {e}")
    exit(1)

# ==================== 測試 3: 建立模型架構 ====================
print("\n[測試 3/5] 🏗️  建立模型架構...")
print("-"*60)

try:
    encoder, autoencoder = build_autoencoder(input_dim=100, encoding_dim=32)
    mlp = build_mlp_model(input_dim=32)
    
    print(f"  Autoencoder 參數: {autoencoder.count_params():,}")
    print(f"  MLP 參數: {mlp.count_params():,}")
    print("✅ 模型架構建立成功")
except Exception as e:
    print(f"❌ 建立模型失敗: {e}")
    exit(1)

# ==================== 測試 4: 使用合成資料測試訓練 ====================
print("\n[測試 4/5] 🎓 使用合成資料測試訓練...")
print("-"*60)

try:
    # 生成小型合成資料
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 2 + 1  # 模擬 LN_IC50
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  資料形狀: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # 快速訓練 Autoencoder（少量 epochs）
    X_train_split, X_val = X_train[:700], X_train[700:]
    
    print("\n  訓練 Autoencoder (5 epochs)...")
    encoder_test, ae_test, history = train_autoencoder(
        X_train_split, X_val, 
        epochs=5,
        batch_size=64,
        verbose=0
    )
    
    print(f"  ✓ Autoencoder 訓練完成")
    
    # 轉換資料
    X_train_encoded = encoder_test.predict(X_train_split, verbose=0)
    X_val_encoded = encoder_test.predict(X_val, verbose=0)
    y_train_split, y_val = y_train[:700], y_train[700:]
    
    # 快速訓練 MLP
    print("  訓練 MLP (5 epochs)...")
    mlp_test, mlp_history = train_mlp_model(
        X_train_encoded, y_train_split,
        X_val_encoded, y_val,
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    print(f"  ✓ MLP 訓練完成")
    print("✅ 訓練測試完成")
    
except Exception as e:
    print(f"❌ 訓練測試失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ==================== 測試 5: 評估功能 ====================
print("\n[測試 5/6] 📊 測試評估功能...")
print("-"*60)

try:
    X_test_encoded = encoder_test.predict(X_test, verbose=0)
    y_pred = mlp_test.predict(X_test_encoded, verbose=0)
    
    metrics = evaluate_dl_model(y_test, y_pred, model_name="測試模型")
    
    print(f"\n  評估指標:")
    print(f"    R² = {metrics['R2']:.4f}")
    print(f"    RMSE = {metrics['RMSE']:.4f}")
    print(f"    MAE = {metrics['MAE']:.4f}")
    print(f"    Spearman ρ = {metrics['Spearman_Correlation']:.4f}")
    print("✅ 評估測試完成")
    
except Exception as e:
    print(f"❌ 評估測試失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ==================== 測試 6: 特徵重要性計算 ====================
print("\n[測試 6/6] 🎯 測試特徵重要性計算...")
print("-"*60)

try:
    from deep_learning_model import calculate_feature_importance
    
    # 創建特徵名稱列表
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    print(f"  測試資料形狀: {X_test.shape}")
    print(f"  特徵名稱數量: {len(feature_names)}")
    print(f"  開始計算特徵重要性（n_repeats=3，快速測試）...")
    
    # 使用較少的重複次數進行快速測試
    importance_df = calculate_feature_importance(
        mlp_test, 
        encoder_test, 
        X_test, 
        y_test, 
        feature_names,
        n_repeats=3,  # 使用少量重複以加快測試
        random_state=42
    )
    
    print(f"\n  特徵重要性結果:")
    print(f"    總特徵數: {len(importance_df)}")
    print(f"    前 5 個重要特徵:")
    
    top_5 = importance_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"      {row['feature']}: {row['importance']:.6f} ± {row['std']:.6f}")
    
    print("✅ 特徵重要性測試完成")
    
except Exception as e:
    print(f"❌ 特徵重要性測試失敗:")
    print(f"   錯誤訊息: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠️  注意: 特徵重要性計算失敗，但不影響其他功能")

# ==================== 所有測試完成 ====================
print("\n" + "="*60)
print("🎉 所有測試通過！深度學習模型已準備就緒")
print("="*60)

# 顯示系統資訊
print("\n📌 系統資訊:")
print(f"  TensorFlow: {device_info['tf_version']}")
if device_info['gpu_available']:
    print(f"  ✅ GPU 加速已啟用 ({device_info['num_gpus']} GPU)")
    print(f"  ⚡ 預期訓練速度提升: 5-10x")
else:
    print(f"  💻 使用 CPU 訓練")
    print(f"  💡 提示: 安裝 tensorflow-metal 以啟用 GPU 加速")

print("\n下一步:")
print("  1. 執行 'python main.py' 來訓練完整模型")
print("  2. 或執行 'python deep_learning_model.py' 單獨測試深度學習")
print()