import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import tensorflow as tf

# 設定全域繪圖風格
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

# Mako Palette Colors
MAKO_PALETTE = sns.color_palette("mako", as_cmap=False)
MAKO_COLOR_1 = MAKO_PALETTE[1] # Darker Blue/Green
MAKO_COLOR_2 = MAKO_PALETTE[4] # Lighter Blue/Green

def analyze_branch_importance(model, save_path):
    """
    分析雙塔模型的 Branch Importance (Macro-Level Explainability)
    透過分析 Fusion Layer 的權重，判斷 Cell Branch 和 Drug Branch 的相對重要性。
    """
    print("\n🔍 分析 Branch Importance (Macro-Level)...")
    
    # 1. 找出 Fusion Layer 的權重
    # 我們在模型定義中已經將該層命名為 'fusion_dense'
    fusion_dense_layer = None
    
    for layer in model.layers:
        if layer.name == 'fusion_dense':
            fusion_dense_layer = layer
            break
    
    if fusion_dense_layer is None:
        print("⚠️ 無法自動識別 Fusion Layer，跳過 Branch Importance 分析。")
        return

    # 2. 提取權重
    weights, biases = fusion_dense_layer.get_weights()
    # weights shape: (192, 128)
    
    # 3. 計算絕對權重總和
    # 動態計算 Cell Branch 和 Drug Branch 的維度
    try:
        # 尋找 Concatenate 層以確定輸入維度
        concat_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                concat_layer = layer
                break
        
        if concat_layer is None:
             # Fallback: 使用 Config (如果不幸找不到 Concatenate 層)
             from src.config import ModelConfig
             # 這裡假設 ModelConfig.BEST_DL_PARAMS 正確反映了當前模型架構
             # 注意：這可能不準確如果模型是用不同參數訓練的，但比硬編碼 128 好
             cell_units = ModelConfig.BEST_DL_PARAMS['cell_units']
             cell_dim = cell_units[-1] if isinstance(cell_units, list) else cell_units
             print(f"   ⚠️ 無法找到 Concatenate 層，嘗試使用 Config 推斷 Cell Dim: {cell_dim}")
        else:
            # Concatenate 層的輸入形狀通常是 list of tuples
            # input_shape: [(None, cell_dim), (None, drug_dim)]
            input_shapes = concat_layer.input_shape
            cell_dim = input_shapes[0][-1]
            print(f"   ℹ️ 自動偵測到 Cell Branch Output Dim: {cell_dim}")

    except Exception as e:
        print(f"   ⚠️ 維度偵測失敗 ({e})，將使用預設值 128 (可能不準確)")
        cell_dim = 128

    # 前 cell_dim 個來自 Cell Branch，其餘來自 Drug Branch
    cell_weights = weights[:cell_dim, :]
    drug_weights = weights[cell_dim:, :]
    
    cell_importance = np.sum(np.abs(cell_weights))
    drug_importance = np.sum(np.abs(drug_weights))
    
    total_importance = cell_importance + drug_importance
    cell_pct = (cell_importance / total_importance) * 100
    drug_pct = (drug_importance / total_importance) * 100
    
    print(f"   -> Cell Branch Importance: {cell_pct:.2f}%")
    print(f"   -> Drug Branch Importance: {drug_pct:.2f}%")
    
    # 4. 繪圖
    plt.figure(figsize=(8, 6))
    data = pd.DataFrame({
        'Branch': ['Cell Line Features', 'Drug Features'],
        'Importance': [cell_pct, drug_pct]
    })
    
    ax = sns.barplot(x='Branch', y='Importance', data=data, palette=[MAKO_COLOR_2, MAKO_COLOR_1])
    plt.title('Macro-Level Interpretability: Which Branch Matters More?', fontsize=14, fontweight='bold')
    plt.ylabel('Relative Importance (%)')
    plt.ylim(0, 100)
    
    # Add labels
    for i, v in enumerate([cell_pct, drug_pct]):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Branch Importance 圖表已儲存: {save_path}")
    plt.close()

def analyze_shap_values(model, X_train_list, X_test_list, feature_names, save_path_summary):
    """
    使用 SHAP 分析特徵重要性 (Micro-Level Explainability)
    :param X_train_list: [X_cell_train, X_drug_train]
    :param X_test_list: [X_cell_test, X_drug_test]
    :param feature_names: {'cell': [...], 'drug': [...]}
    """
    print("\n🔍 分析 SHAP Values (Micro-Level)...")
    print("   (這可能需要幾分鐘，只會分析前 100 筆測試資料)")
    
    try:
        # 0. 確保輸入為 Numpy Array (避免 DataFrame Indexing 問題)
        X_train_list = [x.values if hasattr(x, 'values') else x for x in X_train_list]
        X_test_list = [x.values if hasattr(x, 'values') else x for x in X_test_list]

        # 1. 準備背景資料 (Background Data)
        # SHAP 需要背景資料來估計期望值。為了速度，我們從訓練集隨機取樣 100 筆。
        n_background = 100
        n_test = 100
        
        # 隨機取樣索引
        train_indices = np.random.choice(X_train_list[0].shape[0], n_background, replace=False)
        test_indices = np.arange(min(n_test, X_test_list[0].shape[0]))
        
        # 構建背景資料列表 [X_cell_bg, X_drug_bg]
        background_data = [
            X_train_list[0][train_indices],
            X_train_list[1][train_indices]
        ]
        
        # 構建測試資料列表 [X_cell_test, X_drug_test]
        test_data = [
            X_test_list[0][test_indices],
            X_test_list[1][test_indices]
        ]
        
        # 2. 建立 Explainer
        # 使用 GradientExplainer (適用於 TF/Keras)
        # 注意: DeepExplainer 有時在 TF2.x會有相容性問題，GradientExplainer 通常較穩
        explainer = shap.GradientExplainer(model, background_data)
        
        # 3. 計算 SHAP Values
        shap_values = explainer.shap_values(test_data)
        
        # shap_values 是一個 list，包含兩個 array (對應兩個輸入)
        # 每個 array 的 shape 是 (n_test, n_features, 1) (因為輸出是 1 維)
        # 我們需要把它們壓平並合併
        
        # 處理 Cell SHAP
        shap_cell = shap_values[0]
        if isinstance(shap_cell, list): shap_cell = shap_cell[0] # 有時會多一層 list
        if len(shap_cell.shape) == 3: shap_cell = shap_cell.squeeze(-1) # (100, n_cell, 1) -> (100, n_cell)
        
        # 處理 Drug SHAP
        shap_drug = shap_values[1]
        if isinstance(shap_drug, list): shap_drug = shap_drug[0]
        if len(shap_drug.shape) == 3: shap_drug = shap_drug.squeeze(-1)
        
        # 合併 SHAP values 和 Feature Names
        combined_shap_values = np.hstack([shap_cell, shap_drug])
        combined_feature_names = feature_names['cell'] + feature_names['drug']
        combined_test_data = np.hstack([test_data[0], test_data[1]])
        
        print(f"   -> SHAP 計算完成。Shape: {combined_shap_values.shape}")
        
        # 4. 繪製 Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            combined_shap_values, 
            combined_test_data, 
            feature_names=combined_feature_names,
            show=False,
            plot_type="dot",
            max_display=15
        )
        plt.title('SHAP Value Summary (Top 15 Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path_summary, dpi=300, bbox_inches='tight')
        print(f"✅ SHAP Summary Plot 已儲存: {save_path_summary}")
        plt.close()
        
    except Exception as e:
        print(f"❌ SHAP 分析失敗: {e}")
        import traceback
        traceback.print_exc()
