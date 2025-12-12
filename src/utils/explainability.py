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
    透過分析 Fusion Layer 的權重，判斷 Cell, Drug, Target, Pathway Branch 的相對重要性。
    """
    print("\n🔍 分析 Branch Importance (Macro-Level)...")
    
    # 1. 找出 Fusion Layer 的權重
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
    # weights shape: (input_dim, units)
    
    # 3. 計算絕對權重總和
    # 動態計算 branches 的維度
    try:
        # 尋找 Concatenate 層
        concat_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                concat_layer = layer
                break
        
        if concat_layer is None:
             print("⚠️ 無法找到 Concatenate 層，無法分解權重。")
             return
        else:
            # Concatenate 層的輸入形狀
            # input_shape may be a property or we need get_input_shape_at(0)
            try:
                input_shapes = concat_layer.input_shape
            except AttributeError:
                try:
                    input_shapes = concat_layer.get_input_shape_at(0)
                except:
                    # Fallback: inspect input tensors directly
                     input_shapes = [t.shape for t in concat_layer.input]

            # Handle cases where input_shape is a list of tuples or TensorShapes
            # If it's a single tuple (unexpected for Concatenate), wrap it
            if not isinstance(input_shapes, list):
                input_shapes = [input_shapes]

            dims = [s[-1] for s in input_shapes]
            
            if len(dims) == 5:
                cell_dim, drug_num_dim, drug_id_dim, target_dim, pathway_dim = dims
            elif len(dims) == 4:
                # Legacy 4-branch (Cell, DrugNum, Target, Pathway) - No Drug ID
                cell_dim, drug_num_dim, target_dim, pathway_dim = dims
                drug_id_dim = 0
            elif len(dims) == 2:
                # Fallback for old model structure
                cell_dim, drug_num_dim = dims
                drug_id_dim, target_dim, pathway_dim = 0, 0, 0
            else:
                print(f"⚠️ Concatenate 層輸入數量異常 ({len(dims)})，無法分析。")
                return
            
            print(f"   ℹ️ Branch Dims - Cell: {cell_dim}, DrugNum: {drug_num_dim}, DrugID: {drug_id_dim}, Target: {target_dim}, Pathway: {pathway_dim}")

    except Exception as e:
        print(f"   ⚠️ 維度偵測失敗 ({e})，跳過分析。")
        return

    # Slicing weights
    # Note: Concatenate layer merges inputs in order of call: 
    # [x_cell, x_drug_num, x_drug_id, x_target, x_pathway]
    current_idx = 0
    
    total_expected_dim = cell_dim + drug_num_dim + drug_id_dim + target_dim + pathway_dim
    if weights.shape[0] != total_expected_dim:
         print(f"   ⚠️ Weights shape mismatch. Expected {total_expected_dim}, got {weights.shape[0]}")
         return

    w_cell = weights[current_idx : current_idx + cell_dim, :]
    current_idx += cell_dim
    
    w_drug_num = weights[current_idx : current_idx + drug_num_dim, :] if drug_num_dim > 0 else np.array([])
    current_idx += drug_num_dim
    
    w_drug_id = weights[current_idx : current_idx + drug_id_dim, :] if drug_id_dim > 0 else np.array([])
    current_idx += drug_id_dim
    
    w_target = weights[current_idx : current_idx + target_dim, :] if target_dim > 0 else np.array([])
    current_idx += target_dim
    
    w_pathway = weights[current_idx : current_idx + pathway_dim, :] if pathway_dim > 0 else np.array([])
    
    # Calculate Importance
    def get_imp(w): return np.sum(np.abs(w))
    
    imp_cell = get_imp(w_cell)
    imp_drug_num = get_imp(w_drug_num)
    imp_drug_id = get_imp(w_drug_id)
    imp_target = get_imp(w_target)
    imp_pathway = get_imp(w_pathway)
    
    total_importance = imp_cell + imp_drug_num + imp_drug_id + imp_target + imp_pathway
    if total_importance == 0: total_importance = 1
    
    pct_cell = (imp_cell / total_importance) * 100
    pct_drug_num = (imp_drug_num / total_importance) * 100
    pct_drug_id = (imp_drug_id / total_importance) * 100
    pct_target = (imp_target / total_importance) * 100
    pct_pathway = (imp_pathway / total_importance) * 100
    
    print(f"   -> Cell: {pct_cell:.1f}%, DrugNum: {pct_drug_num:.1f}%, DrugID: {pct_drug_id:.1f}%, Target: {pct_target:.1f}%, Pathway: {pct_pathway:.1f}%")
    
    # 4. 繪圖
    plt.figure(figsize=(10, 6))
    
    branches = ['Cell Line', 'Drug Numeric', 'Drug ID Emb', 'Target Emb', 'Pathway Emb']
    pcts = [pct_cell, pct_drug_num, pct_drug_id, pct_target, pct_pathway]
    
    # Filter out empty branches
    data_list = []
    for b, p in zip(branches, pcts):
        if p > 0:
            data_list.append({'Branch': b, 'Importance': p})
            
    data = pd.DataFrame(data_list)
    
    if not data.empty:
        ax = sns.barplot(x='Branch', y='Importance', data=data, palette='viridis')
        plt.title('Macro-Level Interpretability: Branch Importance', fontsize=14, fontweight='bold')
        plt.ylabel('Relative Importance (%)')
        plt.ylim(0, 100)
        
        for i, row in data.iterrows():
            ax.text(i, row.Importance + 1, f"{row.Importance:.1f}%", ha='center', fontsize=12, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✅ Branch Importance 圖表已儲存: {save_path}")
    plt.close()

def analyze_shap_values(model, X_train_list, X_test_list, feature_names, save_path_summary):
    """
    使用 SHAP 分析特徵重要性 (Micro-Level Explainability)
    :param X_train_list: List of input arrays [X_cell, X_drug, X_target, X_pathway]
    """
    print("\n🔍 分析 SHAP Values (Micro-Level)...")
    print("   (這可能需要幾分鐘，只會分析前 50 筆測試資料)")
    
    try:
        # 0. Convert to Numpy
        X_train_list = [x.values if hasattr(x, 'values') else x for x in X_train_list]
        X_test_list = [x.values if hasattr(x, 'values') else x for x in X_test_list]

        # --- FIX: Ensure all inputs are at least 2D (N, 1) ---
        # SHAP/Keras often expects (N, 1) for scalars like indices, not (N,)
        def ensure_2d(arr):
            if arr.ndim == 1:
                return arr.reshape(-1, 1)
            return arr
            
        X_train_list = [ensure_2d(x) for x in X_train_list]
        X_test_list = [ensure_2d(x) for x in X_test_list]

        # 1. Prepare Background & Test Data
        # Reduce samples to speed up and avoid OOM
        n_background = 100
        n_test = 500
        
        # Random Sample Indices
        train_indices = np.random.choice(X_train_list[0].shape[0], min(n_background, X_train_list[0].shape[0]), replace=False)
        test_indices = np.arange(min(n_test, X_test_list[0].shape[0]))
        
        # Slice data
        background_data = [x[train_indices] for x in X_train_list]
        test_data = [x[test_indices] for x in X_test_list]
        
        # 2. Explainer
        # Try DeepExplainer (better for TF), Fallback to KernelExplainer (Robust)
        shap_values = None
        explainer_type = "Deep"

        try:
            # SHAP DeepExplainer needs the model and background data
            # DeepExplainer fails if any input has 0 columns (shape=(N, 0)) e.g. empty drug features
            has_empty_input = any(x.shape[1] == 0 for x in background_data)
            if has_empty_input:
                raise ValueError("Input has 0 dimensions (e.g. empty drug features), DeepExplainer cannot handle this.")

            explainer = shap.DeepExplainer(model, background_data)
            shap_values = explainer.shap_values(test_data)
        
        except Exception as e:
            print(f"   ⚠️ DeepExplainer failed ({e}). trying KernelExplainer (slower but robust)...")
            explainer_type = "Kernel"
            
            # KernelExplainer treats model as black box function f(x) -> y
            # We must flatten list of inputs into one matrix X_combined, but handle 0-dim inputs carefully.
            
            # Calculate split points based on dimensions
            dims = [x.shape[1] for x in background_data]
            
            # If a dim is 0 (empty input), it contributes 0 columns to hstack.
            # We must record where it 'should' be to reconstruct the list for the model.
            
            # Filter out 0-dim inputs for KernelExplainer's data matrix (since we can't hstack 0 cols effectively with some tools or it's useless)
            # Actually hstack works with (N, 0). (N, 10) + (N, 0) -> (N, 10).
            # But we need to know how to split back.
            
            split_points = np.cumsum(dims)[:-1] # [10, 10, 11] for dims [10, 0, 1, 1]
            
            def model_wrapper(X_combined):
                # X_combined is (N, total_non_zero_features)
                # Split back into chunks
                xs = np.split(X_combined, split_points, axis=1)
                
                # Reshape to match original expectations
                # np.split returns arrays. If a chunk was 0-width, it will have shape (N, 0).
                # This should be compatible with Keras model inputs.
                return model.predict(xs, verbose=0).flatten()
            
            # Combine background data
            X_bg_combined = np.hstack(background_data)
            X_test_combined = np.hstack(test_data)
            
            # Check if we have ANY features at all
            if X_bg_combined.shape[1] == 0:
                 print("   ⚠️ Total features dim is 0. Cannot run SHAP.")
                 return

            # Summarize background for KernelExplainer speed using K-Means
            # Use smaller K if background is small
            k_val = min(10, X_bg_combined.shape[0])
            try:
                X_bg_summary = shap.kmeans(X_bg_combined, k_val) 
            except:
                X_bg_summary = X_bg_combined # Fallback if kmeans fails
            
            explainer = shap.KernelExplainer(model_wrapper, X_bg_summary)
            
            # Run SHAP (can be slow, so limit samples)
            shap_values = explainer.shap_values(X_test_combined, nsamples=100)
            
            # Compatibility: KernelExplainer returns array (for single output), expected list of arrays matching inputs
            if isinstance(shap_values, list): 
                shap_values = shap_values[0] # Regression returns [values]
            
            # shap_values is (N_test, Total_Features). We need to split back to list of (N_test, Dim_i)
            shap_values = np.split(shap_values, split_points, axis=1)

        
        # 3. Process SHAP values for Plotting (Common for both Deep and Kernel)
        # shap_values is list of arrays (one for each input head).
        
        # Flatten and combine
        combined_shap = []
        combined_data = []
        combined_names = []
        
        # --- 重新組合特徵名稱，確保順序一致 ---
        def _add_branch(shap_arr, data_arr, name_key):
            """安全地把一個分支加入 combined 列表，確保長度匹配。"""
            if shap_arr is None or data_arr is None:
                return
            # 1. 轉成 2D
            if len(shap_arr.shape) == 3:
                shap_arr = shap_arr.squeeze(-1)
            # 2. 檢查維度
            if shap_arr.shape[1] != data_arr.shape[1]:
                print(f"⚠️ {name_key} SHAP 與資料維度不符，跳過此分支。")
                return
            # 3. 加入
            combined_shap.append(shap_arr)
            combined_data.append(data_arr)
            # 只取對應長度的名稱
            names = feature_names.get(name_key, [])
            if len(names) != data_arr.shape[1]:
                # print(f"⚠️ {name_key} 名稱長度與特徵數不符，使用前 {data_arr.shape[1]} 個名稱。")
                names = names[: data_arr.shape[1]]
            combined_names.extend(names)

        # Cell
        _add_branch(shap_values[0], test_data[0], 'cell')
        # Drug Numeric (Index 1)
        if len(shap_values) > 1:
            _add_branch(shap_values[1], test_data[1], 'drug_numeric')
            
        # Drug ID (Index 2) - Skip or include if needed (usually not meaningful for summary plot as it is just an index)
        # We skip index 2 for the summary plot logic usually.

        # Target (Index 3)
        if len(shap_values) > 3:
            _add_branch(shap_values[3], test_data[3], 'target')

        # Pathway (Index 4)
        if len(shap_values) > 4:
            _add_branch(shap_values[4], test_data[4], 'pathway')

        # Combine
        if not combined_shap:
            print("⚠️ No SHAP values to plot.")
            return

        final_shap_values = np.hstack(combined_shap)
        final_test_data = np.hstack(combined_data)
        
        print(f"   -> SHAP 計算完成 ({explainer_type}Explainer)。Shape: {final_shap_values.shape}")
        
        # 4. Plot
        plt.figure(figsize=(10, 8)) # Adjusted size for 20 features
        shap.summary_plot(
            final_shap_values, 
            final_test_data, 
            feature_names=combined_names,
            show=False,
            plot_type="dot",
            max_display=20 # Optimized for PPT visibility
        )
        plt.title('SHAP Value Summary (Global)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path_summary, dpi=300, bbox_inches='tight')
        print(f"✅ SHAP Summary Plot 已儲存: {save_path_summary}")
        plt.close()
        
    except Exception as e:
        print(f"❌ SHAP 分析失敗: {e}")
        import traceback
        traceback.print_exc()
