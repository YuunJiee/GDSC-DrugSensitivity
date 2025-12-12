import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_selection import VarianceThreshold

def load_raw_data(file_path):
    """
    讀取原始資料
    """
    print(f"📂 讀取原始資料: {file_path}")
    return pd.read_csv(file_path)

def load_and_preprocess_baseline(file_path, variance_threshold=0.01, include_ids=True):
    """
    基線模型 (Baseline) 的資料前處理
    使用 One-Hot Encoding 並加入 VarianceThreshold 特徵選擇
    :param include_ids: 是否保留 DRUG_ID 和 COSMIC_ID (True=高分模式, False=泛化模式)
    """
    print(f"Step 1: 正在讀取資料並進行前處理 (Baseline): {file_path}...")
    print(f"   -> 模式: {'保留 ID 特徵 (高分模式)' if include_ids else '移除 ID 特徵 (泛化模式)'}")

    df = pd.read_csv(file_path)
    target = 'LN_IC50'
    
    # 與 DL 保持一致：移除目標變數缺失值
    df = df.dropna(subset=[target])

    # 基礎移除欄位 (Data Leakage)
    drop_cols = [
        'CELL_LINE_NAME', 'DRUG_NAME', 
        'AUC', 'Z_SCORE', 'RMSE',
        'NLME_RESULT_ID', 'NLME_CURVE_ID', 'SANGER_MODEL_ID'
    ]
    
    # 如果不包含 ID，則額外移除 ID 欄位
    if not include_ids:
        drop_cols.extend(['DRUG_ID', 'COSMIC_ID'])
        print("   -> 已設定移除 DRUG_ID 與 COSMIC_ID")
        
    # 移除無意義的 Y/N 資料存在標記 (Data Availability Flags) - 與 DL 一致
    drop_cols.extend([
        'Gene Expression', 'CNA', 'Methylation', 'Drug Response', 
        'Exome mutation', 'Whole Genome Sequencing (WGS)'
    ])

    # 分割訓練/測試集
    # Method 2: Blind Cell Split (Prevent Data Leakage) - 與 DL 保持一致
    if 'COSMIC_ID' in df.columns:
        print("   -> 模式: Blind Cell Split (依據 COSMIC_ID 切分)")
        from sklearn.model_selection import GroupShuffleSplit
        
        # 即使我們上面 drop 了 COSMIC_ID，我們可以用原始 df 來取得 Group
        # 注意：我們之前 drop 了 COSMIC_ID (line 39), 但如果 include_ids=False, 我們需要它來 split
        # 修正：line 39 已經 drop 了。我們可以用原始 df 的 index 來對應，或者在 drop 之前先取出來做 group
        # 更好作法：不要在 line 39 drop COSMIC_ID，而是等到 split 完後再從 X 中移除
        
        pass # Will implement logic below
    
    # Re-logic: We need to handle COSMIC_ID carefully.
    # Let's refactor: Keep COSMIC_ID until split, then drop.
    pass

    # New implementation logic
    # 1. Drop non-essential cols but KEEP COSMIC_ID for splitting
    cols_to_drop_early = [c for c in drop_cols if c not in ['COSMIC_ID', 'DRUG_ID']]
    # If include_ids=False, we explicitly want to drop them from FEATURES, but we need COSMIC_ID for SPLITTING.
    
    df_cleaned = df.drop(columns=[c for c in cols_to_drop_early if c in df.columns])
    
    # One-Hot Encoding
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
    
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    
    # Split
    if 'COSMIC_ID' in df.columns:
         print("   -> 模式: Blind Cell Split (依據 COSMIC_ID 切分)")
         from sklearn.model_selection import GroupShuffleSplit
         gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
         groups = df['COSMIC_ID']
         
         train_idx, test_idx = next(gss.split(X, y, groups=groups))
         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
         # Fallback
         stratify_col = None
         if 'GDSC Tissue descriptor 1' in df_cleaned.columns:
             stratify_col = df_cleaned['GDSC Tissue descriptor 1'].fillna('Unknown')
         X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_col
         )
         
    # Now remove IDs from X if include_ids=False
    if not include_ids:
        # COSMIC_ID and DRUG_ID might be in X
        cols_to_remove = [c for c in ['COSMIC_ID', 'DRUG_ID'] if c in X_train.columns]
        if cols_to_remove:
            X_train = X_train.drop(columns=cols_to_remove)
            X_test = X_test.drop(columns=cols_to_remove)
            print(f"   -> Split 後已移除 ID 欄位: {cols_to_remove}")
    
    # --- 優化：特徵選擇 (移除低變異數特徵) ---
    print(f"   -> 執行特徵選擇 (VarianceThreshold={variance_threshold})...")
    selector = VarianceThreshold(threshold=variance_threshold)
    
    # 僅在訓練集上 fit
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # 獲取保留的特徵名稱 (Use X_train.columns as it matches the fitted data)
    feature_names = X_train.columns[selector.get_support()]
    print(f"      保留特徵數: {len(feature_names)} / {X_train.shape[1]}")
    
    # 轉換回 DataFrame 以保持欄位名稱 (對 XGBoost 重要)
    X_train = pd.DataFrame(X_train_selected, columns=feature_names, index=X_train.index)
    X_test = pd.DataFrame(X_test_selected, columns=feature_names, index=X_test.index)

    # 確保 DRUG_ID 和 COSMIC_ID 被視為數值特徵 (如果它們在特徵選擇中被保留)
    # 這裡不需要額外操作，因為它們原本就是數值

    print(f"   -> 資料形狀: X_train={X_train.shape}, X_test={X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_names

def load_and_preprocess_dl(file_path, include_ids=True):
    """
    深度學習模型 (Deep Learning) 的資料前處理
    使用 Label Encoding，並將特徵拆分為 Cell Line 和 Drug 兩組
    :param include_ids: 是否保留 DRUG_ID 和 COSMIC_ID
    """
    print(f"   -> 模式: {'保留 ID 特徵' if include_ids else '移除 ID 特徵'}")
    
    df = pd.read_csv(file_path)
    
    print(f"   原始資料形狀: {df.shape}")
    
    # 移除目標變數缺失值
    df_clean = df.dropna(subset=['LN_IC50'])
    
    # 定義特徵群組
    # 1. 藥物特徵 (ID vs Numeric)
    # DRUG_ID 特殊處理：如果有 ID，將其獨立出來做 Embedding，不放在 numeric 中
    drug_numeric_cols = [] 
    # 如果有其他藥物數值特徵 (例如分子量)，但在這份資料集似乎沒有，除非有額外 merge
    # 假設目前沒有其他藥物特徵，drug_numeric_cols 為空 (若 No_IDs) 或保留非 ID 欄位
    
    # 定義 Embedding/Multi-hot 特徵
    target_col = 'TARGET'
    pathway_col = 'TARGET_PATHWAY'
    
    # 基礎移除欄位
    exclude_cols = [
        'CELL_LINE_NAME', 'DRUG_NAME', 
        'AUC', 'Z_SCORE', 'RMSE', 
        'NLME_RESULT_ID', 'NLME_CURVE_ID', 'SANGER_MODEL_ID',
        'LN_IC50', target_col, pathway_col,
        # 移除無意義的 Y/N 資料存在標記 (Data Availability Flags)
        'Gene Expression', 'CNA', 'Methylation', 'Drug Response', 
        'Exome mutation', 'Whole Genome Sequencing (WGS)'
    ]
    # DRUG_ID 不在 exclude_cols 中 (除非 No_IDs)，但也不算 cell feature
    if not include_ids:
        exclude_cols.append('DRUG_ID')
        exclude_cols.append('COSMIC_ID')
    
    # ---------------------------------------------------------
    # 1. 處理細胞特徵 (Cell Features) - One-Hot Encoding
    # ---------------------------------------------------------
    print("\n🔄 處理細胞特徵 (One-Hot Encoding)...")
    
    # 潛在的細胞特徵欄位 = 所有欄位 - 排除欄位 - (Target/Pathway 已排除) - (DRUG_ID 若存在)
    potential_cell_cols = [c for c in df_clean.columns if c not in exclude_cols]
    
    # 確保 DRUG_ID 和 COSMIC_ID 不被算作細胞特徵 (即使 include_ids=True)
    if 'DRUG_ID' in potential_cell_cols: potential_cell_cols.remove('DRUG_ID')
    if 'COSMIC_ID' in potential_cell_cols: potential_cell_cols.remove('COSMIC_ID') # COSMIC_ID 僅用於 Split，不作為特徵輸入? 
    # User 通常不希望 COSMIC_ID 進入模型 (Blind Cell)，除非 embedding。
    # 這裡假設 COSMIC_ID 不入模。

    X_cell_raw = df_clean[potential_cell_cols]
    
    # 自動識別 object 類型的欄位進行 One-Hot
    categorical_cell_cols = X_cell_raw.select_dtypes(include=['object']).columns.tolist()
    
    # 使用 get_dummies 進行 One-Hot
    X_cell_processed = pd.get_dummies(X_cell_raw, columns=categorical_cell_cols, drop_first=False)
    X_cell_processed = X_cell_processed.astype(float)
    
    print(f"   -> 細胞特徵處理完成。維度: {X_cell_processed.shape}")

    # ---------------------------------------------------------
    # 2. 處理 Target 特徵 (Multi-Hot Encoding)
    # ---------------------------------------------------------
    print("🔄 處理 Target 特徵 (Multi-Hot Encoding)...")
    targets = df_clean[target_col].fillna('Unknown').astype(str)
    targets_split = targets.apply(lambda x: [t.strip() for t in x.split(',')])
    mlb_target = MultiLabelBinarizer()
    X_target_encoded = mlb_target.fit_transform(targets_split)
    print(f"   -> Target 編碼完成。維度: {X_target_encoded.shape}")
    
    # ---------------------------------------------------------
    # 3. 處理 Pathway 特徵 (One-Hot Encoding)
    # ---------------------------------------------------------
    print("🔄 處理 Pathway 特徵 (One-Hot Encoding)...")
    pathways = df_clean[pathway_col].fillna('Unknown').astype(str)
    X_pathway_encoded = pd.get_dummies(pathways).astype(float).values
    print(f"   -> Pathway 編碼完成。維度: {X_pathway_encoded.shape}")

    # ---------------------------------------------------------
    # 4. 處理藥物特徵 (Numeric & ID Embedding)
    # ---------------------------------------------------------
    X_drug_numeric = pd.DataFrame() # 暫無其他數值特徵
    # 如果未來有物理化學性質，在這裡加入
    # 目前 GDSC 資料集主要只有 ID 和 Target/Pathway
    
    # 填補空的 numeric (避免 shape (N, 0) 造成問題，但我們 Model 已經有處理 dummy input)
    # 為了方便 split，還是生成一個 (N, 0) 的 DF? 
    # 讓它保持 (N, 0)，Model 會處理。
    X_drug_numeric = pd.DataFrame(index=df_clean.index) # Empty DF with correct index
    
    # 處理 DRUG_ID Embedding
    X_drug_id = None
    drug_vocab_size = 0
    drug_le = None
    
    if include_ids and 'DRUG_ID' in df_clean.columns:
        print("🔄 處理 Drug ID (Label Encoding for Embedding)...")
        drug_le = LabelEncoder()
        # 轉成字串再編碼比較保險，或者確保是 int
        drug_ids = df_clean['DRUG_ID'].astype(str)
        X_drug_id = drug_le.fit_transform(drug_ids).reshape(-1, 1) # (N, 1)
        drug_vocab_size = len(drug_le.classes_)
        print(f"   -> Drug ID 編碼完成。Vocab Size: {drug_vocab_size}")
    else:
        # No_IDs 模式：給一個全 0 的 dummy ID，並且 vocab_size=0 表示不使用 Embedding
        X_drug_id = np.zeros((len(df_clean), 1), dtype=int)
        drug_vocab_size = 0

    # ---------------------------------------------------------
    # 5. 整合與分割
    # ---------------------------------------------------------
    y = df_clean['LN_IC50']
    
    feature_names = {
        'cell': X_cell_processed.columns.tolist(),
        'drug_numeric': [],
        'target': list(mlb_target.classes_),
        'pathway': list(pd.get_dummies(pathways).columns),
        'drug_vocab_size': drug_vocab_size
    }
    
    # 回傳 dimensions
    # dims = (target_dim, pathway_dim, drug_vocab_size)
    dims = (X_target_encoded.shape[1], X_pathway_encoded.shape[1], drug_vocab_size)
    
    # Arrays to split
    # [Cell, DrugNum, DrugID, Target, Pathway, y]
    arrays = [X_cell_processed, X_drug_numeric, X_drug_id, X_target_encoded, X_pathway_encoded, y]
    
    # Split Strategy
    stratify_col = None
    
    # Method 2: Blind Cell Split (Prevent Data Leakage)
    if 'COSMIC_ID' in df_clean.columns:
        print("   -> 模式: Blind Cell Split (依據 COSMIC_ID 切分)")
        from sklearn.model_selection import GroupShuffleSplit
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        groups = df_clean['COSMIC_ID']
        
        train_idx, test_idx = next(gss.split(X_cell_processed, y, groups=groups))
        
        def split_array(arr, tr_idx, te_idx):
            if hasattr(arr, 'iloc'): return arr.iloc[tr_idx], arr.iloc[te_idx]
            return arr[tr_idx], arr[te_idx]
            
        X_cell_tr, X_cell_te = split_array(X_cell_processed, train_idx, test_idx)
        X_drug_num_tr, X_drug_num_te = split_array(X_drug_numeric, train_idx, test_idx)
        X_drug_id_tr, X_drug_id_te = split_array(X_drug_id, train_idx, test_idx)
        X_target_tr, X_target_te = split_array(X_target_encoded, train_idx, test_idx)
        X_pathway_tr, X_pathway_te = split_array(X_pathway_encoded, train_idx, test_idx)
        y_tr, y_te = split_array(y, train_idx, test_idx)
        
        # Verify
        train_cells = set(df_clean.iloc[train_idx]['COSMIC_ID'])
        test_cells = set(df_clean.iloc[test_idx]['COSMIC_ID'])
        overlap = train_cells.intersection(test_cells)
        print(f"      [Check] Train Cells: {len(train_cells)}, Test Cells: {len(test_cells)}, Overlap: {len(overlap)}")
        
    else:
        print("   ⚠️ 警告: 找不到 COSMIC_ID，退回 Stratified Split")
        if 'GDSC Tissue descriptor 1' in df_clean.columns:
            stratify_col = df_clean['GDSC Tissue descriptor 1'].fillna('Unknown')
            
        split_res = train_test_split(*arrays, test_size=0.2, random_state=42, stratify=stratify_col)
        # Unpack 6 pairs
        X_cell_tr, X_cell_te = split_res[0], split_res[1]
        X_drug_num_tr, X_drug_num_te = split_res[2], split_res[3]
        X_drug_id_tr, X_drug_id_te = split_res[4], split_res[5]
        X_target_tr, X_target_te = split_res[6], split_res[7]
        X_pathway_tr, X_pathway_te = split_res[8], split_res[9]
        y_tr, y_te = split_res[10], split_res[11]

    print(f"✅ 資料分割完成 (DL):")
    print(f"   X_cell_train: {X_cell_tr.shape}")
    print(f"   Drug ID Vocab: {drug_vocab_size}")
    
    encoders = {'mlb_target': mlb_target, 'drug_le': drug_le} 
    
    # Return 5 input arrays per set now
    return (X_cell_tr, X_drug_num_tr, X_drug_id_tr, X_target_tr, X_pathway_tr), \
           (X_cell_te, X_drug_num_te, X_drug_id_te, X_target_te, X_pathway_te), \
           y_tr, y_te, \
           feature_names, encoders, dims
