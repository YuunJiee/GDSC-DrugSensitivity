import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df_cleaned = df.drop(columns=existing_drop_cols)
    print(f"   -> 已移除欄位: {existing_drop_cols}")

    # One-Hot Encoding
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

    # 分割 X, y
    X = df_processed.drop(columns=[target])
    y = df_processed[target]
    
    # 分割訓練/測試集
    # 準備 Stratified Sampling 的標籤
    stratify_col = None
    if 'GDSC Tissue descriptor 1' in df_cleaned.columns:
        stratify_col = df_cleaned['GDSC Tissue descriptor 1'].fillna('Unknown')
        print("   -> 已啟用 Stratified Sampling (依據 GDSC Tissue descriptor 1)")
    
    # 分割訓練/測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_col
    )
    
    # --- 優化：特徵選擇 (移除低變異數特徵) ---
    print(f"   -> 執行特徵選擇 (VarianceThreshold={variance_threshold})...")
    selector = VarianceThreshold(threshold=variance_threshold)
    
    # 僅在訓練集上 fit
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # 獲取保留的特徵名稱
    feature_names = X.columns[selector.get_support()]
    print(f"      保留特徵數: {len(feature_names)} / {X.shape[1]}")
    
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
    print(f"\n📂 正在讀取資料 (Deep Learning): {file_path}")
    print(f"   -> 模式: {'保留 ID 特徵' if include_ids else '移除 ID 特徵'}")
    
    df = pd.read_csv(file_path)
    
    print(f"   原始資料形狀: {df.shape}")
    
    # 移除目標變數缺失值
    df_clean = df.dropna(subset=['LN_IC50'])
    
    # 處理類別型特徵
    print("\n🔄 處理類別型特徵...")
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # 基礎移除欄位 (不包含在訓練特徵中，但可能用於 ID)
    exclude_cols = [
        'CELL_LINE_NAME', 'DRUG_NAME', 
        'AUC', 'Z_SCORE', 'RMSE', 
        'NLME_RESULT_ID', 'NLME_CURVE_ID', 'SANGER_MODEL_ID'
    ]
    
    # 對類別型特徵進行 Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        if col not in exclude_cols and col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
            
    # 定義特徵群組
    # 1. 藥物特徵
    drug_feature_cols = ['TARGET', 'TARGET_PATHWAY']
    if include_ids:
        drug_feature_cols.append('DRUG_ID')
        
    # 2. 細胞株特徵 (剩下的都是)
    # 先列出所有潛在的特徵欄位
    all_potential_features = [c for c in df_clean.columns if c not in exclude_cols and c != 'LN_IC50']
    
    # 細胞特徵 = 所有特徵 - 藥物特徵
    cell_feature_cols = [c for c in all_potential_features if c not in drug_feature_cols]
    
    # 如果不包含 ID，確保 COSMIC_ID 和 DRUG_ID 不在細胞特徵中
    if not include_ids:
        if 'COSMIC_ID' in cell_feature_cols:
            cell_feature_cols.remove('COSMIC_ID')
        if 'DRUG_ID' in cell_feature_cols:
            cell_feature_cols.remove('DRUG_ID')
        
    print(f"   -> 藥物特徵 ({len(drug_feature_cols)}): {drug_feature_cols}")
    print(f"   -> 細胞特徵 ({len(cell_feature_cols)}): {cell_feature_cols}")
    
    # 建立 X_cell, X_drug, y
    X_cell = df_clean[cell_feature_cols]
    X_drug = df_clean[drug_feature_cols]
    y = df_clean['LN_IC50']
    
    feature_names = {
        'cell': cell_feature_cols,
        'drug': drug_feature_cols
    }
    
    # 分割訓練/測試集
    # 準備 Stratified Sampling 的標籤
    stratify_col = None
    if 'GDSC Tissue descriptor 1' in df_clean.columns:
        stratify_col = df_clean['GDSC Tissue descriptor 1']
        print("   -> 已啟用 Stratified Sampling (依據 GDSC Tissue descriptor 1)")

    # 分割訓練/測試集 (同時分割三個 array)
    X_cell_train, X_cell_test, X_drug_train, X_drug_test, y_train, y_test = train_test_split(
        X_cell, X_drug, y, test_size=0.2, random_state=42, stratify=stratify_col
    )
    
    print(f"✅ 資料分割完成 (DL):")
    print(f"   X_cell_train: {X_cell_train.shape}, X_drug_train: {X_drug_train.shape}")
    print(f"   X_cell_test:  {X_cell_test.shape},  X_drug_test:  {X_drug_test.shape}")
    
    return X_cell_train, X_drug_train, X_cell_test, X_drug_test, y_train, y_test, feature_names, label_encoders
