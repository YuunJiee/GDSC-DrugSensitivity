# 雙重資料預處理策略說明

## 📋 概述

目前的 `main.py` 使用了**兩種不同的資料預處理策略**：

1. **基線模型（RF + XGBoost）**：使用 **One-Hot Encoding**
2. **深度學習模型**：使用 **Label Encoding**（獨立處理）

這是一個刻意的設計決策，可能會帶來更好的模型效能。

---

## 🔄 資料處理流程對比

### 方案 A：基線模型（RF + XGBoost）

```python
# main.py -> preprocess_data()
df = pd.read_csv('Preprocessing/Data_imputed.csv')

# 移除 ID 欄位
drop_cols = ['COSMIC_ID', 'CELL_LINE_NAME', 'DRUG_ID', 'AUC', 'Z_SCORE', 'RMSE', ...]
df_cleaned = df.drop(columns=drop_cols)

# ⭐ One-Hot Encoding（類別特徵轉為多個二元特徵）
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
df_processed = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**特點：**
- ✅ 特徵數量**大幅增加**（每個類別變數變成多個二元變數）
- ✅ 適合樹狀模型（RF、XGBoost）
- ❌ 對深度學習可能造成**特徵維度過高**的問題

**範例：**
```
原始: TCGA_DESC = ['BRCA', 'LUAD', 'COAD']

One-Hot 後:
- TCGA_DESC_BRCA = [1, 0, 0]
- TCGA_DESC_LUAD = [0, 1, 0]
- TCGA_DESC_COAD = [0, 0, 1]

→ 1個特徵變成 3個特徵
```

---

### 方案 B：深度學習模型

```python
# DL/deep_learning_model.py -> preprocess_data_standalone()
df = pd.read_csv('Preprocessing/Data_imputed.csv')

# 移除缺失值
df_clean = df.dropna(subset=['LN_IC50'])

# 移除不需要的欄位
exclude_cols = ['COSMIC_ID', 'CELL_LINE_NAME', 'DRUG_NAME', 'DRUG_ID']

# ⭐ Label Encoding（類別特徵轉為整數編碼）
for col in categorical_cols:
    if col not in exclude_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# 分割資料
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(...)
```

**特點：**
- ✅ 特徵數量**保持不變**（每個類別變數仍是1個特徵）
- ✅ 適合深度學習（特徵維度較低）
- ✅ 減少記憶體使用
- ❌ 不適合樹狀模型（會錯誤解讀編碼順序）

**範例：**
```
原始: TCGA_DESC = ['BRCA', 'LUAD', 'COAD']

Label Encoding 後:
- TCGA_DESC = [0, 1, 2]

→ 1個特徵仍是 1個特徵
```

---

## 📊 特徵數量對比

假設原始資料有：
- 5 個數值型特徵
- 10 個類別型特徵，每個平均有 20 個類別

### One-Hot Encoding
```
特徵數量 = 5 (數值) + 10 × 20 (One-Hot) = 205 個特徵
```

### Label Encoding
```
特徵數量 = 5 (數值) + 10 (Label) = 15 個特徵
```

**差異：205 vs 15 → 約 13 倍差距！**

---

## 🎯 為什麼要使用不同的策略？

### 基線模型（RF + XGBoost）適合 One-Hot Encoding

**原因：**
1. 樹狀模型可以有效處理高維稀疏特徵
2. One-Hot 避免了順序假設（類別間無大小關係）
3. XGBoost 和 RF 對特徵維度不敏感

**缺點：**
- 訓練速度較慢（特徵多）
- 記憶體使用較高

### 深度學習模型適合 Label Encoding

**原因：**
1. 神經網路可以自動學習特徵表示
2. 減少參數數量（第一層權重矩陣更小）
3. 訓練更快、更穩定
4. 配合 Embedding 層效果更好（未來可擴展）

**缺點：**
- 可能引入虛假的順序關係
- 但深度學習可以透過非線性層打破這個假設

---

## 🔧 在 main.py 中的實作

```python
# main.py

# 1. 基線模型使用共用的預處理
X_train, X_test, y_train, y_test, features = preprocess_data(file_path)

# 訓練 RF 和 XGBoost
rf_model, rf_pred = train_rf_model(X_train, y_train, X_test)
xgb_model, xgb_pred = train_xgb_model(X_train, y_train, X_test)

# 2. 深度學習使用獨立的預處理
if RUN_DEEP_LEARNING:
    # ⭐ 獨立的資料處理
    X_train_dl, X_test_dl, y_train_dl, y_test_dl, features_dl, encoders = \
        preprocess_data_standalone(file_path)
    
    # 訓練深度學習模型
    encoder, mlp_model, dl_pred, dl_metrics = run_deep_learning_pipeline(
        X_train_dl, X_test_dl, y_train_dl, y_test_dl, features_dl
    )
```

---

## ⚠️ 重要注意事項

### 1. 測試集不同
由於使用不同的資料處理流程（且 `random_state=42` 相同），三個模型的測試集**樣本可能略有差異**：
- 基線模型：使用 One-Hot 後的測試集
- 深度學習：使用 Label Encoding 後的測試集

但由於隨機種子相同，樣本順序應該一致。

### 2. 特徵名稱不同
- `features`（基線）：包含 One-Hot 後的特徵名（如 `TCGA_DESC_BRCA`）
- `features_dl`（深度學習）：保持原始特徵名（如 `TCGA_DESC`）

### 3. 效能比較的公平性
雖然測試集略有不同，但由於：
- 資料來源相同（`Data_imputed.csv`）
- 隨機種子相同（`random_state=42`）
- 測試比例相同（`test_size=0.2`）

效能比較**仍然具有參考價值**，可以合理比較三個模型的相對效能。

---

## 📈 預期結果

### One-Hot Encoding 優勢場景
- 類別特徵具有明確的語義差異
- 樹狀模型（RF、XGBoost）

### Label Encoding 優勢場景
- 深度學習模型
- 特徵維度過高導致的記憶體問題
- 訓練速度要求

---

## 🚀 執行方式

### 僅執行基線模型
```python
# main.py
RUN_DEEP_LEARNING = False
```

執行：
```bash
python3 main.py
```

### 執行所有模型（含深度學習）
```python
# main.py
RUN_DEEP_LEARNING = True
```

執行：
```bash
python3 main.py
```

輸出會包含：
```
💡 注意：深度學習模型使用獨立的資料預處理流程（Label Encoding）
   這與基線模型的 One-Hot Encoding 不同，可能產生更好的結果

============================================================
最終模型效能比較
============================================================

💡 說明：
   - Random Forest & XGBoost: 使用 One-Hot Encoding
   - Deep Learning: 使用 Label Encoding（獨立資料處理）
   - 由於資料處理方式不同，測試集可能略有差異

模型                          RMSE         R²          
------------------------------------------------------------
Random Forest                X.XXXX       0.XXXX
XGBoost                      X.XXXX       0.XXXX
Deep Learning (Neural Net)   X.XXXX       0.XXXX
============================================================
```

---

## 🎯 總結

這種雙重預處理策略允許：
1. ✅ 基線模型使用最適合的編碼方式（One-Hot）
2. ✅ 深度學習模型使用更有效的編碼方式（Label）
3. ✅ 公平比較不同模型的效能
4. ✅ 充分發揮每種模型的優勢

這是**刻意的設計決策**，而非錯誤！

---

**最後更新：** 2025-11-30  
**版本：** 1.0.0
