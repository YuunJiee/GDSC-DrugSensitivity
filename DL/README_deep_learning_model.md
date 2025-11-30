# Deep Learning Model - 使用說明

## 📝 概述

`deep_learning_model.py` 是基於 Kaggle Notebook 改進的神經網路回歸模型（方案B），用於預測藥物敏感性（LN_IC50）。

## ✨ 特點

### 1. **雙模式運行**
- **獨立執行模式**：可直接運行進行完整測試
- **模組調用模式**：可被 `main.py` import 使用

### 2. **完整功能**
- ✅ 類別型特徵自動編碼（Label Encoding）
- ✅ 特徵標準化（StandardScaler）
- ✅ 深層神經網路（256→128→64）
- ✅ BatchNormalization + Dropout
- ✅ EarlyStopping + ReduceLROnPlateau callbacks
- ✅ GPU 自動檢測（支援 Apple Silicon Metal）
- ✅ 完整的評估指標（R², RMSE, MAE, Spearman）
- ✅ 自動生成視覺化圖表

### 3. **與 main.py 完美整合**
- 符合 `main.py` 的接口規範
- 返回值：`encoder, model, y_pred, metrics`

---

## 🚀 使用方法

### 方法 1：被 main.py 調用（推薦）

在 `main.py` 中設定：
```python
RUN_DEEP_LEARNING = True  # 啟用深度學習模型
```

然後執行：
```bash
python main.py
# 或
python3 main.py
```

`main.py` 會自動調用：
```python
from DL.deep_learning_model import run_deep_learning_pipeline

encoder, model, y_pred, metrics = run_deep_learning_pipeline(
    X_train, X_test, y_train, y_test, feature_names
)
```

### 方法 2：獨立執行測試

從專案根目錄執行：
```bash
cd DL
python3 deep_learning_model.py
```

或從專案根目錄：
```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from DL.deep_learning_model import run_deep_learning_pipeline
from main import preprocess_data

X_train, X_test, y_train, y_test, features = preprocess_data('Preprocessing/Data_imputed.csv')
encoder, model, y_pred, metrics = run_deep_learning_pipeline(
    X_train, X_test, y_train, y_test, features
)
print(f'R² = {metrics[\"R2\"]:.4f}')
"
```

---

## 📊 輸出結果

### 1. 評估指標
```
==============================================================
Neural Network 評估結果
==============================================================
  R² Score:              0.XXXX
  RMSE:                  X.XXXX
  MAE:                   X.XXXX
  MSE:                   X.XXXX
  Spearman Correlation:  0.XXXX (p=X.XXXXe-XX)
==============================================================
```

### 2. 生成的圖表
- `dl_learning_curves.png` - 訓練過程曲線（MAE & Loss）
- `dl_predictions_vs_actual.png` - 預測 vs 實際值散點圖
- `dl_feature_importance.png` - Top 20 重要特徵

---

## 🏗️ 模型架構

```
Neural_Network_Regression
├─ Input Layer (n_features)
├─ Dense(256) + BatchNorm + ReLU + Dropout(0.3)
├─ Dense(128) + BatchNorm + ReLU + Dropout(0.3)
├─ Dense(64) + ReLU + Dropout(0.2)
└─ Output(1) - Linear activation
```

**參數：**
- Optimizer: Adam
- Loss: MSE (Mean Squared Error)
- Metrics: MAE, MSE
- Batch Size: 64
- Max Epochs: 100
- Early Stopping: patience=10
- Learning Rate Reduction: factor=0.5, patience=5

---

## 🔧 主要函數

### `run_deep_learning_pipeline(X_train, X_test, y_train, y_test, feature_names)`

**描述：** 執行完整的深度學習訓練流程

**參數：**
- `X_train`, `X_test` - 訓練/測試特徵（DataFrame 或 Array）
- `y_train`, `y_test` - 訓練/測試目標變數（Series 或 Array）
- `feature_names` - 特徵名稱列表

**返回值：**
- `encoder` (StandardScaler) - 特徵標準化器
- `model` (Keras Model) - 訓練好的深度學習模型
- `y_pred` (ndarray) - 測試集預測結果
- `metrics` (dict) - 評估指標字典
  ```python
  {
      'R2': float,
      'RMSE': float,
      'MAE': float,
      'MSE': float,
      'Spearman_Correlation': float,
      'Spearman_PValue': float
  }
  ```

### 其他輔助函數

- `check_gpu_availability()` - 檢查 GPU 可用性
- `build_neural_network(input_dim)` - 建立神經網路模型
- `preprocess_data_standalone(file_path)` - 獨立資料預處理（僅用於獨立執行）

---

## ⚙️ 環境需求

```bash
# Python 套件
pandas
numpy
matplotlib
scikit-learn
scipy
tensorflow  # 或 tensorflow-macos (Apple Silicon)

# 選用（Apple Silicon GPU 加速）
tensorflow-metal
```

安裝方式：
```bash
pip install pandas numpy matplotlib scikit-learn scipy tensorflow

# Apple Silicon 用戶額外安裝
pip install tensorflow-metal
```

---

## 📝 資料處理流程

### 被 main.py 調用時：
```
main.py
  └─ preprocess_data() 
      ├─ 讀取 Data_imputed.csv
      ├─ 移除 ID 欄位
      ├─ One-Hot Encoding
      └─ 分割訓練/測試集
                ↓
  └─ run_deep_learning_pipeline()
      ├─ 特徵標準化 (StandardScaler)
      ├─ 切分驗證集
      ├─ 建立模型
      ├─ 訓練模型
      ├─ 評估模型
      └─ 生成視覺化
```

### 獨立執行時：
```
deep_learning_model.py
  └─ preprocess_data_standalone()
      ├─ 讀取 Data_imputed.csv
      ├─ 移除缺失值
      ├─ Label Encoding (類別型特徵)
      └─ 分割訓練/測試集
                ↓
  └─ run_deep_learning_pipeline()
      └─ (同上)
```

---

## 🎯 與 Kaggle 原始版本的改進

| 項目 | Kaggle 原始版 | 方案B（本版本） |
|------|--------------|----------------|
| 資料讀取 | ❌ 需要外部 `merged_df` | ✅ 自動處理 |
| 變數定義 | ❌ `scaler` 未定義 | ✅ 完整定義 |
| 類別型特徵 | ❌ 被忽略 | ✅ Label Encoding |
| 模型深度 | 基本（2層） | ✅ 深層（3層）|
| 正規化 | 僅 Dropout | ✅ BatchNorm + Dropout |
| Callbacks | ❌ 無 | ✅ EarlyStopping + ReduceLR |
| 評估指標 | MAE | ✅ MAE + RMSE + R² + Spearman |
| 視覺化 | 基本 | ✅ 3種專業圖表 |
| GPU 支援 | ❓ 未檢測 | ✅ 自動檢測 Metal/CUDA |
| 模組化 | ❌ 無 | ✅ 可獨立/調用 |

---

## 🐛 故障排除

### 問題 1: ModuleNotFoundError

**錯誤：**
```
ModuleNotFoundError: No module named 'sklearn'
```

**解決：**
```bash
pip install scikit-learn tensorflow pandas numpy matplotlib scipy
```

### 問題 2: 找不到檔案

**錯誤：**
```
FileNotFoundError: [Errno 2] No such file or directory: '../Preprocessing/Data_imputed.csv'
```

**解決：**
- 確認從正確的目錄執行
- 從專案根目錄：`python -c "..."`
- 從 DL 目錄：`python deep_learning_model.py`

### 問題 3: Metal 插件衝突（Apple Silicon）

**錯誤：**
```
Metal device set to: Apple M1
 Metal PluggableDevice already registered
```

**解決：**
這是「警告」不是錯誤，可以忽略。或在代碼開頭加入：
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

---

## 📞 聯絡資訊

如有問題請查閱：
- 專案 README
- main.py 註解
- Kaggle 原始筆記本: https://www.kaggle.com/code/siborakauri/drug-sensitivity

---

**最後更新：** 2025-11-30  
**版本：** 1.0.0 (方案B完整版)
