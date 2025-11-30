# 深度學習模型使用指南

## 📚 模型架構

### 🔹 Autoencoder（特徵降維）
```
輸入層 (n 個特徵)
    ↓
Dense(512) + BatchNorm + Dropout(0.3)
    ↓
Dense(256) + BatchNorm + Dropout(0.2)
    ↓
Dense(128) - 編碼層（壓縮特徵）
    ↓
Dense(256) + BatchNorm + Dropout(0.2)
    ↓
Dense(512) + BatchNorm
    ↓
輸出層 (n 個特徵) - 重建原始輸入
```

### 🔹 MLP 回歸模型
```
輸入層 (128 個編碼特徵)
    ↓
Dense(64) + BatchNorm + Dropout(0.3)
    ↓
Dense(32) + BatchNorm + Dropout(0.24)
    ↓
Dense(16) + BatchNorm
    ↓
輸出層 (1) - 預測 LN_IC50
```

## 🚀 使用方式

### 方法 1：執行完整專案（推薦）
```bash
# 啟動虛擬環境
source venv/bin/activate

# 執行主程式（包含基線模型 + 深度學習模型）
python main.py
```

### 方法 2：只執行深度學習模型
```bash
# 啟動虛擬環境
source venv/bin/activate

# 單獨執行深度學習模型
python deep_learning_model.py
```

### 方法 3：只執行基線模型
在 `main.py` 中設定：
```python
RUN_DEEP_LEARNING = False  # 設為 False
```
然後執行：
```bash
python main.py
```

## 📊 評估指標

深度學習模型提供以下評估指標：

| 指標 | 說明 |
|------|------|
| **MAE** | Mean Absolute Error（平均絕對誤差） |
| **MSE** | Mean Squared Error（均方誤差） |
| **RMSE** | Root Mean Squared Error（均方根誤差） |
| **R²** | R-squared（決定係數，越接近 1 越好） |
| **Spearman ρ** | Spearman Correlation（等級相關係數） |

## 📈 生成的視覺化圖表

執行後會生成以下圖表：

### 基線模型（機器學習）
- `model_comparison.png` - RF 和 XGBoost 的預測比較

### 深度學習模型
- `dl_learning_curves.png` - 訓練過程的學習曲線
- `dl_predictions_vs_actual.png` - 預測值 vs 實際值散點圖
- `dl_feature_importance.png` - 前 20 個重要特徵

## ⚙️ 模型配置

### Autoencoder 參數
- **Encoding dimension**: 128（可調整）
- **L2 regularization**: 0.001
- **Dropout**: 0.2-0.3
- **Optimizer**: Adam (lr=0.001)
- **Early stopping patience**: 15 epochs
- **Learning rate decay**: ReduceLROnPlateau

### MLP 參數
- **Hidden layers**: [64, 32, 16]
- **L2 regularization**: 0.001
- **Dropout**: 0.24-0.3
- **Optimizer**: Adam (lr=0.001)
- **Early stopping patience**: 25 epochs
- **Learning rate decay**: ReduceLROnPlateau

## 🔧 自定義模型

如果想調整模型架構，可以修改 `deep_learning_model.py` 中的函數：

```python
# 修改編碼維度
def build_autoencoder(input_dim, encoding_dim=64, ...):  # 原為 128

# 修改 MLP 層數
def build_mlp_model(input_dim, ...):
    model = keras.Sequential([
        layers.Dense(128, ...),  # 可增加神經元數量
        layers.Dense(64, ...),
        # 可添加更多層
    ])
```

## 📝 匯出模型供其他用途

在你的程式中匯入：

```python
from deep_learning_model import run_deep_learning_pipeline

# 執行完整 pipeline
encoder, mlp_model, predictions, metrics = run_deep_learning_pipeline(
    X_train, X_test, y_train, y_test, feature_names
)

# 使用訓練好的模型進行預測
X_encoded = encoder.predict(X_new)
y_pred = mlp_model.predict(X_encoded)
```

## ⏱️ 預計執行時間

- **Autoencoder 訓練**: 5-10 分鐘
- **MLP 訓練**: 3-8 分鐘
- **特徵重要性計算**: 5-15 分鐘（可選）
- **總計**: ~15-30 分鐘（取決於硬體和資料大小）

## 💡 提示

1. **GPU 加速**: 如果有 GPU，TensorFlow 會自動使用，訓練速度可提升 5-10 倍
2. **記憶體需求**: 建議至少 8GB RAM
3. **Early Stopping**: 會自動在驗證損失不再改善時停止訓練
4. **特徵重要性**: 計算較耗時，可在 `run_deep_learning_pipeline()` 中註解掉

## 🐛 常見問題

### Q: 訓練時間過長？
A: 可以減少 epochs 數量或 n_repeats（特徵重要性）

### Q: 模型效果不佳？
A: 嘗試：
- 調整 encoding_dim（特徵壓縮程度）
- 增加/減少正則化強度
- 調整 Dropout 比率
- 增加訓練資料量

### Q: 記憶體不足？
A: 減少 batch_size 或 encoding_dim
