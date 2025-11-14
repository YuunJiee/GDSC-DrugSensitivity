# GDSC dataset — Preprocessing summary

檔案位置
- 原始資料檔：`GDSC_DATASET.csv`
- 已處理資料：`processed/Data_imputed.csv`（整個資料集補完缺失值的版本）

主要 preprocessing 步驟（摘要）
1. 移除/保留的欄位
   - 移除在分析中不作為特徵的 meta 欄位（`CELL_LINE_NAME`和`DRUG_NAME`

2. 缺失值處理（drug-by-drug）
   - 對每個 `DRUG_NAME` 的子集合分別處理缺失值（drug-by-drug）：
     - 組織/癌症類別欄位：在該藥物內使用其它相關欄位或該欄目的眾數填補，若仍缺則以 `Unknown`。
     - `TARGET` / `TARGET_PATHWAY`：若該藥物所有樣本皆缺值，標記為 `Unknown for this drug`，否則以該藥物已有的已知值填補缺值，維持藥物一致性。
     - 其他類別欄位（例如 MSI、Screen Medium、Growth Properties）：於藥物子集合內以主要組織類別的眾數填補。
     - 基因組特徵（CNA、Gene Expression、Methylation）：在藥物內以組織分群的眾數填補。