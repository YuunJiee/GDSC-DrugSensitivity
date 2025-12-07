import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from src.config import PathConfig

# 設定全域繪圖風格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# 通用設定：直式圖片 (Portrait) + 專業冷色調 (Mako)
PORTRAIT_SIZE = (8, 10)
MAIN_PALETTE = 'mako'

def analyze_missing_data(output_dir):
    """
    分析缺失值快照 (missing_data_snapshot.csv)
    """
    snapshot_path = os.path.join(PathConfig.PROCESSED_DATA_DIR, 'missing_data_snapshot.csv')
    if not os.path.exists(snapshot_path):
        print(f"⚠️ 未找到缺失值快照: {snapshot_path}，跳過缺失值深入分析。")
        return

    print(f"🔍 分析缺失值快照: {snapshot_path}")
    df_missing = pd.read_csv(snapshot_path)

    missing_counts = df_missing.isnull().sum()
    missing_pct = (missing_counts / len(df_missing)) * 100
    missing_stats = pd.DataFrame({'Missing Count': missing_counts, 'Missing Percent': missing_pct})
    missing_stats = missing_stats[missing_stats['Missing Count'] > 0].sort_values('Missing Percent', ascending=False)
    
    plt.figure(figsize=PORTRAIT_SIZE) # 直式
    ax = sns.barplot(y=missing_stats.index, x=missing_stats['Missing Percent'], palette=MAIN_PALETTE)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f%%', padding=3)
    plt.title('Percentage of Missing Values by Feature')
    plt.xlabel('Missing Percentage (%)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/0_missing_percentage.png')
    plt.close()
    
    # 儲存統計表
    tables_dir = output_dir.replace('figures/eda', 'tables/eda')
    os.makedirs(tables_dir, exist_ok=True)
    missing_stats.to_csv(f'{tables_dir}/0_missing_stats.csv')
    print(f"   -> 缺失值分析完成，圖表已存至 {output_dir}")

def analyze_data_quality(df, output_path=None):
    """
    分析資料品質並回傳報告
    """
    print("🔍 分析資料品質...")
    report = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'missing_values': {},
        'high_missing_features': [],
        'skewed_features': []
    }
    
    # 缺失值分析
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for col, count in missing.items():
        pct = (count / len(df)) * 100
        report['missing_values'][col] = {'count': int(count), 'percent': float(pct)}
        if pct > 50:
            report['high_missing_features'].append(col)
            
    # 數值分佈分析
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'LN_IC50': # Skip target
            skew = df[col].skew()
            if abs(skew) > 1:
                report['skewed_features'].append({'feature': col, 'skewness': float(skew)})
                
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"   -> 資料品質報告已儲存: {output_path}")
        
    return report

def perform_eda(df, output_dir=None):
    """
    執行探索性資料分析 (EDA) 並儲存圖表與統計數據
    """
    if output_dir is None:
        output_dir = os.path.join(PathConfig.FIGURES_DIR, 'eda')
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 建立表格儲存目錄
    tables_dir = output_dir.replace('figures/eda', 'tables/eda')
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)
        
    print(f"\n📊 開始 EDA 分析")
    print(f"   -> 圖表儲存至: {output_dir}")
    print(f"   -> 表格儲存至: {tables_dir}")
    
    # 0. 缺失值深入分析 (新增)
    analyze_missing_data(output_dir)
    
    # 0.1 資料品質分析
    analyze_data_quality(df, output_path=os.path.join(PathConfig.REPORTS_DIR, 'data_quality.json'))
    
    # 1. 目標變數分佈 (LN_IC50)
    plt.figure(figsize=(8, 6)) # Histogram 稍微方一點沒關係
    sns.histplot(df['LN_IC50'], kde=True, bins=50, color='#3b7c98') # Mako 風格的藍綠色
    plt.title('Distribution of Drug Sensitivity (LN_IC50)')
    plt.xlabel('LN_IC50 (Lower = More Sensitive)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_ic50_distribution.png')
    plt.close()
    
    # 儲存統計數據
    ic50_stats = df['LN_IC50'].describe()
    ic50_stats.to_csv(f'{tables_dir}/1_ic50_distribution_stats.csv')
    print("   -> 已儲存: 1_ic50_distribution.png & stats.csv")

    # 2. 不同組織類型的樣本數 (Top 20)
    if 'GDSC Tissue descriptor 1' in df.columns:
        plt.figure(figsize=PORTRAIT_SIZE)
        top_tissues = df['GDSC Tissue descriptor 1'].value_counts().head(20)
        # 轉成 DataFrame 以便使用 sns.barplot 的 data 參數，避免 FutureWarning
        top_tissues_df = top_tissues.reset_index()
        top_tissues_df.columns = ['Tissue Type', 'Count']
        
        ax = sns.barplot(data=top_tissues_df, y='Tissue Type', x='Count', palette=MAIN_PALETTE)
        for i in ax.containers:
            ax.bar_label(i, fmt='%d', padding=3)
        plt.title('Top 20 Tissue Types by Sample Count')
        plt.xlabel('Count')
        plt.ylabel('Tissue Type')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/2_tissue_counts.png')
        plt.close()
        
        # 儲存統計數據
        top_tissues.to_csv(f'{tables_dir}/2_tissue_counts.csv')
        print("   -> 已儲存: 2_tissue_counts.png & .csv")

    # 3. 不同靶點路徑的 IC50 分佈 (Boxplot)
    if 'TARGET_PATHWAY' in df.columns:
        plt.figure(figsize=PORTRAIT_SIZE)
        # 排序：按中位數排序
        order = df.groupby('TARGET_PATHWAY')['LN_IC50'].median().sort_values().index
        sns.boxplot(x='LN_IC50', y='TARGET_PATHWAY', data=df, order=order, palette=MAIN_PALETTE)
        plt.title('Drug Sensitivity by Target Pathway')
        plt.xlabel('LN_IC50')
        plt.ylabel('Target Pathway')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/3_pathway_sensitivity.png')
        plt.close()
        
        # 儲存統計數據
        pathway_stats = df.groupby('TARGET_PATHWAY')['LN_IC50'].agg(['count', 'mean', 'median', 'std']).sort_values('median')
        pathway_stats.to_csv(f'{tables_dir}/3_pathway_sensitivity_stats.csv')
        print("   -> 已儲存: 3_pathway_sensitivity.png & stats.csv")

    # 4. 最敏感的前 20 種藥物
    if 'DRUG_NAME' in df.columns:
        plt.figure(figsize=PORTRAIT_SIZE)
        drug_sensitivity = df.groupby('DRUG_NAME')['LN_IC50'].mean().sort_values().head(20)
        
        drug_df = drug_sensitivity.reset_index()
        drug_df.columns = ['Drug Name', 'Mean LN_IC50']
        
        ax = sns.barplot(data=drug_df, y='Drug Name', x='Mean LN_IC50', palette=MAIN_PALETTE)
        for i in ax.containers:
            ax.bar_label(i, fmt='%.2f', padding=3)
        plt.title('Top 20 Most Potent Drugs (Lowest Mean LN_IC50)')
        plt.xlabel('Mean LN_IC50')
        plt.ylabel('Drug Name')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/4_top_drugs.png')
        plt.close()
        
        # 儲存統計數據
        drug_stats = df.groupby('DRUG_NAME')['LN_IC50'].agg(['count', 'mean', 'median']).sort_values('mean').head(50)
        drug_stats.to_csv(f'{tables_dir}/4_top_drugs_stats.csv')
        print("   -> 已儲存: 4_top_drugs.png & stats.csv")

    # 5. 數值變數相關性熱圖
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 排除 ID 類
    cols_to_plot = [c for c in numeric_cols if 'ID' not in c and 'id' not in c.lower()]
    
    if len(cols_to_plot) > 1:
        plt.figure(figsize=PORTRAIT_SIZE) # 直式熱圖
        corr = df[cols_to_plot].corr()
        sns.heatmap(corr, annot=True, cmap=MAIN_PALETTE, fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/5_correlation_heatmap.png')
        plt.close()
        
        # 儲存相關係數矩陣
        corr.to_csv(f'{tables_dir}/5_correlation_matrix.csv')
        print("   -> 已儲存: 5_correlation_heatmap.png & matrix.csv")

    print("✅ EDA 分析完成！\n")
