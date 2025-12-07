import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score

# 導入重構後的模組
from src.data.preprocessing import load_raw_data, load_and_preprocess_baseline, load_and_preprocess_dl
from src.data.eda import perform_eda
from src.models.baseline import train_rf_model, train_xgb_model
from src.models.deep_learning.deep_learning_model import run_deep_learning_pipeline
from src.utils.evaluation import calculate_metrics
from src.utils.visualization import plot_comparison, plot_residuals, plot_training_history
from src.utils.explainability import analyze_branch_importance, analyze_shap_values
# Import optuna tuning module
from src.optimization.optuna_tuning import run_optimization

# 設定繪圖風格
plt.style.use('ggplot')

def run_experiment(file_path, include_ids, run_baseline=True, run_dl=False):
    """
    執行單次實驗
    :param include_ids: 是否包含 ID 特徵
    :param run_baseline: 是否執行基線模型
    :param run_dl: 是否執行深度學習
    :return: 實驗結果字典
    """
    mode_name = "With_IDs" if include_ids else "No_IDs"
    print(f"\n{'='*80}")
    print(f"🚀 開始實驗: {mode_name} (包含 ID: {include_ids})")
    print(f"{'='*80}")

    results = {}

    # 1. 基線模型流程
    if run_baseline:
        print(f"\n--- [Baseline] 資料處理 ({mode_name}) ---")
        X_train, X_test, y_train, y_test, features = load_and_preprocess_baseline(
            file_path, variance_threshold=0.01, include_ids=include_ids
        )
        
        # 1.2 訓練模型
        rf_model, rf_pred = train_rf_model(X_train, y_train, X_test)
        xgb_model, xgb_pred = train_xgb_model(X_train, y_train, X_test)
        
        # 1.3 評估
        rf_rmse, rf_r2, _ = calculate_metrics(y_test, rf_pred, f"Random Forest ({mode_name})")
        xgb_rmse, xgb_r2, _ = calculate_metrics(y_test, xgb_pred, f"XGBoost ({mode_name})")
        
        # 繪製殘差圖
        plot_residuals(y_test, rf_pred, f"Random Forest ({mode_name})", f"results/figures/residuals_rf_{mode_name}.png")
        plot_residuals(y_test, xgb_pred, f"XGBoost ({mode_name})", f"results/figures/residuals_xgb_{mode_name}.png")
        
        results['RF'] = {'RMSE': rf_rmse, 'R2': rf_r2}
        results['XGB'] = {'RMSE': xgb_rmse, 'R2': xgb_r2}

        # 顯示 XGB 特徵重要性 (僅顯示前 5 個，避免洗版)
        feat_imp = pd.DataFrame({
            'feature': features,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n[XGBoost - {mode_name}] Top 5 特徵:\n", feat_imp.head(5))
        
        # 儲存特徵重要性表
        os.makedirs('results/tables', exist_ok=True)
        feat_imp.head(50).to_csv(f'results/tables/feature_importance_xgb_{mode_name}.csv', index=False)
        print(f"   -> 已儲存特徵重要性表: results/tables/feature_importance_xgb_{mode_name}.csv")

    # 2. 深度學習模型流程
    if run_dl:
        print(f"\n--- [Deep Learning] 訓練 ({mode_name}) ---")
        
        # 2.1 資料處理 (DL)
        X_cell_train, X_drug_train, X_cell_test, X_drug_test, y_train_dl, y_test_dl, features_dl, encoders = \
            load_and_preprocess_dl(file_path, include_ids=include_ids)
        
        # 2.2 執行 Pipeline
        scalers, mlp_model, dl_pred, dl_metrics, history = run_deep_learning_pipeline(
            X_cell_train, X_drug_train, X_cell_test, X_drug_test, y_train_dl, y_test_dl, features_dl
        )
        
        # 繪製學習曲線
        plot_training_history(history, f"results/figures/dl_learning_curves_{mode_name}.png")
        
        # --- 新增: 可解釋性分析 (Explainability) ---
        # 1. Macro-Level: Branch Importance
        analyze_branch_importance(mlp_model, f"results/figures/dl_branch_importance_{mode_name}.png")
        
        # 2. Micro-Level: SHAP Values
        # 注意: 這裡傳入的是原始數據的 list，SHAP 會自己處理
        analyze_shap_values(
            mlp_model, 
            [X_cell_train, X_drug_train], 
            [X_cell_test, X_drug_test], 
            features_dl, 
            f"results/figures/dl_shap_summary_{mode_name}.png"
        )
        
        results['DL'] = {'RMSE': dl_metrics['RMSE'], 'R2': dl_metrics['R2']}
        
        # 繪製殘差圖
        plot_residuals(y_test_dl, dl_pred, f"Deep Learning ({mode_name})", f"results/figures/residuals_dl_{mode_name}.png")

    # 儲存模型評估指標 (Merge with existing if available)
    metrics_path = f'results/tables/metrics_{mode_name}.csv'
    if os.path.exists(metrics_path):
        try:
            existing_df = pd.read_csv(metrics_path, index_col=0)
            existing_results = existing_df.T.to_dict()
            # Update existing results with new ones
            existing_results.update(results)
            results = existing_results
        except Exception as e:
            print(f"⚠️ 無法讀取舊的結果檔案，將建立新檔案: {e}")

    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(metrics_path)
    print(f"   -> 已儲存模型評估指標: {metrics_path}")

    return results

def plot_experiment_comparison(results_with_ids, results_no_ids):
    """繪製實驗對比圖"""
    # 動態偵測有哪些模型
    all_models = set(results_with_ids.keys()) | set(results_no_ids.keys())
    # 排序：RF, XGB, DL (如果存在)
    preferred_order = ['RF', 'XGB', 'DL']
    models = [m for m in preferred_order if m in all_models]
    # 加入其他可能的模型
    for m in all_models:
        if m not in models:
            models.append(m)
    
    if not models:
        print("⚠️ 沒有可用的模型結果進行繪圖。")
        return

    r2_with_ids = []
    r2_no_ids = []
    
    for m in models:
        r2_with_ids.append(results_with_ids.get(m, {}).get('R2', 0))
        r2_no_ids.append(results_no_ids.get(m, {}).get('R2', 0))
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, r2_with_ids, width, label='With IDs (Memorization)', color='#3498db')
    rects2 = ax.bar(x + width/2, r2_no_ids, width, label='No IDs (Generalization)', color='#e74c3c')
    
    ax.set_ylabel('R² Score')
    ax.set_title('Impact of ID Features on Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png')
    print("\n📊 對比圖表已儲存為: experiment_comparison.png")
    # plt.show() # 如果在非互動環境可註解掉

import argparse

def load_saved_results(mode_name):
    """從 CSV 載入儲存的實驗結果"""
    path = f'results/tables/metrics_{mode_name}.csv'
    if not os.path.exists(path):
        print(f"⚠️ 找不到 {mode_name} 的結果檔案 ({path})，無法進行比較。")
        return {}
    
    try:
        df = pd.read_csv(path, index_col=0)
        return df.T.to_dict()
    except Exception as e:
        print(f"⚠️ 讀取 {mode_name} 結果失敗: {e}")
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDSC Drug Sensitivity Prediction Pipeline")
    parser.add_argument('--eda', action='store_true', help='Perform EDA')
    parser.add_argument('--baseline', action='store_true', help='Run Baseline Models')
    parser.add_argument('--dl', action='store_true', help='Run Deep Learning Model')
    parser.add_argument('--optimize', action='store_true', help='Run Optimization')
    parser.add_argument('--compare', action='store_true', help='Compare Results')
    parser.add_argument('--viz-features', action='store_true', help='Visualize Features')
    parser.add_argument('--all', action='store_true', help='Run All Steps')
    parser.add_argument('--viz-with-id', action='store_true', help='Experiment With IDs')
    parser.add_argument('--viz-no-id', action='store_true', help='Experiment Without IDs')
    
    args = parser.parse_args()
    file_path = 'data/raw/GDSC_DATASET.csv'
    
    try:
        if args.optimize:
            try:
                print("\n=== 優化模式 ===")
                print("1. Deep Learning (dl)")
                print("2. Random Forest (rf)")
                print("3. XGBoost (xgb)")
                model_choice = input("輸入模型代號 (dl/rf/xgb) [預設: dl]: ").strip().lower()
                if not model_choice: model_choice = 'dl'
                if model_choice not in ['dl', 'rf', 'xgb']: model_choice = 'dl'
                
                run_optimization(file_path, include_ids=True, model_type=model_choice)
            except KeyboardInterrupt:
                print("\n優化已取消。")
        # 0. 執行 EDA
        if args.eda or args.all:
            print("Step 0: 執行探索性資料分析 (EDA)...")
            # 使用原始資料進行 EDA，以呈現真實的缺失值狀況
            raw_df = load_raw_data('data/raw/GDSC_DATASET.csv')
            perform_eda(raw_df)
        
        results_with_ids = {}
        results_no_ids = {}

        # 1. 執行實驗 (Baseline & DL)
        # 為了避免重複載入資料，我們將 Baseline 和 DL 的執行邏輯整合在 run_experiment 中
        # 但透過參數控制是否執行 DL
        
        run_baseline = args.baseline or args.all
        run_dl = args.dl or args.all
        
        if run_baseline or run_dl:
            # Determine which modes to run
            # Default to running both
            do_with_id = True
            do_no_id = True
            
            # If specific mode flags are provided, they override the default "run both" behavior
            if args.viz_with_id or args.viz_no_id: # Changed from args.with_id / args.no_id to viz flags
                do_with_id = args.viz_with_id
                do_no_id = args.viz_no_id

            # 實驗 1: 包含 ID (高分模式)
            if do_with_id:
                print(f"\n🚀 執行實驗: With_IDs (Baseline: {run_baseline}, DL: {run_dl})")
                results_with_ids = run_experiment(file_path, include_ids=True, run_baseline=run_baseline, run_dl=run_dl)
            
            # 實驗 2: 不包含 ID (泛化模式)
            if do_no_id:
                print(f"\n🚀 執行實驗: No_IDs (Baseline: {run_baseline}, DL: {run_dl})")
                results_no_ids = run_experiment(file_path, include_ids=False, run_baseline=run_baseline, run_dl=run_dl)
        
        # 2. 執行比較
        if args.compare or args.all:
            print("\nStep 3: 執行模型比較...")
            
            # 如果結果字典是空的 (代表這次沒跑模型)，嘗試從檔案載入
            if not results_with_ids:
                print("   -> 嘗試載入 With_IDs 歷史結果...")
                results_with_ids = load_saved_results("With_IDs")
            
            if not results_no_ids:
                print("   -> 嘗試載入 No_IDs 歷史結果...")
                results_no_ids = load_saved_results("No_IDs")
                
            if results_with_ids and results_no_ids:
                # 繪製比較圖
                plot_experiment_comparison(results_with_ids, results_no_ids)
                
                # 儲存最終實驗總結表
                summary_data = []
                # 取得所有模型名稱 (聯集)
                all_models = set(results_with_ids.keys()) | set(results_no_ids.keys())
                
                for model in all_models:
                     row = {'Model': model}
                     if model in results_with_ids:
                         row['With_IDs_R2'] = results_with_ids[model]['R2']
                         row['With_IDs_RMSE'] = results_with_ids[model]['RMSE']
                     if model in results_no_ids:
                         row['No_IDs_R2'] = results_no_ids[model]['R2']
                         row['No_IDs_RMSE'] = results_no_ids[model]['RMSE']
                     summary_data.append(row)
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv('results/tables/final_experiment_summary.csv', index=False)
                print(f"\n✅ 最終實驗總結表已儲存: results/tables/final_experiment_summary.csv")
            else:
                 # 若無法比較（例如只跑了其中一種），至少印出訊息
                 pass

        # 獨立視覺化功能
        should_viz = args.viz_features or args.all
        # 如果使用者只指定了 --viz-with-id 或 --viz-no-id 但沒有指定執行實驗 (--baseline/--dl)，則假設是為了視覺化
        if (args.viz_with_id or args.viz_no_id) and not (args.baseline or args.dl):
            should_viz = True

        if should_viz:
            from src.utils.visualization import plot_feature_importance, plot_single_mode_performance_bar
            
            print("\n📊 正在繪製視覺化圖表...")
            
            # 決定要畫哪些
            show_no_id = True
            show_with_id = True
            
            # 如果有指定特定模式，則只畫該模式 (除非是 viz-features 這種大開關)
            if (args.viz_with_id or args.viz_no_id) and not args.viz_features and not args.all:
                 show_no_id = args.viz_no_id
                 show_with_id = args.viz_with_id

            # --- 1. 特徵重要性 & R2 表現圖 (No_IDs) ---
            if show_no_id:
                # Feature Importance
                csv_path_no = 'results/tables/feature_importance_xgb_No_IDs.csv'
                if os.path.exists(csv_path_no):
                    try:
                        df = pd.read_csv(csv_path_no)
                        plot_feature_importance(df['feature'], df['importance'], top_n=20, save_path='results/figures/feature_importance_xgb_No_IDs.png')
                        print("   -> No_IDs 特徵圖已更新")
                    except Exception as e:
                        print(f"   ⚠️ 無法繪製 No_IDs 特徵圖: {e}")
                elif args.viz_no_id:
                     print(f"   ⚠️ 找不到 No_IDs 特徵結果 ({csv_path_no})")
                
                # R2 Score Bar Chart
                metrics_no = load_saved_results("No_IDs")
                if metrics_no:
                    try:
                        plot_single_mode_performance_bar(metrics_no, "No_IDs", 'results/figures/performance_bar_No_IDs.png')
                    except Exception as e:
                        print(f"   ⚠️ 無法繪製 No_IDs R2 圖: {e}")

            # --- 2. 特徵重要性 & R2 表現圖 (With_IDs) ---
            if show_with_id:
                # Feature Importance
                csv_path_with = 'results/tables/feature_importance_xgb_With_IDs.csv'
                if os.path.exists(csv_path_with):
                    try:
                        df = pd.read_csv(csv_path_with)
                        plot_feature_importance(df['feature'], df['importance'], top_n=20, save_path='results/figures/feature_importance_xgb_With_IDs.png')
                        print("   -> With_IDs 特徵圖已更新")
                    except Exception as e:
                        print(f"   ⚠️ 無法繪製 With_IDs 特徵圖: {e}")
                elif args.viz_with_id:
                     print(f"   ⚠️ 找不到 With_IDs 特徵結果 ({csv_path_with})")
                
                # R2 Score Bar Chart
                metrics_with = load_saved_results("With_IDs")
                if metrics_with:
                    try:
                        plot_single_mode_performance_bar(metrics_with, "With_IDs", 'results/figures/performance_bar_With_IDs.png')
                    except Exception as e:
                        print(f"   ⚠️ 無法繪製 With_IDs R2 圖: {e}")

    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()

