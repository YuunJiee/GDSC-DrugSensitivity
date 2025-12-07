import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 設定全域繪圖風格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Mako 色調
MAKO_COLOR_1 = '#3b7c98' # 淺藍綠
MAKO_COLOR_2 = '#2a5674' # 深藍綠
MAKO_PALETTE = 'mako'

def plot_comparison(y_test, rf_pred, xgb_pred, rf_rmse, xgb_rmse, save_path='results/figures/model_comparison.png'):
    """
    繪製基線模型比較圖 (Random Forest vs XGBoost)
    """
    plt.figure(figsize=(14, 6))

    # 左圖: RF
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, rf_pred, alpha=0.5, color=MAKO_COLOR_1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'Random Forest\nRMSE: {rf_rmse:.3f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # 右圖: XGB
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, xgb_pred, alpha=0.5, color=MAKO_COLOR_2)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'XGBoost\nRMSE: {xgb_rmse:.3f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n圖表已保存為 '{save_path}'")

def plot_training_history(history, save_path='results/figures/dl_learning_curves.png'):
    """
    繪製深度學習訓練過程曲線
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # MAE 曲線
    ax1.plot(history.history['mean_absolute_error'], label='Train MAE', linewidth=2, color=MAKO_COLOR_1)
    ax1.plot(history.history['val_mean_absolute_error'], label='Val MAE', linewidth=2, color=MAKO_COLOR_2)
    ax1.set_title('Model MAE over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss 曲線
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2, color=MAKO_COLOR_1)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2, color=MAKO_COLOR_2)
    ax2.set_title('Model Loss (MSE) over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 訓練曲線已保存至: {save_path}")
    plt.close()

def plot_predictions(y_test, y_pred, metrics, save_path='results/figures/dl_predictions_vs_actual.png'):
    """
    繪製預測 vs 實際值散點圖 (深度學習)
    """
    plt.figure(figsize=(10, 8))
    
    # 散點圖
    plt.scatter(y_test, y_pred, alpha=0.5, s=20, color=MAKO_COLOR_1, edgecolors=MAKO_COLOR_2, linewidth=0.5)
    
    # 完美預測線
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # 標題和標籤
    plt.title(f'Predictions vs Actual IC50 Values\nR² = {metrics["R2"]:.4f}, RMSE = {metrics["RMSE"]:.4f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Actual LN_IC50', fontsize=12)
    plt.ylabel('Predicted LN_IC50', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加統計資訊
    textstr = f'MAE = {metrics["MAE"]:.4f}\nSpearman ρ = {metrics["Spearman_Correlation"]:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=MAKO_COLOR_2))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 預測散點圖已保存至: {save_path}")
    plt.close()

def plot_feature_importance(feature_names, importance_values, top_n=20, save_path='results/figures/dl_feature_importance.png'):
    """
    繪製特徵重要性
    """
    # 創建 DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False).head(top_n)
    
    # 繪圖
    plt.figure(figsize=(12, 8))
    # 使用 mako palette
    ax = sns.barplot(x='importance', y='feature', data=importance_df, palette=MAKO_PALETTE)
    
    # 添加數值標籤
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
        
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 特徵重要性圖已保存至: {save_path}")
    plt.close()
    
    return importance_df

def plot_residuals(y_test, y_pred, model_name, save_path):
    """
    繪製殘差圖 (Residual Plot)
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color=MAKO_COLOR_2)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residual Plot - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ 殘差圖已保存至: {save_path}")

def plot_single_mode_performance_bar(results, mode_name, save_path):
    """
    繪製單一模式下的模型 R² 表現長條圖
    
    :param results: 字典, 格式如 {'RF': {'R2': 0.8, 'RMSE': 0.5}, 'XGB': ...}
    :param mode_name: 字串, e.g. "With_IDs" or "No_IDs"
    :param save_path: 圖片儲存路徑
    """
    if not results:
        print(f"⚠️ {mode_name} 沒有結果可供繪圖。")
        return

    models = list(results.keys())
    r2_scores = [results[m].get('R2', 0) for m in models]
    
    # Sort models by R2 score for better visualization
    sorted_pairs = sorted(zip(models, r2_scores), key=lambda x: x[1], reverse=True)
    models = [p[0] for p in sorted_pairs]
    r2_scores = [p[1] for p in sorted_pairs]

    plt.figure(figsize=(8, 6))
    
    # Create bar plot
    ax = sns.barplot(x=models, y=r2_scores, palette=MAKO_PALETTE)
    
    plt.title(f'Model Performance (R²) - {mode_name}', fontsize=14, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, 1.1)  # R2 cap at 1.1 for label space, usually max is 1.0
    
    # Add value labels
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)

    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {mode_name} R² 圖表已保存至: {save_path}")

def plot_experiment_comparison(results_with_ids, results_no_ids, save_path='experiment_comparison.png'):
    """
    繪製實驗對比圖 (With IDs vs No IDs)
    """
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
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # 使用 Mako 配色
    # With IDs (Memorization) -> Darker color
    # No IDs (Generalization) -> Lighter color
    rects1 = ax.bar(x - width/2, r2_with_ids, width, label='With IDs (Memorization)', color=MAKO_COLOR_2)
    rects2 = ax.bar(x + width/2, r2_no_ids, width, label='No IDs (Generalization)', color=MAKO_COLOR_1)
    
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Impact of ID Features on Model Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    # 將圖例移至右上方外部
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 對比圖表已儲存為: {save_path}")
