
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.data.preprocessing import load_and_preprocess_dl
from src.utils.explainability import analyze_branch_importance, analyze_shap_values

def run_standalone_shap(mode='No_IDs'):
    print(f"🚀 Running Standalone SHAP Analysis for mode: {mode}")
    
    # 1. Load Model & Scalers
    model_path = f"results/models/dl_model_{mode}.h5"
    scalers_path = f"results/models/dl_scalers_{mode}.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scalers_path):
        print(f"❌ Model or Scalers not found for mode {mode}.")
        print(f"   Model: {model_path}")
        print(f"   Scalers: {scalers_path}")
        print("   Please run 'python main.py --dl' first.")
        return

    print("   -> Loading model and scalers...")
    model = load_model(model_path)
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    scaler_cell, scaler_drug = scalers
    
    # 2. Load Data
    print("   -> Loading data...")
    file_path = 'data/raw/GDSC_DATASET.csv'
    include_ids = (mode == 'With_IDs')
    
    # This returns unscaled data splits
    dl_data = load_and_preprocess_dl(file_path, include_ids=include_ids)
    
    X_train_tuple = dl_data[0] # (X_cell, X_drug, X_target, X_pathway)
    X_test_tuple = dl_data[1]
    features_dl = dl_data[4]
    
    # 3. Scale Data
    print("   -> Scaling data...")
    def scale_data_list(data_tuple):
        # Update unpack for 5 elements
        x_c, x_d_num, x_d_id, x_t, x_p = data_tuple
        
        x_c_s = scaler_cell.transform(x_c)
        
        if x_d_num.shape[1] > 0:
            x_d_num_s = scaler_drug.transform(x_d_num)
        else:
            x_d_num_s = np.zeros((x_d_num.shape[0], 1))
        
        # DrugID, Target, Pathway do not need scaling
        return [x_c_s, x_d_num_s, x_d_id, x_t, x_p]

    X_train_list_scaled = scale_data_list(X_train_tuple)
    X_test_list_scaled = scale_data_list(X_test_tuple)
    
    # 4. Run Analysis
    output_dir = "results/figures_standalone"
    os.makedirs(output_dir, exist_ok=True)
    
    # Branch Importance
    # analyze_branch_importance(model, f"{output_dir}/dl_branch_importance_{mode}.png")
    
    # SHAP
    analyze_shap_values(
        model, 
        X_train_list_scaled, 
        X_test_list_scaled, 
        features_dl, 
        f"{output_dir}/dl_shap_summary_{mode}.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="No_IDs", choices=["With_IDs", "No_IDs"], help="Mode to analyze")
    args = parser.parse_args()
    
    run_standalone_shap(args.mode)
