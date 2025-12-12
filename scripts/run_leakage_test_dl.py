
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from src.data.preprocessing import load_and_preprocess_dl
from src.models.deep_learning.deep_learning_model import GDSCNeuralNetwork
from src.utils.evaluation import evaluate_dl_model
from src.config import ModelConfig

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def run_leakage_test():
    """
    Runs a standalone leakage test by simulating a Random Split (Standard Benchmark)
    and comparing it to the Blind Split logic.
    """
    print("🚀 Starting Leakage Test (Random Split simulation)...")
    
    # 1. Define Paths
    DATA_PATH = 'data/processed/Data_imputed.csv'  # Correct filename found in directory
    TEMP_PATH = 'data/processed/temp_leakage_test.csv'

    # Check if data exists
    if not os.path.exists(DATA_PATH):
        # Try without _data suffix if not found
        DATA_PATH = 'data/processed/gdsc_processed.csv'
        if not os.path.exists(DATA_PATH):
            print(f"❌ Data not found at {DATA_PATH}")
            return

    print(f"   -> Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 2. Rename COSMIC_ID to force Random Split in preprocessing
    # The load_and_preprocess_dl function falls back to stratified shuffle split if COSMIC_ID is missing.
    if 'COSMIC_ID' in df.columns:
        print("   -> 🕵️ Hiding COSMIC_ID to force Random Split (Simulating Leakage)...")
        df.rename(columns={'COSMIC_ID': 'COSMIC_ID_HIDDEN'}, inplace=True)
    
    df.to_csv(TEMP_PATH, index=False)

    df.to_csv(TEMP_PATH, index=False)

    modes = [True, False]
    results = {}

    for use_id in modes:
        mode_str = "With_IDs" if use_id else "No_IDs"
        print(f"\n🔄 Running Random Split Test: {mode_str} ...")

        # 3. Load Data
        # This will print a warning about "COSMIC_ID not found", which is expected.
        X_train_tuple, X_test_tuple, y_train, y_test, feature_names, encoders, dims = load_and_preprocess_dl(
            TEMP_PATH, include_ids=use_id
        )

        X_cell_train = X_train_tuple[0] 
        print(f"   -> Data Loaded. Train Samples: {X_cell_train.shape[0]}, Test Samples: {X_test_tuple[0].shape[0]}")

        # 4. Initialize Model
        # Unpack dimensions
        target_dim, pathway_dim, drug_vocab_size = dims
        
        # Use best params but fewer epochs for speed
        model_wrapper = GDSCNeuralNetwork(
            cell_input_dim=X_cell_train.shape[1],
            drug_input_dim=X_train_tuple[1].shape[1],
            target_input_dim=target_dim,
            pathway_input_dim=pathway_dim,
            drug_vocab_size=drug_vocab_size,
            learning_rate=0.001
        )

        # 5. Train
        print(f"   -> Training {mode_str} Model (10 Epochs)...")
        
        history = model_wrapper.fit(
            X_train_tuple, y_train,
            X_val_list=None, y_val=None, 
            epochs=10, 
            batch_size=64
        )

        # 6. Evaluate
        print(f"   -> Evaluating {mode_str}...")
        y_pred = model_wrapper.predict(X_test_tuple)
        metrics = evaluate_dl_model(y_test, y_pred)
        results[mode_str] = metrics['R2']

    print("\n" + "="*60)
    print("🧪 RANDOM SPLIT TEST RESULTS (Data Leakage Analysis)")
    print("="*60)
    print(f"With_IDs (Memorization):  R² = {results['With_IDs']:.4f}")
    print(f"No_IDs   (Cell Leakage):  R² = {results['No_IDs']:.4f}")
    print("-" * 60)
    print("Comparison:")
    print(f"Impact of Drug ID Memorization: +{results['With_IDs'] - results['No_IDs']:.4f}")
    print("="*60)

    # Cleanup
    if os.path.exists(TEMP_PATH):
        os.remove(TEMP_PATH)

if __name__ == "__main__":
    run_leakage_test()
