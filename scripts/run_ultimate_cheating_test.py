
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from src.models.deep_learning.deep_learning_model import GDSCNeuralNetwork
from src.utils.evaluation import evaluate_dl_model
from src.config import ModelConfig

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_cheating(file_path):
    """
    Modified preprocessing that DELIBERATELY INCLUDES 'AUC' and 'Z_SCORE'
    """
    print(f"🕵️ Loading data for Cheating Test: {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Hide COSMIC_ID to force Random Split (Simulating standard Kaggle approach)
    if 'COSMIC_ID' in df.columns:
        df.rename(columns={'COSMIC_ID': 'COSMIC_ID_HIDDEN'}, inplace=True)

    # 2. Cleanup
    df_clean = df.dropna(subset=['LN_IC50', 'AUC', 'Z_SCORE']) # Ensure these exist
    
    # 3. Define Features - THE CHEAT
    # We normally drop AUC and Z_SCORE. Here we keep them.
    # We will treat them as "Numerical Features" and feed them into the 'Cell' branch 
    # (or Drug Numeric branch) for simplicity. 
    
    exclude_cols = [
        'CELL_LINE_NAME', 'DRUG_NAME', 
        'NLME_RESULT_ID', 'NLME_CURVE_ID', 'SANGER_MODEL_ID',
        'LN_IC50', 'TARGET', 'TARGET_PATHWAY',
        'Gene Expression', 'CNA', 'Methylation', 'Drug Response', 
        'Exome mutation', 'Whole Genome Sequencing (WGS)',
        # 'AUC', 'Z_SCORE'  <-- REVEALED!
    ]
    
    # Identify Cell inclusions (+ The Cheat Features)
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
    
    # Exclude ID columns if they exist (we will handle Drug ID separately)
    if 'DRUG_ID' in feature_cols: feature_cols.remove('DRUG_ID')
    if 'COSMIC_ID_HIDDEN' in feature_cols: feature_cols.remove('COSMIC_ID_HIDDEN')

    print(f"   -> Including {len(feature_cols)} features (Contains AUC/Z_SCORE? {'AUC' in feature_cols})")

    # One-Hot / Processing
    X_numeric = df_clean[feature_cols].copy()
    
    # Simple encoding for object cols if any (unlikely for AUC/Z_SCORE but for others)
    X_numeric = pd.get_dummies(X_numeric, drop_first=True).astype(float)
    
    # 4. Target/Pathway/DrugID (Standard)
    # Target
    mlb_target = MultiLabelBinarizer()
    targets = df_clean['TARGET'].fillna('Unknown').astype(str).apply(lambda x: [t.strip() for t in x.split(',')])
    X_target = mlb_target.fit_transform(targets)
    
    # Pathway
    X_pathway = pd.get_dummies(df_clean['TARGET_PATHWAY'].fillna('Unknown')).astype(float).values
    
    # Drug ID
    drug_le = LabelEncoder()
    X_drug_id = drug_le.fit_transform(df_clean['DRUG_ID'].astype(str)).reshape(-1, 1)
    drug_vocab_size = len(drug_le.classes_)
    
    # Dummy Drug Numeric (since we put everything in "Cell/General" branch)
    X_drug_num = np.zeros((len(df_clean), 1))
    
    y = df_clean['LN_IC50'].values
    
    # 5. Split (Random Split because COSMIC_ID is hidden)
    # Using simple train_test_split
    X_num_tr, X_num_te, \
    X_dnum_tr, X_dnum_te, \
    X_did_tr, X_did_te, \
    X_tgt_tr, X_tgt_te, \
    X_path_tr, X_path_te, \
    y_tr, y_te = train_test_split(
        X_numeric.values, X_drug_num, X_drug_id, X_target, X_pathway, y,
        test_size=0.2, random_state=42
    )

    return (X_num_tr, X_dnum_tr, X_did_tr, X_tgt_tr, X_path_tr), \
           (X_num_te, X_dnum_te, X_did_te, X_tgt_te, X_path_te), \
           y_tr, y_te, \
           (X_target.shape[1], X_pathway.shape[1], drug_vocab_size)


def run_test():
    DATA_PATH = 'data/processed/Data_imputed.csv'
    
    print("🚀 Running ULTIMATE CHEATING TEST (Inputs: AUC + Z_SCORE + IDs + Random Split)...")
    
    # Load
    X_train, X_test, y_train, y_test, dims = load_and_preprocess_cheating(DATA_PATH)
    
    # Model
    print("   -> Initializing Model...")
    model = GDSCNeuralNetwork(
        cell_input_dim=X_train[0].shape[1], # This now includes AUC/Z_SCORE
        drug_input_dim=1,
        target_input_dim=dims[0],
        pathway_input_dim=dims[1],
        drug_vocab_size=dims[2],
        learning_rate=0.001
    )
    
    # Train
    print("   -> Training (5 Epochs is enough to see the magic)...")
    model.fit(X_train, y_train, epochs=5, batch_size=64)
    
    # Eval
    print("   -> Evaluating...")
    y_pred = model.predict(X_test)
    metrics = evaluate_dl_model(y_test, y_pred)
    
    print("\n" + "="*50)
    print("🛑 ULTIMATE CHEATING RESULTS (With AUC/Z_SCORE)")
    print("="*50)
    print(f"R² Score: {metrics['R2']:.5f}")
    print(f"RMSE:     {metrics['RMSE']:.5f}")
    print("="*50)

if __name__ == "__main__":
    run_test()
