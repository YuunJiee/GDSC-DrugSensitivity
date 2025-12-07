import pandas as pd
import numpy as np
import os
import argparse
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

def impute_tissue_data(df):
    """Imputes tissue descriptors and cancer type globally."""
    print("Imputing tissue data...")
    tissue_cols = ['GDSC Tissue descriptor 1', 'GDSC Tissue descriptor 2', 'Cancer Type (matching TCGA label)', 'TCGA_DESC']
    
    # 1. Try to fill based on other tissue columns (consistency check)
    for col in tissue_cols:
        if df[col].isnull().any():
            # Group by other tissue columns to find the mode
            others = [c for c in tissue_cols if c != col]
            for other in others:
                if df[other].notnull().any():
                    df[col] = df[col].fillna(df.groupby(other)[col].transform(
                        lambda x: x.mode()[0] if not x.mode().empty else np.nan
                    ))
            
            # 2. Global Mode Fallback
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            
    return df

def impute_drug_features(df):
    """Imputes Drug-specific features (TARGET, TARGET_PATHWAY)."""
    print("Imputing drug features...")
    # These features should be constant for a given drug.
    drug_cols = ['TARGET', 'TARGET_PATHWAY']
    
    for col in drug_cols:
        if col in df.columns:
             df[col] = df.groupby('DRUG_NAME')[col].transform(
                 lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
             )
             # Fallback for drugs that have NO data for these columns
             df[col] = df[col].fillna('Unknown')
             
    return df

def impute_genomic_features(df):
    """Imputes Genomic features (CNA, Gene Expression, Methylation)."""
    print("Imputing genomic features...")
    genomic_cols = ['CNA', 'Gene Expression', 'Methylation']
    
    for col in genomic_cols:
        if col in df.columns:
            # 1. Group by Tissue Descriptor 1 -> Mode
            df[col] = df[col].fillna(df.groupby('GDSC Tissue descriptor 1')[col].transform(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan
            ))
            
            # 2. Global Mode Fallback (or 'Unknown')
            df[col] = df[col].fillna('Unknown')
            
    return df

def impute_other_categorical(df):
    """Imputes other categorical variables."""
    print("Imputing other categorical features...")
    other_cols = ['Microsatellite instability Status (MSI)', 'Screen Medium', 'Growth Properties']
    
    for col in other_cols:
        if col in df.columns:
            # Group by Tissue -> Mode
            df[col] = df[col].fillna(df.groupby('GDSC Tissue descriptor 1')[col].transform(
                lambda x: x.mode()[0] if not x.mode().empty else np.nan
            ))
            df[col] = df[col].fillna('Unknown')
            
    return df

def impute_numeric_features(df):
    """Imputes numeric features (LN_IC50, AUC, Z_SCORE)."""
    print("Imputing numeric features...")
    numeric_cols = ['LN_IC50', 'AUC', 'Z_SCORE']
    
    for col in numeric_cols:
        if col in df.columns:
            # Strategy: Group by Tissue -> Median
            df[col] = df[col].fillna(df.groupby('GDSC Tissue descriptor 1')[col].transform('median'))
            
            # Global Median Fallback
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                
    return df

def main(input_path, output_dir):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # [Report] Save snapshot of missing data for EDA
    print("Identifying missing data for report...")
    rows_with_missing = df[df.isnull().any(axis=1)]
    if not rows_with_missing.empty:
        missing_report_path = os.path.join(output_dir, 'missing_data_snapshot.csv')
        rows_with_missing.to_csv(missing_report_path, index=False)
        print(f"Saved {len(rows_with_missing)} rows with missing values to {missing_report_path}")
    else:
        print("No missing values found in raw data.")
    
    # Check missing before
    print("Missing values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # --- Pipeline ---
    df = impute_tissue_data(df)
    df = impute_drug_features(df)
    df = impute_genomic_features(df)
    df = impute_other_categorical(df)
    df = impute_numeric_features(df)
    # ----------------
    
    # Check missing after
    print("\nMissing values after imputation:")
    missing_after = df.isnull().sum()[df.isnull().sum() > 0]
    if not missing_after.empty:
        print(missing_after)
    else:
        print("No missing values remain.")

    os.makedirs(output_dir, exist_ok=True)
    
    # Save Data_imputed.csv
    output_path = os.path.join(output_dir, 'Data_imputed.csv')
    df.to_csv(output_path, index=False)
    print(f"\nSaved imputed data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GDSC raw data and handle missing values.")
    parser.add_argument('--input', type=str, default='data/raw/GDSC_DATASET.csv', help='Path to raw dataset')
    parser.add_argument('--output', type=str, default='data/processed', help='Directory to save processed data')
    
    args = parser.parse_args()
    
    main(args.input, args.output)
