"""
深度學習模型 - Neural Network Regression
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

from src.config import ModelConfig
from src.utils.evaluation import evaluate_dl_model
# Visualization is now handled by the caller, not inside the model class

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model

class GDSCNeuralNetwork:
    def __init__(self, cell_input_dim, drug_input_dim, learning_rate=0.001, hyperparams=None):
        self.cell_input_dim = cell_input_dim
        self.drug_input_dim = drug_input_dim
        self.learning_rate = learning_rate
        
        # Default hyperparameters (Optimized)
        self.hp = ModelConfig.BEST_DL_PARAMS.copy()
        
        # Update with provided hyperparams if any
        if hyperparams:
            self.hp.update(hyperparams)
            
        self.model = self._build_model()
        self.scaler_cell = StandardScaler()
        self.scaler_drug = StandardScaler()
        
    def _build_model(self):
        """建立雙塔深度神經網路模型 (Dual-Branch Network)"""
        # --- Branch 1: Cell Line Features ---
        cell_input = Input(shape=(self.cell_input_dim,), name='cell_input')
        x_cell = cell_input
        
        for i in range(self.hp['cell_layers']):
            units = self.hp['cell_units'][i] if isinstance(self.hp['cell_units'], list) else self.hp['cell_units']
            x_cell = Dense(units, activation='relu')(x_cell)
            x_cell = BatchNormalization()(x_cell)
            if i < self.hp['cell_layers'] - 1: # No dropout after last layer of branch usually, but original had. Keeping original logic roughly.
                 # Wait, original had dropout after 1st layer, not 2nd. 
                 # Let's simple apply dropout after each block except maybe the last one before concat if needed, 
                 # but usually dropout is fine everywhere.
                 pass
            x_cell = Dropout(self.hp['cell_dropout'])(x_cell)

        # --- Branch 2: Drug Features ---
        drug_input = Input(shape=(self.drug_input_dim,), name='drug_input')
        x_drug = drug_input
        
        for i in range(self.hp['drug_layers']):
            units = self.hp['drug_units'][i] if isinstance(self.hp['drug_units'], list) else self.hp['drug_units']
            x_drug = Dense(units, activation='relu')(x_drug)
            x_drug = BatchNormalization()(x_drug)
            x_drug = Dropout(self.hp['drug_dropout'])(x_drug)
        
        # --- Fusion Layer ---
        combined = Concatenate()([x_cell, x_drug])
        x = combined
        
        for i in range(self.hp['fusion_layers']):
            units = self.hp['fusion_units'][i] if isinstance(self.hp['fusion_units'], list) else self.hp['fusion_units']
            if i == 0:
                x = Dense(units, activation='relu', name='fusion_dense')(x)
                x = Dropout(self.hp['fusion_dropout'])(x)
            else:
                x = Dense(units, activation='relu')(x)
        
        output = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=[cell_input, drug_input], outputs=output, name='Dual_Branch_Network')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mse']
        )
        return model

    def fit(self, X_train_list, y_train, X_val_list=None, y_val=None, epochs=100, batch_size=64):
        """
        訓練模型
        :param X_train_list: [X_cell_train, X_drug_train]
        """
        X_cell_train, X_drug_train = X_train_list
        
        # Scale data separately
        X_cell_train_scaled = self.scaler_cell.fit_transform(X_cell_train)
        X_drug_train_scaled = self.scaler_drug.fit_transform(X_drug_train)
        
        validation_data = None
        if X_val_list is not None and y_val is not None:
            X_cell_val, X_drug_val = X_val_list
            X_cell_val_scaled = self.scaler_cell.transform(X_cell_val)
            X_drug_val_scaled = self.scaler_drug.transform(X_drug_val)
            validation_data = ([X_cell_val_scaled, X_drug_val_scaled], y_val)
            
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        
        history = self.model.fit(
            [X_cell_train_scaled, X_drug_train_scaled], y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        return history

    def predict(self, X_list):
        """預測"""
        X_cell, X_drug = X_list
        X_cell_scaled = self.scaler_cell.transform(X_cell)
        X_drug_scaled = self.scaler_drug.transform(X_drug)
        return self.model.predict([X_cell_scaled, X_drug_scaled], verbose=0).flatten()

def check_gpu_availability():
    """檢查並顯示 GPU 狀態"""
    print("\n" + "="*60)
    print("🔍 硬體設備檢測")
    print("="*60)
    print(f"TensorFlow 版本: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ 找到 {len(gpus)} 個 GPU: {[gpu.name for gpu in gpus]}")
    else:
        print("⚠️  未找到 GPU，將使用 CPU 訓練")
    print("="*60 + "\n")

def run_cv_deep_learning(X, y, n_splits=5, epochs=50, batch_size=64):
    """
    執行 K-Fold Cross-Validation
    """
    print(f"\n🔄 執行 {n_splits}-Fold Cross-Validation...")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=ModelConfig.RANDOM_SEED)
    
    fold_metrics = []
    
    # Pre-scale X for CV to avoid fitting scaler inside loop if we assume X is already numeric
    # But strictly speaking we should fit scaler on train fold only.
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n   Fold {fold+1}/{n_splits}")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Initialize model
        nn_model = GDSCNeuralNetwork(input_dim=X.shape[1])
        
        # Train
        # Note: fit() handles scaling internally
        nn_model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, epochs=epochs, batch_size=batch_size)
        
        # Evaluate
        y_pred = nn_model.predict(X_val_fold)
        r2 = r2_score(y_val_fold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        
        print(f"     -> R2: {r2:.4f}, RMSE: {rmse:.4f}")
        fold_metrics.append({'R2': r2, 'RMSE': rmse})
        
    # 計算平均指標
    avg_r2 = np.mean([m['R2'] for m in fold_metrics])
    avg_rmse = np.mean([m['RMSE'] for m in fold_metrics])
    
    print(f"\n✅ CV 結果: 平均 R2 = {avg_r2:.4f}, 平均 RMSE = {avg_rmse:.4f}")
    return avg_r2, avg_rmse

def run_deep_learning_pipeline(X_cell_train, X_drug_train, X_cell_test, X_drug_test, y_train, y_test, feature_names):
    """
    執行完整的深度學習 Pipeline (Dual-Branch)
    """
    print("\n" + "="*60)
    print("深度學習模型訓練開始 (Dual-Branch Architecture)")
    print("="*60)
    
    check_gpu_availability()
    
    # 1. 資料格式轉換
    if hasattr(X_cell_train, 'values'): X_cell_train = X_cell_train.values
    if hasattr(X_drug_train, 'values'): X_drug_train = X_drug_train.values
    if hasattr(X_cell_test, 'values'): X_cell_test = X_cell_test.values
    if hasattr(X_drug_test, 'values'): X_drug_test = X_drug_test.values
    if hasattr(y_train, 'values'): y_train = y_train.values
    if hasattr(y_test, 'values'): y_test = y_test.values
    
    # 3. 驗證集切分 (需同時切分 Cell 和 Drug)
    # train_test_split 可以同時處理多個 array
    X_cell_train_final, X_cell_val, X_drug_train_final, X_drug_val, y_train_final, y_val = train_test_split(
        X_cell_train, X_drug_train, y_train, 
        test_size=ModelConfig.TEST_SIZE, random_state=ModelConfig.RANDOM_SEED
    )
    
    # 4. 建立與訓練模型
    nn_model = GDSCNeuralNetwork(
        cell_input_dim=X_cell_train.shape[1],
        drug_input_dim=X_drug_train.shape[1],
        learning_rate=ModelConfig.DL_PARAMS['learning_rate']
    )
    
    history = nn_model.fit(
        [X_cell_train_final, X_drug_train_final], y_train_final, 
        [X_cell_val, X_drug_val], y_val, 
        epochs=ModelConfig.DL_PARAMS['epochs'], 
        batch_size=ModelConfig.DL_PARAMS['batch_size']
    )
    
    # 5. 評估模型
    y_pred = nn_model.predict([X_cell_test, X_drug_test])
    metrics = evaluate_dl_model(y_test, y_pred)
    
    # Return objects for visualization in main.py or caller
    # 回傳兩個 scaler (tuple)
    return (nn_model.scaler_cell, nn_model.scaler_drug), nn_model.model, y_pred, metrics, history
