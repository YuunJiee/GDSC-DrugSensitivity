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

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Embedding, Flatten
from tensorflow.keras.models import Model

class GDSCNeuralNetwork:
    def __init__(self, cell_input_dim, drug_input_dim, target_input_dim=None, pathway_input_dim=None, drug_vocab_size=0, learning_rate=0.001, hyperparams=None):
        self.cell_input_dim = cell_input_dim
        self.drug_input_dim = drug_input_dim
        self.target_input_dim = target_input_dim
        self.pathway_input_dim = pathway_input_dim
        self.drug_vocab_size = drug_vocab_size # For Drug ID Embedding
        self.learning_rate = learning_rate
        
        # Default hyperparameters (Optimized)
        self.hp = ModelConfig.BEST_DL_PARAMS.copy()
        
        # Update with provided hyperparams if any
        if hyperparams:
            self.hp.update(hyperparams)
            
        self.model = self._build_model()
        self.scaler_cell = StandardScaler()
        # Scale drug numeric only
        self.scaler_drug = StandardScaler()
        
    def _build_model(self):
        """建立多輸入深度神經網路模型 (Multi-Input Network)"""
        inputs_list = []
        features_to_concat = []

        # --- Branch 1: Cell Line Features ---
        cell_input = Input(shape=(self.cell_input_dim,), name='cell_input')
        inputs_list.append(cell_input)
        x_cell = cell_input
        
        for i in range(self.hp['cell_layers']):
            units = self.hp['cell_units'][i] if isinstance(self.hp['cell_units'], list) else self.hp['cell_units']
            x_cell = Dense(units, activation='relu')(x_cell)
            x_cell = BatchNormalization()(x_cell)
            x_cell = Dropout(self.hp['cell_dropout'])(x_cell) 
        features_to_concat.append(x_cell)

        # --- Branch 2: Drug Numerical Features ---
        drug_num_input = Input(shape=(max(1, self.drug_input_dim),), name='drug_numeric_input')
        inputs_list.append(drug_num_input)
        
        if self.drug_input_dim > 0:
            x_drug = drug_num_input
            for i in range(self.hp['drug_layers']):
                units = self.hp['drug_units'][i] if isinstance(self.hp['drug_units'], list) else self.hp['drug_units']
                x_drug = Dense(units, activation='relu')(x_drug)
                x_drug = BatchNormalization()(x_drug)
                x_drug = Dropout(self.hp['drug_dropout'])(x_drug)
            features_to_concat.append(x_drug)
        else:
            # If 0 dim, we expect dummy input (shape=(1,)) which contains all zeros
            # We MUST connect it to the graph. We can just append it to features.
            # Input is already float32 by default.
            x_drug = drug_num_input
            features_to_concat.append(x_drug)

        # --- Branch 3: Drug ID Embedding (New) ---
        drug_id_input = Input(shape=(1,), name='drug_id_input')
        inputs_list.append(drug_id_input)
        
        if self.drug_vocab_size > 0:
            # Embedding dimension
            drug_emb_dim = 16 
            x_drug_id = Embedding(input_dim=self.drug_vocab_size + 1, output_dim=drug_emb_dim, name='drug_embedding')(drug_id_input)
            x_drug_id = Flatten()(x_drug_id)
            features_to_concat.append(x_drug_id)
        else:
            # No embedding (No_IDs mode or empty)
            # Dummy input (zeros), append directly
            x_drug_id = drug_id_input
            features_to_concat.append(x_drug_id)

        # --- Branch 4: Target Features (Multi-Hot) ---
        target_emb_dim = 16 
        target_input = Input(shape=(self.target_input_dim,), name='target_input')
        inputs_list.append(target_input)
        
        x_target = Dense(target_emb_dim, activation='relu', name='target_projection')(target_input)
        features_to_concat.append(x_target)
        
        # --- Branch 5: Pathway Features (One-Hot) ---
        pathway_emb_dim = 16
        pathway_input = Input(shape=(self.pathway_input_dim,), name='pathway_input')
        inputs_list.append(pathway_input)
        
        x_pathway = Dense(pathway_emb_dim, activation='relu', name='pathway_projection')(pathway_input)
        features_to_concat.append(x_pathway)
        
        # --- Fusion Layer ---
        combined = Concatenate()(features_to_concat)
        x = combined
        
        for i in range(self.hp['fusion_layers']):
            units = self.hp['fusion_units'][i] if isinstance(self.hp['fusion_units'], list) else self.hp['fusion_units']
            if i == 0:
                x = Dense(units, activation='relu', name='fusion_dense')(x)
                x = Dropout(self.hp['fusion_dropout'])(x)
            else:
                x = Dense(units, activation='relu')(x)
        
        output = Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs_list, outputs=output, name='Multi_Input_Network')
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mse']
        )
        return model

    def fit(self, X_train_list, y_train, X_val_list=None, y_val=None, epochs=100, batch_size=64):
        """
        訓練模型
        :param X_train_list: [X_cell, X_drug_num, X_drug_id, X_target, X_pathway]
        """
        X_cell_train, X_drug_num_train, X_drug_id_train, X_target_train, X_pathway_train = X_train_list
        
        # Scale numerical data separately
        X_cell_train_scaled = self.scaler_cell.fit_transform(X_cell_train)
        
        # Handle X_drug numeric
        if self.drug_input_dim > 0:
            X_drug_num_train_scaled = self.scaler_drug.fit_transform(X_drug_num_train)
        else:
            # Create dummy zeros
            X_drug_num_train_scaled = np.zeros((X_drug_num_train.shape[0], 1))
            
        train_inputs = [X_cell_train_scaled, X_drug_num_train_scaled, X_drug_id_train, X_target_train, X_pathway_train]
        
        validation_data = None
        if X_val_list is not None and y_val is not None:
            X_cell_val, X_drug_num_val, X_drug_id_val, X_target_val, X_pathway_val = X_val_list
            X_cell_val_scaled = self.scaler_cell.transform(X_cell_val)
            
            if self.drug_input_dim > 0:
                X_drug_num_val_scaled = self.scaler_drug.transform(X_drug_num_val)
            else:
                X_drug_num_val_scaled = np.zeros((X_drug_num_val.shape[0], 1))
            
            validation_data = ([X_cell_val_scaled, X_drug_num_val_scaled, X_drug_id_val, X_target_val, X_pathway_val], y_val)
            
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        
        history = self.model.fit(
            train_inputs, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        return history

    def predict(self, X_list):
        """預測"""
        X_cell, X_drug_num, X_drug_id, X_target, X_pathway = X_list
        X_cell_scaled = self.scaler_cell.transform(X_cell)
        
        if self.drug_input_dim > 0:
            X_drug_num_scaled = self.scaler_drug.transform(X_drug_num)
        else:
            X_drug_num_scaled = np.zeros((X_drug_num.shape[0], 1))
            
        return self.model.predict([X_cell_scaled, X_drug_num_scaled, X_drug_id, X_target, X_pathway], verbose=0).flatten()

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

def run_cv_deep_learning_dummy(X, y, n_splits=5, epochs=50, batch_size=64):
    """
    Placeholder.
    """
    print("⚠️ K-Fold CV not adapted yet.")
    return 0.0, 0.0

def run_deep_learning_pipeline(X_train_tuple, X_test_tuple, y_train, y_test, feature_names, input_dims):
    """
    執行完整的深度學習 Pipeline
    """
    print("\n" + "="*60)
    print("深度學習模型訓練開始 (Multi-Input with Drug Embedding)")
    print("="*60)
    
    check_gpu_availability()
    
    # Unpack 5 elements
    X_cell_train, X_drug_num_train, X_drug_id_train, X_target_train, X_pathway_train = X_train_tuple
    X_cell_test, X_drug_num_test, X_drug_id_test, X_target_test, X_pathway_test = X_test_tuple
    
    # Unpack 3 dims
    target_dim, pathway_dim, drug_vocab_size = input_dims
    
    # 1. 資料格式轉換 (Ensure arrays)
    def to_numpy(x): return x.values if hasattr(x, 'values') else x
    
    X_cell_train = to_numpy(X_cell_train)
    X_drug_num_train = to_numpy(X_drug_num_train)
    X_drug_id_train = to_numpy(X_drug_id_train) # ID (int)
    # Target/Pathway already numpy
    
    X_cell_test = to_numpy(X_cell_test)
    X_drug_num_test = to_numpy(X_drug_num_test)
    X_drug_id_test = to_numpy(X_drug_id_test)
    
    y_train = to_numpy(y_train)
    y_test = to_numpy(y_test)
    
    # 3. 驗證集切分
    # Split all 6 arrays
    arrays = [X_cell_train, X_drug_num_train, X_drug_id_train, X_target_train, X_pathway_train, y_train]
    split_res = train_test_split(
        *arrays,
        test_size=ModelConfig.TEST_SIZE, random_state=ModelConfig.RANDOM_SEED
    )
    
    X_cell_t, X_cell_v = split_res[0], split_res[1]
    X_drug_num_t, X_drug_num_v = split_res[2], split_res[3]
    X_drug_id_t, X_drug_id_v = split_res[4], split_res[5]
    X_target_t, X_target_v = split_res[6], split_res[7]
    X_pathway_t, X_pathway_v = split_res[8], split_res[9]
    y_t, y_v = split_res[10], split_res[11]
    
    # 4. 建立與訓練模型
    nn_model = GDSCNeuralNetwork(
        cell_input_dim=X_cell_t.shape[1],
        drug_input_dim=X_drug_num_t.shape[1],
        target_input_dim=target_dim,
        pathway_input_dim=pathway_dim,
        drug_vocab_size=drug_vocab_size,
        learning_rate=ModelConfig.DL_PARAMS['learning_rate']
    )
    
    history = nn_model.fit(
        [X_cell_t, X_drug_num_t, X_drug_id_t, X_target_t, X_pathway_t], y_t,
        [X_cell_v, X_drug_num_v, X_drug_id_v, X_target_v, X_pathway_v], y_v,
        epochs=ModelConfig.DL_PARAMS['epochs'], 
        batch_size=ModelConfig.DL_PARAMS['batch_size']
    )
    
    # 5. 評估模型
    y_pred = nn_model.predict([X_cell_test, X_drug_num_test, X_drug_id_test, X_target_test, X_pathway_test])
    metrics = evaluate_dl_model(y_test, y_pred)
    
    return (nn_model.scaler_cell, nn_model.scaler_drug), nn_model.model, y_pred, metrics, history
