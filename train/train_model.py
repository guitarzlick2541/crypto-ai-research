"""
train/train_model.py
====================
Build and train LSTM/GRU models with improved architecture and callbacks.
"""

import os
import sys
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from evaluation.metrics import calculate_mae, calculate_rmse, calculate_mape
from utils.scaling import inverse_transform_close
import config
from preprocessing import DataProcessor


def build_model(model_type, input_shape):
    """
    Build LSTM or GRU Model Structure (Enhanced Architecture)
    
    Architecture: 128 → 64 → 32 → Dense(16) → Dense(1)
    พร้อม Dropout + BatchNormalization
    
    Args:
        model_type (str): 'LSTM' or 'GRU'
        input_shape (tuple): (time_steps, num_features)
    """
    model = Sequential()
    
    LayerClass = LSTM if model_type.upper() == 'LSTM' else GRU
    
    # Layer 1 - 128 units
    model.add(LayerClass(units=128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(config.DROPOUT_RATE_HIGH))
    model.add(BatchNormalization())
    
    # Layer 2 - 64 units
    model.add(LayerClass(units=64, return_sequences=True))
    model.add(Dropout(config.DROPOUT_RATE_HIGH))
    model.add(BatchNormalization())
    
    # Layer 3 - 32 units
    model.add(LayerClass(units=32, return_sequences=False))
    model.add(Dropout(config.DROPOUT_RATE_LOW))
    
    # Dense Layers
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(coin, timeframe, model_type="LSTM"):
    """
    Main function to train a model (LSTM/GRU) with improved pipeline
    
    Args:
        coin (str): 'btc' or 'eth'
        timeframe (str): '1h' or '4h'
        model_type (str): 'LSTM' or 'GRU'
        
    Returns:
        tuple: (mae, rmse, mape)
    """
    config._ensure_directories(config.MODEL_DIR, config.RESULT_DIR)
    print(f"\n🚀 Starting {model_type} Training for {coin.upper()} ({timeframe})...")
    
    # 1. Load Data
    processor = DataProcessor()
    try:
        df = processor.load_data(coin, timeframe)
    except FileNotFoundError:
        print(f"❌ Skip: Data file not found for {coin} {timeframe}")
        return None, None, None

    # 2. Split & Scale (Multi-Feature with Data Leakage Prevention)
    train_scaled, test_scaled, scaler = processor.get_train_test_data(df)
    
    print(f"   Features: {len(config.FEATURE_COLUMNS)} ({', '.join(config.FEATURE_COLUMNS[:5])}...)")
    
    # 3. Create Sliding Windows (Multi-Feature)
    X_train, y_train = processor.create_sliding_window(train_scaled)
    X_test, y_test = processor.create_sliding_window(test_scaled)
    
    print(f"   Train data shape: {X_train.shape}")
    print(f"   Test data shape:  {X_test.shape}")
    
    # 4. Build & Train Model
    model = build_model(model_type, (X_train.shape[1], X_train.shape[2]))
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    model.fit(X_train, y_train, 
              batch_size=config.BATCH_SIZE, 
              epochs=config.EPOCHS,
              validation_split=config.VALIDATION_SPLIT,
              callbacks=callbacks,
              verbose=1)
    
    # 5. Save Model & Scaler
    model_path = os.path.join(config.MODEL_DIR, f"{model_type.lower()}_{coin.lower()}_{timeframe}.h5")
    model.save(model_path)
    print(f"✅ Model saved to: {model_path}")
    
    scaler_path = os.path.join(config.MODEL_DIR, f"scaler_{coin.lower()}_{timeframe}.save")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # 6. Evaluate (using shared inverse_transform utility)
    predictions = model.predict(X_test)
    
    predictions_rescaled = inverse_transform_close(scaler, predictions)
    y_test_rescaled = inverse_transform_close(scaler, y_test)
    
    mae = calculate_mae(y_test_rescaled, predictions_rescaled)
    rmse = calculate_rmse(y_test_rescaled, predictions_rescaled)
    mape = calculate_mape(y_test_rescaled, predictions_rescaled)
    
    print(f"📊 Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    return mae, rmse, mape
