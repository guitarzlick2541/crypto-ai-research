import os
import sys
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import calculate_mae, calculate_rmse, calculate_mape
import config
from preprocessing import DataProcessor

def build_model(model_type, input_shape):
    """
    Build LSTM or GRU Model Structure
    
    Args:
        model_type (str): 'LSTM' or 'GRU'
        input_shape (tuple): (time_steps, features)
    """
    model = Sequential()
    
    LayerClass = LSTM if model_type.upper() == 'LSTM' else GRU
    
    # Layer 1
    model.add(LayerClass(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    # Layer 2
    model.add(LayerClass(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output Layer
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(coin, timeframe, model_type="LSTM"):
    """
    Main function to train a model (LSTM/GRU)
    
    Args:
        coin (str): 'btc' or 'eth'
        timeframe (str): '1h' or '4h'
        model_type (str): 'LSTM' or 'GRU'
        
    Returns:
        tuple: (mae, rmse, mape)
    """
    print(f"\n🚀 Starting {model_type} Training for {coin.upper()} ({timeframe})...")
    
    # 1. Load Data
    processor = DataProcessor()
    try:
        df = processor.load_data(coin, timeframe)
    except FileNotFoundError:
        print(f"❌ Skip: Data file not found for {coin} {timeframe}")
        return None, None, None

    # 2. Split & Scale (Correctly preventing Data Leakage)
    # This uses the new method in preprocessing.py that splits BEFORE scaling
    train_scaled, test_scaled, scaler = processor.get_train_test_data(df)
    
    # 3. Create Sliding Windows
    X_train, y_train = processor.create_sliding_window(train_scaled)
    X_test, y_test = processor.create_sliding_window(test_scaled)
    
    print(f"   Train data shape: {X_train.shape}")
    print(f"   Test data shape:  {X_test.shape}")
    
    # 4. Build & Train Model
    model = build_model(model_type, (X_train.shape[1], 1))
    
    model.fit(X_train, y_train, 
              batch_size=config.BATCH_SIZE, 
              epochs=config.EPOCHS,
              validation_split=config.VALIDATION_SPLIT,
              verbose=1)
    
    # 5. Save Model & Scaler
    model_path = os.path.join(config.MODEL_DIR, f"{model_type.lower()}_{coin.lower()}_{timeframe}.h5")
    model.save(model_path)
    print(f"✅ Model saved to: {model_path}")
    
    scaler_path = os.path.join(config.MODEL_DIR, f"scaler_{coin.lower()}_{timeframe}.save")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # 6. Evaluate
    predictions = model.predict(X_test)
    
    # Inverse transform
    predictions_rescaled = scaler.inverse_transform(predictions)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = calculate_mae(y_test_rescaled, predictions_rescaled)
    rmse = calculate_rmse(y_test_rescaled, predictions_rescaled)
    mape = calculate_mape(y_test_rescaled, predictions_rescaled)
    
    print(f"📊 Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    return mae, rmse, mape
