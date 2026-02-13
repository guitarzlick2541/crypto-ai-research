"""
utils/scaling.py
================
Shared utility for inverse-transforming predictions from multi-feature scalers.
Used by training, evaluation, and inference to eliminate code duplication.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def inverse_transform_close(scaler: MinMaxScaler, values: np.ndarray, target_index: int = 0) -> np.ndarray:
    """
    Inverse transform เฉพาะ target column (close price) จาก multi-feature scaler
    
    ใช้คุณสมบัติของ MinMaxScaler ที่ scale แต่ละ feature อิสระจากกัน
    โดยใช้ scaler.min_ และ scaler.scale_ โดยตรง
    
    Formula: x_original = (x_scaled - min_[i]) / scale_[i]
    ซึ่ง MinMaxScaler เก็บ: x_scaled = x * scale_[i] + min_[i]
    ดังนั้น: x_original = (x_scaled - min_[i]) / scale_[i]
    
    Args:
        scaler: fitted MinMaxScaler (multi-feature)
        values: np.ndarray of scaled values (1D or 2D)
        target_index: index ของ feature ที่ต้องการ inverse (default: 0 = close)
        
    Returns:
        np.ndarray: inverse-transformed values (same shape as input)
    """
    values_flat = np.asarray(values).flatten()
    
    # ใช้สูตร inverse ตรงจาก scaler parameters
    # MinMaxScaler: X_scaled = X * scale_ + min_
    # Inverse:      X = (X_scaled - min_) / scale_
    result = (values_flat - scaler.min_[target_index]) / scaler.scale_[target_index]
    
    return result.reshape(values.shape) if hasattr(values, 'shape') and len(values.shape) > 1 else result
