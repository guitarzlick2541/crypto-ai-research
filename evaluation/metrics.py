import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mae(y_true, y_pred):
    """ค่าเฉลี่ยความคลาดเคลื่อนสัมบูรณ์ (Mean Absolute Error)"""
    return mean_absolute_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    """รากที่สองของค่าเฉลี่ยความคลาดเคลื่อนกำลังสอง (Root Mean Squared Error)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """ค่าเฉลี่ยร้อยละความคลาดเคลื่อนสัมบูรณ์ (Mean Absolute Percentage Error)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # ป้องกันการหารด้วยศูนย์ (Avoid division by zero)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
