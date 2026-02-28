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


def calculate_directional_accuracy(y_true, y_pred):
    """
    คำนวณความแม่นยำในการทำนายทิศทาง (Directional Accuracy / Hit Rate)
    
    วัดว่าโมเดลทำนายทิศทางการเปลี่ยนแปลงราคาถูกต้องกี่เปอร์เซ็นต์
    (ราคาขึ้น/ราคาลง เทียบกับค่าก่อนหน้า)
    
    ค่า > 50% หมายความว่าดีกว่าการสุ่ม (random guess)
    
    Args:
        y_true: ค่าจริง (actual prices)
        y_pred: ค่าที่ทำนาย (predicted prices)
        
    Returns:
        float: Directional Accuracy (0-100%)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) < 2:
        return 0.0
    
    # คำนวณทิศทางจริง: ราคาขึ้น (+1) หรือลง (-1) เทียบกับ step ก่อนหน้า
    true_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
    
    # นับจำนวนที่ทำนายทิศทางถูก
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    
    return (correct / total) * 100

