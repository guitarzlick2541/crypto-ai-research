"""
utils/indicators.py
====================
Shared Technical Indicator computation module.
Used by BOTH training (preprocessing.py) AND inference (predictor_fixed.py)
to ensure consistent feature engineering.

⚠️ CRITICAL: Any change here affects both training and inference.
   If you modify indicators, you MUST retrain all models.
"""

import pandas as pd
import numpy as np

# ป้องกัน division by zero
EPSILON = 1e-10


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณ Technical Indicators จาก OHLCV data
    
    ใช้ร่วมกันระหว่าง training pipeline และ inference pipeline
    เพื่อให้แน่ใจว่าสูตรคำนวณตรงกัน 100%
    
    Indicators:
    - SMA (7, 25)
    - EMA (12, 26)
    - RSI (14)
    - MACD + Signal Line
    - Bollinger Bands (upper, lower)
    - Price Returns (% change)
    
    Args:
        df (pd.DataFrame): ต้องมี column 'close' เป็นอย่างน้อย
        
    Returns:
        pd.DataFrame: ข้อมูลพร้อม Technical Indicators (copy ใหม่)
    """
    df = df.copy()
    close = df['close']
    
    # SMA (Simple Moving Average)
    df['sma_7'] = close.rolling(window=7, min_periods=1).mean()
    df['sma_25'] = close.rolling(window=25, min_periods=1).mean()
    
    # EMA (Exponential Moving Average)
    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['ema_26'] = close.ewm(span=26, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + EPSILON)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    sma_20 = close.rolling(window=20, min_periods=1).mean()
    std_20 = close.rolling(window=20, min_periods=1).std().fillna(0)
    df['bb_upper'] = sma_20 + (std_20 * 2)
    df['bb_lower'] = sma_20 - (std_20 * 2)
    
    # Price Returns (% change)
    df['returns'] = close.pct_change().fillna(0)
    
    # เติมค่า NaN ที่เกิดจาก rolling
    df = df.ffill().bfill()
    
    return df
