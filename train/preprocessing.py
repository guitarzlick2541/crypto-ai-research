"""
train/preprocessing.py
======================
โมดูลสำหรับเตรียมข้อมูล (Data Preprocessing) สำหรับงานวิจัย Crypto AI

โมดูลนี้จัดการกระบวนการ ETL (Extract, Transform, Load) สำหรับ:
1. การดึงข้อมูลดิบจากไฟล์ CSV
2. การคำนวณ Technical Indicators (ผ่าน shared utils/indicators.py)
3. การแปลงข้อมูล (Normalization) แบบ Multi-Feature
4. การสร้าง Sliding Window Dataset สำหรับ LSTM/GRU
"""

import os
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler

# Ensure we can find project modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

try:
    import config
except ImportError:
    from train import config

from utils.indicators import compute_technical_indicators


class DataProcessor:
    def __init__(self):
        """เริ่มต้น DataProcessor พร้อมกับ MinMaxScaler (multi-feature)"""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self, coin, timeframe):
        """
        โหลดข้อมูลดิบจากไฟล์ CSV
        
        Args:
            coin (str): สัญลักษณ์เหรียญ (เช่น 'btc')
            timeframe (str): ช่วงเวลา (เช่น '1h')
            
        Returns:
            pd.DataFrame: ข้อมูลที่โหลดมา
        """
        filename = f"{coin.lower()}_{timeframe}.csv"
        filepath = os.path.join(config.DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Data file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        return df

    def validate_data(self, df, coin, timeframe):
        """
        ตรวจสอบความถูกต้องของข้อมูล (Data Integrity)
        """
        print(f"   Validating {coin.upper()} {timeframe} data...")
        
        if df.empty:
            raise ValueError(f"❌ Dataframe is empty for {coin} {timeframe}")
            
        required_cols = ['close', 'open', 'high', 'low', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"❌ Missing '{col}' column in {coin} {timeframe}")
            
        # ตรวจสอบค่า NaN ใน column close
        if df['close'].isnull().any():
            print(f"      ⚠️ Warning: Found NaN values in 'close'. Dropping...")
            df.dropna(subset=['close'], inplace=True)
            
        print("      Validation passed.")
        return df

    # -----------------------------------------------------------------
    # Train/Test Split & Scaling (Multi-Feature)
    # -----------------------------------------------------------------
    def get_train_test_data(self, df, split_ratio=0.8):
        """
        แบ่งข้อมูล Train/Test และทำ Scaling อย่างถูกต้อง (ป้องกัน Data Leakage)
        รองรับ Multi-Feature
        
        Process:
        1. เพิ่ม Technical Indicators (via shared module)
        2. Split Data (Train/Test)
        3. Fit Scaler on Train ONLY
        4. Transform Train and Test
        
        Returns:
            tuple: (train_scaled, test_scaled, scaler)
        """
        # เพิ่ม Technical Indicators (ใช้ shared function)
        df = compute_technical_indicators(df)
        
        # เลือก features ที่ต้องการ
        feature_cols = [col for col in config.FEATURE_COLUMNS if col in df.columns]
        data = df[feature_cols].values
        
        train_size = int(len(data) * split_ratio)
        
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Fit Scaler on Train Data ONLY
        self.scaler.fit(train_data)
        
        train_scaled = self.scaler.transform(train_data)
        test_scaled = self.scaler.transform(test_data)
        
        return train_scaled, test_scaled, self.scaler

    def create_sliding_window(self, data, window_size=config.WINDOW_SIZE):
        """
        สร้าง Sliding Window Dataset (X, y) - รองรับ Multi-Feature
        
        Args:
            data: np.array shape (n_samples, n_features)
            window_size: ขนาดของ window
            
        Returns:
            X: np.array shape (samples, window_size, n_features)
            y: np.array shape (samples,) - ทำนายเฉพาะ close (feature index 0)
        """
        X, y = [], []
        
        # ตรวจสอบว่าเป็น multi-feature หรือ single-feature
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i])  # (window_size, n_features)
            y.append(data[i, 0])             # close price (feature index 0)
        
        X, y = np.array(X), np.array(y)
        
        return X, y


def run_preprocessing_pipeline():
    """
    จุดเริ่มต้นหลัก (Main entry point) สำหรับรัน pipeline การเตรียมข้อมูลทั้งหมด
    """
    config._ensure_directories(config.MODEL_DIR, config.RESULT_DIR)
    processor = DataProcessor()
    
    print("="*60)
    print(" STARTING PREPROCESSING PIPELINE")
    print("="*60)
    
    for coin in config.COINS:
        for tf in config.TIMEFRAMES:
            print(f"\ndataset: {coin.upper()} / {tf}")
            try:
                df = processor.load_data(coin, tf)
                df = processor.validate_data(df, coin, tf)
                train_scaled, test_scaled, scaler = processor.get_train_test_data(df)
                print(f"    Train: {train_scaled.shape}, Test: {test_scaled.shape}")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
            
    print("\n" + "="*60)
    print(" PREPROCESSING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_preprocessing_pipeline()
