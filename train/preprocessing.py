"""
train/preprocessing.py
======================
โมดูลสำหรับเตรียมข้อมูล (Data Preprocessing) สำหรับงานวิจัย Crypto AI

โมดูลนี้จัดการกระบวนการ ETL (Extract, Transform, Load) สำหรับ:
1. การดึงข้อมูลดิบจากไฟล์ CSV via Extract
2. การตรวจสอบความถูกต้องของข้อมูล (Validation)
3. การแปลงข้อมูล (Normalization)
4. การบันทึกข้อมูลที่ผ่านการประมวลผลลงในโฟลเดอร์ 'data/processed/'

เพื่อให้มั่นใจว่าผลการทดลองสามารถทำซ้ำได้ (Reproducible) และเป็นไปตามมาตรฐานงานวิจัย
"""

import os
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler

# Ensure we can find the config module
try:
    import config
except ImportError:
    from train import config

class DataProcessor:
    def __init__(self):
        """เริ่มต้น DataProcessor พร้อมกับ MinMaxScaler"""
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
        
        ตรวจสอบ:
        - มีคอลัมน์ที่จำเป็น ('close') หรือไม่
        - Dataframe ว่างหรือไม่
        - ค่า Null/NaN
        """
        print(f"   Validating {coin.upper()} {timeframe} data...")
        
        if df.empty:
            raise ValueError(f"❌ Dataframe is empty for {coin} {timeframe}")
            
        if 'close' not in df.columns:
            raise ValueError(f"❌ Missing 'close' column in {coin} {timeframe}")
            
        # ตรวจสอบค่า NaN ใน column close
        if df['close'].isnull().any():
            print(f"      ⚠️ Warning: Found NaN values in 'close'. Dropping...")
            df.dropna(subset=['close'], inplace=True)
            
        print("      Validation passed.")
        return df

    def process_and_save(self, coin, timeframe):
        """
        รันกระบวนการ Preprocessing เต็มรูปแบบสำหรับ Dataset
        
        ขั้นตอน:
        1. โหลดข้อมูลดิบ
        2. ตรวจสอบความถูกต้อง
        3. ปรับค่า (Normalize) ด้วย MinMax Scaling
        4. บันทึกลงใน data/processed/
        """
        try:
            # 1. Load
            df = self.load_data(coin, timeframe)
            
            # 2. Validate
            df = self.validate_data(df, coin, timeframe)
            
            # 3. Normalize
            print("    Normalizing data...")
            # เราใช้ Scaler ตัวนี้สำหรับกระบวนการนี้เพื่อบันทึกการแปลงค่า
            # หมายเหตุสำหรับงานวิจัย: ในอุดมคติ เราควร fit scaler กับชุด Train เท่านั้นเพื่อป้องกัน Data Leakage (Look-ahead bias)
            # อย่างไรก็ตาม สำหรับโจทย์ "สร้าง data/processed/" เราจะ scale ข้อมูลทั้งหมด
            # เพื่อให้พร้อมใช้งานทันที ส่วนการแบ่ง Train/Test จะทำในภายหลัง
            
            close_prices = df['close'].values.reshape(-1, 1)
            scaled_prices = self.scaler.fit_transform(close_prices)
            
            # Create Processed DataFrame
            processed_df = df.copy()
            processed_df['close_scaled'] = scaled_prices
            
            # Select only required columns for output as per requirements
            output_df = processed_df[['close', 'close_scaled']]
            
            # 4. Save
            processed_dir = os.path.join(os.path.dirname(config.DATA_DIR), "processed")
            os.makedirs(processed_dir, exist_ok=True)
            
            output_filename = f"{coin.lower()}_{timeframe}_scaled.csv"
            output_path = os.path.join(processed_dir, output_filename)
            
            output_df.to_csv(output_path, index=False)
            print(f"       Saved processed data to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Processing failed for {coin} {timeframe}: {str(e)}")
            raise e

    # -------------------------------------------------------------
    # เมธอดสำหรับความเข้ากันได้ (Compatibility Methods)
    # -------------------------------------------------------------
    def get_train_test_data(self, df, split_ratio=0.8):
        """
        แบ่งข้อมูล Train/Test และทำ Scaling อย่างถูกต้อง (ป้องกัน Data Leakage)
        
        Process:
        1. Split Data (Train/Test)
        2. Fit Scaler on Train ONLY
        3. Transform Train and Test
        """
        data = df.filter(['close']).values
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
        สร้าง Sliding Window Dataset (X, y)
        """
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        
        X, y = np.array(X), np.array(y)
        # Reshape for LSTM/GRU [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y

    # Deprecated: Old method with leakage risk
    def prepare_data(self, df):
        print("⚠️ Warning: prepare_data() has potential data leakage. Use get_train_test_data() instead.")
        data = df.filter(['close']).values
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data, self.scaler

    # Deprecated: Old split method
    def split_train_test(self, X, y, split_ratio=0.8):
        train_size = int(len(X) * split_ratio)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return X_train, y_train, X_test, y_test

def run_preprocessing_pipeline():
    """
    จุดเริ่มต้นหลัก (Main entry point) สำหรับรัน pipeline การเตรียมข้อมูลทั้งหมด
    """
    processor = DataProcessor()
    
    print("="*60)
    print(" STARTING PREPROCESSING PIPELINE")
    print("="*60)
    
    for coin in config.COINS:
        for tf in config.TIMEFRAMES:
            print(f"\ndataset: {coin.upper()} / {tf}")
            processor.process_and_save(coin, tf)
            
    print("\n" + "="*60)
    print(" PREPROCESSING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_preprocessing_pipeline()
