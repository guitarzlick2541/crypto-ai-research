"""
evaluation/evaluate_model.py
============================
สคริปต์ประเมินผลสำหรับงานวิจัย Crypto AI (ระดับปริญญาตรี)

สคริปต์นี้จัดการการประเมินผลโมเดล AI อย่างเป็นระบบ
ดำเนินการดังนี้:
1. โหลดโมเดลที่ฝึกแล้ว (LSTM, GRU) และ Scalers
2. โหลดข้อมูลดิบและเตรียม Test Set
3. สร้างการทำนายบนข้อมูลที่ไม่เคยเห็น (Test Set)
4. บันทึกผลการทดลองทางตรรกะ (ค่าจริง vs ค่าทำนาย) ไปยัง `experiments/`
5. คำนวณตัวชี้วัดทางสถิติ (MAE, RMSE, MAPE) และบันทึกลงไฟล์ `evaluation_report.csv`

ผู้เขียน: AI Engineer & Researcher
"""

import os
import sys
import pandas as pd
import joblib
import numpy as np
import math
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# เพิ่ม project root เข้าไปใน path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import config
from train.preprocessing import DataProcessor

# ---------------------------------------------------------
# ค่าคงที่และการตั้งค่า (CONSTANTS & CONFIG)
# ---------------------------------------------------------
EXPERIMENT_DIR = config.RESULT_DIR
RESULT_DIR = config.RESULT_DIR
REPORT_FILE = os.path.join(EXPERIMENT_DIR, "evaluation_report.csv")

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def ensure_directories():
    """ตรวจสอบว่ามีโฟลเดอร์ที่จำเป็นอยู่หรือไม่"""
    for directory in [EXPERIMENT_DIR, RESULT_DIR]:
        if not os.path.exists(directory):
            print(f" Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

def evaluate_models():
    """
    กระบวนการประเมินผลหลัก (Main Evaluation Workflow)
    วนลูปผ่านทุกเหรียญ (Coins), ช่วงเวลา (Timeframes), และโมเดล (Models)
    """
    ensure_directories()
    
    # ลิสต์สำหรับเก็บผลสรุปตัวชี้วัด
    summary_results = []
    
    models_to_eval = [
        {"type": "LSTM", "prefix": "lstm"},
        {"type": "GRU", "prefix": "gru"}
    ]

    print("=" * 60)
    print(" STARTING RESEARCH EVALUATION")
    print("=" * 60)

    for coin in config.COINS:
        for tf in config.TIMEFRAMES:
            print(f"\nProcessing Dataset: {coin.upper()} ({tf})")
            
            # 1. โหลดข้อมูล (Load Data)
            processor = DataProcessor()
            try:
                # โหลด dataframe ดิบเพื่อดึง timestamps
                df_raw = processor.load_data(coin, tf)
                timestamps = df_raw['timestamp'].values
                
                # เตรียมข้อมูล (Scaling) โดยแบ่ง Train/Test ก่อน Fit (FIXED: Prevent Leakage)
                # ใช้ get_train_test_data แทน prepare_data
                train_scaled, test_scaled, scaler = processor.get_train_test_data(df_raw)
                
                # สร้าง Sliding Window สำหรับ Test Set
                # หมายเหตุ: เราสนใจเฉพาะ Test Set ในการ evaluate
                X_test, y_test = processor.create_sliding_window(test_scaled)
                
                # คำนวณ Timestamps สำหรับ Test Set
                # train_scaled size = train_size
                # test_scaled size = len(df) - train_size
                # Sliding window เริ่มที่ index = window_size ใน test_scaled
                # ดังนั้น global index = train_size + window_size
                train_size = len(train_scaled)
                window_size = config.WINDOW_SIZE
                
                test_start_index = train_size + window_size
                test_timestamps = timestamps[test_start_index:]
                
                # ตรวจสอบขนาด (Alignment Check)
                if len(test_timestamps) != len(y_test):
                    diff = len(test_timestamps) - len(y_test)
                    # print(f"   ⚠️ Adjusting timestamps: {len(test_timestamps)} -> {len(y_test)}")
                    test_timestamps = test_timestamps[:len(y_test)]

            except Exception as e:
                print(f"   ❌ Error loading/processing data: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 2. ประเมินผลแต่ละโมเดล (Evaluate each model)
            for m in models_to_eval:
                model_type = m["type"]
                prefix = m["prefix"]
                
                model_filename = f"{prefix}_{coin.lower()}_{tf}.h5"
                scaler_filename = f"scaler_{coin.lower()}_{tf}.save"
                
                model_path = os.path.join(config.MODEL_DIR, model_filename)
                scaler_path = os.path.join(config.MODEL_DIR, scaler_filename)
                
                if not os.path.exists(model_path):
                    print(f"   ⚠️  Model not found: {model_filename}")
                    continue
                    
                try:
                    # โหลดโมเดล และ Scaler (Load model & Scaler)
                    model = load_model(model_path, compile=False)
                    scaler = joblib.load(scaler_path)
                    
                    # ทำนาย (Predict)
                    print(f"    Evaluating {model_type}...")
                    preds_scaled = model.predict(X_test, verbose=0)
                    
                    # แปลงค่ากลับ (Inverse Transform)
                    preds_actual = scaler.inverse_transform(preds_scaled)
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Flatten เพื่อลง DataFrame
                    preds_series = preds_actual.flatten()
                    actual_series = y_test_actual.flatten()
                    
                    # 3. บันทึกผลการทดลอง (ไฟล์ Predictions)
                    predictions_dir = os.path.join(EXPERIMENT_DIR, "predictions")
                    os.makedirs(predictions_dir, exist_ok=True)
                    
                    experiment_filename = f"{coin.lower()}_{model_type.lower()}_{tf}_predictions.csv"
                    experiment_path = os.path.join(predictions_dir, experiment_filename)
                    
                    df_experiment = pd.DataFrame({
                        "timestamp": test_timestamps,
                        "actual": actual_series,
                        "predicted": preds_series
                    })
                    
                    df_experiment.to_csv(experiment_path, index=False)
                    print(f"       Saved usage log: {experiment_path}")
                    
                    # 4. คำนวณตัวชี้วัด (Calculate Metrics)
                    mae = mean_absolute_error(actual_series, preds_series)
                    rmse = np.sqrt(mean_squared_error(actual_series, preds_series))
                    mape = calculate_mape(actual_series, preds_series)
                    
                    # เพิ่มลงในผลสรุป (Add to summary)
                    summary_results.append({
                        "model": model_type,
                        "timeframe": tf,
                        "coin": coin.upper(),
                        "mae": round(mae, 4),
                        "rmse": round(rmse, 4),
                        "mape": round(mape, 4)
                    })
                    
                except Exception as e:
                    print(f"      ❌ Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()

    # 5. บันทึกรายงานสรุป (Save Summary Report)
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        
        # บันทึกรายงานหลัก
        df_summary.to_csv(REPORT_FILE, index=False)
        
        print("\n" + "=" * 60)
        print(" EVALUATION SUMMARY (Saved to evaluation_report.csv)")
        print("=" * 60)
        print(df_summary)
        
        # หาโมเดลที่ดีที่สุด (Determine best model)
        best_model_idx = df_summary['mape'].idxmin()
        best_model = df_summary.loc[best_model_idx]
        print("\n BEST PERFORMING MODEL:")
        print(f"   {best_model['model']} on {best_model['coin']} ({best_model['timeframe']})")
        print(f"   MAPE: {best_model['mape']}%")
        
        # สร้าง README สำหรับ experiments
        readme_path = os.path.join(EXPERIMENT_DIR, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# ผลการทดลอง (Experimental Results)\n\n")
            f.write("โฟลเดอร์นี้เก็บข้อมูลผลลัพธ์การทำนายดิบ (Raw Prediction) ของแต่ละโมเดล\n\n")
            f.write("## รูปแบบไฟล์ (File Format)\n")
            f.write("- **ชื่อไฟล์**: `{coin}_{model}_{timeframe}_predictions.csv`\n")
            f.write("- **คอลัมน์**:\n")
            f.write("  - `timestamp`: วันและเวลาของข้อมูล\n")
            f.write("  - `actual`: ราคาปิดจริง\n")
            f.write("  - `predicted`: ราคาที่ทำนายโดยโมเดล AI\n")
            
    else:
        print("\n❌ No models were evaluated successfully.")

if __name__ == "__main__":
    evaluate_models()
