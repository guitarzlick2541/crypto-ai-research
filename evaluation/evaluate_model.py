"""
evaluation/evaluate_model.py
============================
สคริปต์ประเมินผลสำหรับงานวิจัย Crypto AI

ดำเนินการดังนี้:
1. โหลดโมเดลที่ฝึกแล้ว (LSTM, GRU) และ Scalers
2. โหลดข้อมูลดิบและเตรียม Test Set (Multi-Feature)
3. สร้างการทำนายบนข้อมูลที่ไม่เคยเห็น (Test Set)
4. บันทึกผลการทดลอง (ค่าจริง vs ค่าทำนาย) ไปยัง experiments/
5. คำนวณตัวชี้วัดทางสถิติ (MAE, RMSE, MAPE)
"""

import os
import sys
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# เพิ่ม project root เข้าไปใน path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import config
from train.preprocessing import DataProcessor
from evaluation.metrics import calculate_mae, calculate_rmse, calculate_mape
from utils.scaling import inverse_transform_close

# ---------------------------------------------------------
# ค่าคงที่ (CONSTANTS)
# ---------------------------------------------------------
REPORT_FILE = os.path.join(config.RESULT_DIR, "evaluation_report.csv")


def evaluate_models():
    """
    กระบวนการประเมินผลหลัก (Main Evaluation Workflow)
    รองรับ Multi-Feature Models
    """
    config._ensure_directories(config.RESULT_DIR)
    
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
            
            # 1. โหลดข้อมูล (Load Data) - Multi-Feature
            processor = DataProcessor()
            try:
                df_raw = processor.load_data(coin, tf)
                timestamps = df_raw['timestamp'].values
                
                # เตรียมข้อมูล (Scaling) - Multi-Feature
                train_scaled, test_scaled, scaler = processor.get_train_test_data(df_raw)
                
                # สร้าง Sliding Window สำหรับ Test Set
                X_test, y_test = processor.create_sliding_window(test_scaled)
                
                # คำนวณ Timestamps สำหรับ Test Set
                train_size = len(train_scaled)
                test_start_index = train_size + config.WINDOW_SIZE
                test_timestamps = timestamps[test_start_index:]
                
                if len(test_timestamps) != len(y_test):
                    test_timestamps = test_timestamps[:len(y_test)]

            except Exception as e:
                print(f"   ❌ Error loading/processing data: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 2. ประเมินผลแต่ละโมเดล
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
                    model = load_model(model_path, compile=False)
                    scaler = joblib.load(scaler_path)
                    
                    print(f"    Evaluating {model_type}...")
                    preds_scaled = model.predict(X_test, verbose=0)
                    
                    # Inverse transform (using shared utility)
                    preds_actual = inverse_transform_close(scaler, preds_scaled)
                    actual_series = inverse_transform_close(scaler, y_test)
                    
                    preds_series = preds_actual.flatten()
                    actual_series = actual_series.flatten()
                    
                    # 3. บันทึกผลการทดลอง
                    predictions_dir = os.path.join(config.RESULT_DIR, "predictions")
                    os.makedirs(predictions_dir, exist_ok=True)
                    
                    experiment_filename = f"{coin.lower()}_{model_type.lower()}_{tf}_predictions.csv"
                    experiment_path = os.path.join(predictions_dir, experiment_filename)
                    
                    df_experiment = pd.DataFrame({
                        "timestamp": test_timestamps,
                        "actual": actual_series,
                        "predicted": preds_series
                    })
                    
                    df_experiment.to_csv(experiment_path, index=False)
                    print(f"       Saved predictions: {experiment_path}")
                    
                    # 4. คำนวณตัวชี้วัด (using shared metrics)
                    mae = calculate_mae(actual_series, preds_series)
                    rmse = calculate_rmse(actual_series, preds_series)
                    mape = calculate_mape(actual_series, preds_series)
                    
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

    # 5. บันทึกรายงานสรุป
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(REPORT_FILE, index=False)
        
        print("\n" + "=" * 60)
        print(" EVALUATION SUMMARY (Saved to evaluation_report.csv)")
        print("=" * 60)
        print(df_summary)
        
        best_model_idx = df_summary['mape'].idxmin()
        best_model = df_summary.loc[best_model_idx]
        print("\n BEST PERFORMING MODEL:")
        print(f"   {best_model['model']} on {best_model['coin']} ({best_model['timeframe']})")
        print(f"   MAPE: {best_model['mape']}%")
        
        readme_path = os.path.join(config.RESULT_DIR, "README.md")
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
