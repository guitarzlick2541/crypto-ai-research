"""
evaluation/evaluate_model.py
============================
สคริปต์ประเมินผลสำหรับงานวิจัย Crypto AI

ดำเนินการดังนี้:
1. โหลดโมเดลที่ฝึกแล้ว (LSTM, GRU) และ Scalers
2. โหลดข้อมูลดิบและเตรียม Test Set (Multi-Feature)
3. สร้างการทำนายบนข้อมูลที่ไม่เคยเห็น (Test Set)
4. คำนวณ Baseline Models (Naive Forecast, Moving Average)
5. บันทึกผลการทดลอง (ค่าจริง vs ค่าทำนาย) ไปยัง experiments/
6. คำนวณตัวชี้วัดทางสถิติ (MAE, RMSE, MAPE, Directional Accuracy)
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
from evaluation.metrics import (
    calculate_mae, calculate_rmse, calculate_mape, 
    calculate_directional_accuracy
)
from utils.scaling import inverse_transform_close

# ---------------------------------------------------------
# ค่าคงที่ (CONSTANTS)
# ---------------------------------------------------------
REPORT_FILE = os.path.join(config.RESULT_DIR, "evaluation_report.csv")


def compute_baseline_predictions(actual_series):
    """
    คำนวณ Baseline Models สำหรับเปรียบเทียบกับ Deep Learning
    
    Baselines:
    1. Naive Forecast: ใช้ราคาล่าสุดเป็นค่าทำนาย (y_pred[t] = y_actual[t-1])
    2. Moving Average (7): ค่าเฉลี่ยเคลื่อนที่ 7 ช่วงเวลา
    
    Args:
        actual_series: np.array ของราคาจริง (inverse-transformed)
        
    Returns:
        dict: {
            "naive": {"predictions": np.array, "start_idx": int},
            "ma7": {"predictions": np.array, "start_idx": int}
        }
    """
    actual = np.array(actual_series).flatten()
    
    # 1. Naive Forecast: y_pred[t] = y_actual[t-1]
    # เริ่มจาก index 1 เพราะ Naive ต้องการค่าก่อนหน้า 1 ตัว
    naive_preds = actual[:-1]  # ค่าจริงของ step ก่อนหน้า
    
    # 2. Moving Average (7): ค่าเฉลี่ย 7 ช่วงเวลาก่อนหน้า
    ma_period = 7
    ma_preds = []
    for i in range(ma_period, len(actual)):
        ma_preds.append(np.mean(actual[i - ma_period:i]))
    ma_preds = np.array(ma_preds)
    
    return {
        "naive": {"predictions": naive_preds, "start_idx": 1},
        "ma7": {"predictions": ma_preds, "start_idx": ma_period}
    }


def evaluate_models():
    """
    กระบวนการประเมินผลหลัก (Main Evaluation Workflow)
    รองรับ Multi-Feature Models + Baseline Comparison + Directional Accuracy
    """
    config._ensure_directories(config.RESULT_DIR)
    
    summary_results = []
    
    models_to_eval = [
        {"type": "LSTM", "prefix": "lstm"},
        {"type": "GRU", "prefix": "gru"}
    ]

    print("=" * 60)
    print(" STARTING RESEARCH EVALUATION (with Baselines)")
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
            
            # ตัวแปรเก็บ actual_series สำหรับ baseline (คำนวณครั้งเดียว)
            actual_series_for_baseline = None
            
            # 2. ประเมินผลแต่ละโมเดล (LSTM, GRU)
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
                    saved_scaler = joblib.load(scaler_path)
                    
                    print(f"    Evaluating {model_type}...")
                    preds_scaled = model.predict(X_test, verbose=0)
                    
                    # Inverse transform (using shared utility)
                    preds_actual = inverse_transform_close(saved_scaler, preds_scaled)
                    actual_series = inverse_transform_close(saved_scaler, y_test)
                    
                    preds_series = preds_actual.flatten()
                    actual_series = actual_series.flatten()
                    
                    # เก็บ actual_series สำหรับ baseline (ครั้งแรกเท่านั้น)
                    if actual_series_for_baseline is None:
                        actual_series_for_baseline = actual_series.copy()
                    
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
                    
                    # 4. คำนวณตัวชี้วัด (using shared metrics + directional accuracy)
                    mae = calculate_mae(actual_series, preds_series)
                    rmse = calculate_rmse(actual_series, preds_series)
                    mape = calculate_mape(actual_series, preds_series)
                    da = calculate_directional_accuracy(actual_series, preds_series)
                    
                    summary_results.append({
                        "model": model_type,
                        "timeframe": tf,
                        "coin": coin.upper(),
                        "mae": round(mae, 4),
                        "rmse": round(rmse, 4),
                        "mape": round(mape, 4),
                        "da": round(da, 2)
                    })
                    
                    print(f"       MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, DA: {da:.2f}%")
                    
                except Exception as e:
                    print(f"      ❌ Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 3. คำนวณ Baseline Models (Naive + MA7)
            if actual_series_for_baseline is not None and len(actual_series_for_baseline) > 7:
                print(f"\n    📐 Computing Baselines for {coin.upper()} ({tf})...")
                baselines = compute_baseline_predictions(actual_series_for_baseline)
                
                for baseline_name, baseline_data in baselines.items():
                    start_idx = baseline_data["start_idx"]
                    b_preds = baseline_data["predictions"]
                    b_actual = actual_series_for_baseline[start_idx:]
                    
                    # ตรวจสอบความยาวตรงกัน
                    min_len = min(len(b_preds), len(b_actual))
                    b_preds = b_preds[:min_len]
                    b_actual = b_actual[:min_len]
                    
                    if min_len == 0:
                        continue
                    
                    mae = calculate_mae(b_actual, b_preds)
                    rmse = calculate_rmse(b_actual, b_preds)
                    mape = calculate_mape(b_actual, b_preds)
                    da = calculate_directional_accuracy(b_actual, b_preds)
                    
                    display_name = "Naive" if baseline_name == "naive" else "MA(7)"
                    
                    summary_results.append({
                        "model": display_name,
                        "timeframe": tf,
                        "coin": coin.upper(),
                        "mae": round(mae, 4),
                        "rmse": round(rmse, 4),
                        "mape": round(mape, 4),
                        "da": round(da, 2)
                    })
                    
                    print(f"       {display_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, DA={da:.2f}%")

    # 5. บันทึกรายงานสรุป
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(REPORT_FILE, index=False)
        
        print("\n" + "=" * 60)
        print(" EVALUATION SUMMARY (Saved to evaluation_report.csv)")
        print("=" * 60)
        print(df_summary.to_string(index=False))
        
        # แยกผลลัพธ์ DL models กับ baselines
        dl_models = df_summary[df_summary['model'].isin(['LSTM', 'GRU'])]
        baselines = df_summary[df_summary['model'].isin(['Naive', 'MA(7)'])]
        
        if not dl_models.empty:
            best_model_idx = dl_models['mape'].idxmin()
            best_model = dl_models.loc[best_model_idx]
            print("\n🏆 BEST PERFORMING MODEL:")
            print(f"   {best_model['model']} on {best_model['coin']} ({best_model['timeframe']})")
            print(f"   MAPE: {best_model['mape']}% | DA: {best_model['da']}%")
        
        if not baselines.empty:
            print("\n📐 BASELINE COMPARISON:")
            for _, row in baselines.iterrows():
                print(f"   {row['model']} on {row['coin']} ({row['timeframe']}): "
                      f"MAPE={row['mape']}%, DA={row['da']}%")
        
        readme_path = os.path.join(config.RESULT_DIR, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# ผลการทดลอง (Experimental Results)\n\n")
            f.write("โฟลเดอร์นี้เก็บข้อมูลผลลัพธ์การทำนายดิบ (Raw Prediction) ของแต่ละโมเดล\n\n")
            f.write("## รูปแบบไฟล์ (File Format)\n")
            f.write("- **ชื่อไฟล์**: `{coin}_{model}_{timeframe}_predictions.csv`\n")
            f.write("- **คอลัมน์**:\n")
            f.write("  - `timestamp`: วันและเวลาของข้อมูล\n")
            f.write("  - `actual`: ราคาปิดจริง\n")
            f.write("  - `predicted`: ราคาที่ทำนายโดยโมเดล AI\n\n")
            f.write("## ตัวชี้วัดที่ใช้ (Evaluation Metrics)\n")
            f.write("- **MAE**: Mean Absolute Error\n")
            f.write("- **RMSE**: Root Mean Squared Error\n")
            f.write("- **MAPE**: Mean Absolute Percentage Error (%)\n")
            f.write("- **DA**: Directional Accuracy (%) — ความแม่นยำในการทำนายทิศทางราคา\n\n")
            f.write("## Baseline Models\n")
            f.write("- **Naive Forecast**: ใช้ราคาปิดของแท่งเทียนก่อนหน้าเป็นค่าทำนาย\n")
            f.write("- **MA(7)**: ค่าเฉลี่ยเคลื่อนที่ 7 ช่วงเวลาก่อนหน้า\n")
            
    else:
        print("\n❌ No models were evaluated successfully.")

if __name__ == "__main__":
    evaluate_models()

