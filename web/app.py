"""
Flask Web Application - Real-time Crypto AI Dashboard
======================================================
- โหลดผลวิจัย MAE/RMSE จาก dual_model_results.csv
- API endpoints สำหรับ real-time prediction
- Dashboard แสดงผลแบบ real-time

CHANGELOG:
- v2.0: Disabled debug mode by default (SECURITY FIX)
- v2.0: Added environment variable for debug control
- v2.0: Added production mode configuration
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import sys

# เพิ่ม path ของ src และ inference เพื่อให้ python หา module เจอ
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_DIR = os.path.join(BASE_DIR, "inference")

sys.path.append(BASE_DIR)
sys.path.append(INFERENCE_DIR)

# ใช้ predictor_fixed แทน predictor (FIXED VERSION)
# Removed fallback to buggy predictor.py as it is being deleted
from inference.predictor_fixed import predictor as prediction_logic
print("[INFO] Using FIXED predictor (predictor_fixed.py)")

from inference import load_model as model_loader

# Path Configuration
# -------------------------
RESULT_DIR = os.path.join(BASE_DIR, "experiments")

# การตั้งค่า Flask App
# -------------------------
app = Flask(__name__)
CORS(app)  # เปิดใช้งาน CORS สำหรับ API endpoints

# Helper Functions
# -------------------------
def load_research_results():
    """โหลดผลวิจัยจาก dual_model_results.csv"""
    csv_path = os.path.join(RESULT_DIR, "dual_model_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def get_model_metrics(df, coin, timeframe, model_type):
    """ดึงค่า MAE และ RMSE สำหรับ model ที่ระบุ"""
    if df is None:
        return {"mae": "N/A", "rmse": "N/A"}
    
    row = df[(df["Coin"] == coin.upper()) & 
             (df["Timeframe"] == timeframe) & 
             (df["Model"] == model_type)]
    if not row.empty:
        return {
            "mae": round(row["MAE"].values[0], 2),
            "rmse": round(row["RMSE"].values[0], 2)
        }
    return {"mae": "N/A", "rmse": "N/A"}


# Web Routes
# -------------------------
@app.route("/")
def index():
    """หน้า Dashboard หลัก"""
    # รับค่าจาก URL parameters
    coin = request.args.get("coin", "btc").lower()
    tf = request.args.get("tf", "1h")
    
    # Validate inputs (FIXED: เพิ่ม validation)
    valid_coins = ["btc", "eth"]
    valid_timeframes = ["1h", "4h"]
    
    if coin not in valid_coins:
        coin = "btc"
    if tf not in valid_timeframes:
        tf = "1h"
    
    # โหลดผลวิจัยจาก dual_model_results.csv
    research_df = load_research_results()
    
    # ดึงค่า metrics สำหรับ LSTM และ GRU
    lstm_metrics = get_model_metrics(research_df, coin, tf, "LSTM")
    gru_metrics = get_model_metrics(research_df, coin, tf, "GRU")
    
    # รายการเหรียญและ timeframes ที่รองรับ
    available_coins = ["btc", "eth"]
    available_timeframes = ["1h", "4h"]
    
    return render_template(
        "index.html",
        coin=coin.upper(),
        tf=tf,
        available_coins=available_coins,
        available_timeframes=available_timeframes,
        lstm_mae=lstm_metrics["mae"],
        lstm_rmse=lstm_metrics["rmse"],
        gru_mae=gru_metrics["mae"],
        gru_rmse=gru_metrics["rmse"]
    )


@app.route("/experiments")
def experiments_dashboard():
    """หน้า Experiment Dashboard"""
    return render_template("experiments.html")


@app.route("/health")
def web_health():
    """หน้า Health Check แบบ Web"""
    health = prediction_logic.health_check()
    return render_template("health.html", health=health)


# API Routes - Real-time
# -------------------------
@app.route("/api/predict")
def api_predict():
    """
    API Endpoint สำหรับ real-time prediction
    
    Query Parameters:
        - coin: btc หรือ eth (default: btc)
        - tf: 1h หรือ 4h (default: 1h)
        - model: lstm หรือ gru หรือ both (default: both)
    
    Returns:
        JSON response with prediction results
    """
    coin = request.args.get("coin", "btc").lower()
    tf = request.args.get("tf", "1h")
    model = request.args.get("model", "both").lower()
    
    try:
        if model == "both":
            result = prediction_logic.predict_both_models(coin, tf)
        elif model in ["lstm", "gru"]:
            result = prediction_logic.predict(coin, tf, model.upper())
        else:
            return jsonify({
                "success": False,
                "error": f"Unknown model: {model}. Must be 'lstm', 'gru', or 'both'"
            }), 400
        
        return jsonify({
            "success": True,
            "data": result,
            "version": "2.0-fixed"
        })
        
    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "hint": "Please train the model first using: python train/run_training.py"
        }), 404
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        import traceback
        print(f"❌ Error in /api/predict: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/price")
def api_price():
    """
    API Endpoint สำหรับดึงราคาปัจจุบัน
    
    Query Parameters:
        - coin: btc หรือ eth (default: btc)
        - tf: 1h หรือ 4h (default: 1h)
    
    Returns:
        JSON response with current price and OHLCV data
    """
    coin = request.args.get("coin", "btc").lower()
    tf = request.args.get("tf", "1h")
    limit = request.args.get("limit", 100, type=int)
    
    # Validate inputs
    if coin not in ["btc", "eth"]:
        return jsonify({
            "success": False,
            "error": f"Invalid coin: {coin}. Must be 'btc' or 'eth'"
        }), 400
    
    if tf not in ["1h", "4h"]:
        return jsonify({
            "success": False,
            "error": f"Invalid timeframe: {tf}. Must be '1h' or '4h'"
        }), 400
    
    try:
        data = prediction_logic.fetch_live_data(coin, tf, limit)
        
        return jsonify({
            "success": True,
            "data": {
                "coin": coin.upper(),
                "timeframe": tf,
                "current_price": data["current_price"],
                "timestamp": data["timestamp"],
                "ohlcv": data["ohlcv"][-50:]  # ส่งกลับ 50 candles ล่าสุด
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/models")
def api_models():
    """
    API Endpoint สำหรับดูรายการโมเดลที่มี
    
    Returns:
        JSON response with available models and their status
    """
    try:
        models = prediction_logic.get_available_models()
        
        return jsonify({
            "success": True,
            "data": {
                "models": models,
                "total": len(models),
                "ready": len([m for m in models if m.get("has_scaler", False)]),
                "version": "2.0-fixed"
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/research")
def api_research():
    """
    API Endpoint สำหรับดูผลวิจัย MAE/RMSE
    
    Returns:
        JSON response with research results
    """
    try:
        df = load_research_results()
        
        if df is None:
            return jsonify({
                "success": False,
                "error": "Research results not found. Run evaluation first."
            }), 404
        
        results = df.to_dict(orient="records")
        
        return jsonify({
            "success": True,
            "data": {
                "results": results,
                "total": len(results)
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/health")
def api_health():
    """
    Health check endpoint with detailed info
    
    Returns:
        JSON response with system health status
    """
    try:
        predictor_health = prediction_logic.health_check()
        
        return jsonify({
            "status": "healthy" if predictor_health["status"] == "healthy" else "degraded",
            "service": "crypto-ai-prediction",
            "version": "2.0.0-fixed",
            "predictor": predictor_health,
            "environment": os.getenv("FLASK_ENV", "production")
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# Experiment Dashboard APIs
# -------------------------
@app.route("/api/experiments/accuracy")
def api_experiments_accuracy():
    """
    API Endpoint สำหรับดึงข้อมูล Accuracy (MAE/RMSE) ของ Models
    
    Returns:
        JSON response with accuracy data
    """
    try:
        csv_path = os.path.join(RESULT_DIR, "dual_model_results.csv")
        
        if not os.path.exists(csv_path):
            return jsonify({
                "success": False,
                "error": "Accuracy results not found. Run evaluation first."
            }), 404
        
        df = pd.read_csv(csv_path)
        
        results = []
        for _, row in df.iterrows():
            results.append({
                "coin": row["Coin"],
                "timeframe": row["Timeframe"],
                "model": row["Model"],
                "mae": row["MAE"],
                "rmse": row["RMSE"]
            })
        
        return jsonify({
            "success": True,
            "data": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/experiments/log")
def api_experiments_log():
    """
    API Endpoint สำหรับดึงข้อมูล Experiment Log
    
    Returns:
        JSON response with experiment log and stats
    """
    try:
        csv_path = os.path.join(RESULT_DIR, "experiment_log.csv")
        
        if not os.path.exists(csv_path):
            return jsonify({
                "success": True,
                "data": {
                    "experiments": [],
                    "stats": {
                        "total_experiments": 0,
                        "avg_accuracy": None,
                        "best_model": None
                    },
                    "latency": {
                        "min": None,
                        "avg": None,
                        "max": None
                    }
                }
            })
        
        df = pd.read_csv(csv_path)
        
        # Prepare experiments list
        experiments = []
        for _, row in df.iterrows():
            experiments.append({
                "timestamp": row["timestamp"],
                "experiment_id": row["experiment_id"],
                "model_type": row["model_type"],
                "coin": row["coin"],
                "timeframe": row["timeframe"],
                "accuracy": row["accuracy"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "latency_ms": row["latency_ms"],
                "action": row["action"],
                "status": row["status"],
                "notes": row.get("notes", "")
            })
        
        # Calculate stats
        total_experiments = len(df)
        avg_accuracy = df["accuracy"].mean() if not df.empty else None
        
        # Find best model
        best_model = None
        if not df.empty:
            best_row = df.loc[df["accuracy"].idxmax()]
            best_model = f"{best_row['model_type'].upper()} ({best_row['coin'].upper()}/{best_row['timeframe']})"
        
        # Calculate latency stats
        latency_stats = {
            "min": df["latency_ms"].min() if not df.empty else None,
            "avg": df["latency_ms"].mean() if not df.empty else None,
            "max": df["latency_ms"].max() if not df.empty else None
        }
        
        return jsonify({
            "success": True,
            "data": {
                "experiments": experiments,
                "stats": {
                    "total_experiments": total_experiments,
                    "avg_accuracy": avg_accuracy,
                    "best_model": best_model
                },
                "latency": latency_stats
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# รัน Server
# -------------------------
if __name__ == "__main__":
    # FIXED: ใช้ environment variable ควบคุม debug mode
    is_debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    is_production = os.getenv("FLASK_ENV", "production").lower() == "production"
    
    print("\n" + "="*60)
    print(" Crypto AI Real-time Prediction Server v2.0 (FIXED)")
    print("="*60)
    print(f" Environment: {'PRODUCTION' if is_production else 'DEVELOPMENT'}")
    print(f" Debug Mode: {'ON' if is_debug else 'OFF'}")
    print("="*60)
    print("\n Web Pages:")
    print("   - GET /                  → Dashboard")
    print("   - GET /experiments       → Experiment Dashboard")
    print("   - GET /health            → Health Check Page")
    print("\n API Endpoints:")
    print("   - GET /api/predict       → Real-time Prediction")
    print("   - GET /api/price         → Current Price + OHLCV")
    print("   - GET /api/models        → Available Models")
    print("   - GET /api/research      → Research Results")
    print("   - GET /api/health        → Health Check")
    print("\n" + "="*60 + "\n")
    
    # FIXED: ไม่เปิด debug mode โดย default
    app.run(
        debug=is_debug,
        host="0.0.0.0",
        port=5000,
        threaded=True  # รองรับหลาย request พร้อมกัน
    )
