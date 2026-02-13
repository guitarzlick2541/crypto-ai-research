"""
Real-time Prediction Engine - FIXED VERSION
=============================================
โหลดโมเดล LSTM/GRU ที่ train ไว้แล้ว และทำ prediction แบบ real-time
โดยดึงข้อมูลจาก Binance API

CHANGELOG:
- v3.0: Refactored to use shared utils (indicators, scaling, config)
- v3.0: Fixed dead code conditions (BUG-02)
- v3.0: Fixed scaler_used metadata (BUG-03)
- v3.0: Fixed wrong exception type (BUG-04)
- v3.0: Optimized predict_both_models to avoid redundant API calls (CC-06)
- v2.0: Fixed scaler mismatch bug (CRITICAL)
- v2.0: Added retry logic for Binance API
- v2.0: Added input validation
- v2.0: Improved error handling
"""

import os
import numpy as np
import pandas as pd
import requests
import joblib
import sys
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# -------------------------
# Import from shared modules
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from train import config
from utils.indicators import compute_technical_indicators
from utils.scaling import inverse_transform_close


class RealTimePredictor:
    """Engine สำหรับทำ real-time prediction - FIXED VERSION"""
    
    def __init__(self):
        self.models = {}  # Cache สำหรับเก็บโมเดลที่โหลดแล้ว
        self.scalers = {}  # Cache สำหรับเก็บ scaler
        
        # สร้าง session พร้อม retry strategy (ใช้ซ้ำได้)
        self.session = self._create_session()
    
    def _create_session(self, max_retries: int = 3) -> requests.Session:
        """
        สร้าง requests session พร้อม retry strategy
        
        Args:
            max_retries: จำนวนครั้งที่ลองใหม่ถ้า fail
            
        Returns:
            requests.Session พร้อม retry mechanism
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # รอ 1, 2, 4 วินาที
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session
    
    def _validate_inputs(self, coin: str, timeframe: str, model_type: str = None) -> tuple:
        """
        ตรวจ input validity
        
        Returns:
            tuple: (coin_lower, timeframe_lower, model_type_upper)
            
        Raises:
            ValueError: ถ้า input ไม่ถูกต้อง
        """
        coin_lower = coin.lower()
        timeframe_lower = timeframe.lower()
        
        if coin_lower not in config.COINS:
            raise ValueError(
                f"Invalid coin: '{coin}'. "
                f"Must be one of: {config.COINS}"
            )
        
        if timeframe_lower not in config.TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe: '{timeframe}'. "
                f"Must be one of: {config.TIMEFRAMES}"
            )
        
        if model_type is not None:
            model_upper = model_type.upper()
            if model_upper not in config.MODEL_TYPES:
                raise ValueError(
                    f"Invalid model_type: '{model_type}'. "
                    f"Must be one of: {config.MODEL_TYPES}"
                )
            return coin_lower, timeframe_lower, model_upper
        
        return coin_lower, timeframe_lower, None
    
    def _get_model_path(self, model_type: str, coin: str, timeframe: str) -> str:
        """สร้าง path ของไฟล์โมเดล"""
        filename = f"{model_type.lower()}_{coin.lower()}_{timeframe}.h5"
        return os.path.join(config.MODEL_DIR, filename)
    
    def _get_scaler_path(self, coin: str, timeframe: str) -> str:
        """สร้าง path ของไฟล์ scaler"""
        filename = f"scaler_{coin.lower()}_{timeframe}.save"
        return os.path.join(config.MODEL_DIR, filename)
    
    def _load_model(self, model_type: str, coin: str, timeframe: str):
        """โหลดโมเดลจากไฟล์ .h5 (ใช้ cache)"""
        cache_key = f"{model_type}_{coin}_{timeframe}"
        
        if cache_key not in self.models:
            model_path = self._get_model_path(model_type, coin, timeframe)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Please train the model first using: python train/run_training.py"
                )
            
            print(f"[INFO] Loading model: {model_path}")
            self.models[cache_key] = load_model(model_path, compile=False)
        
        return self.models[cache_key]
    
    def _load_scaler(self, coin: str, timeframe: str) -> MinMaxScaler:
        """
        โหลด scaler ที่ train ไว้ (CRITICAL FIX)
        """
        cache_key = f"scaler_{coin}_{timeframe}"
        
        if cache_key not in self.scalers:
            scaler_path = self._get_scaler_path(coin, timeframe)
            
            if os.path.exists(scaler_path):
                print(f"[INFO] Loading scaler: {scaler_path}")
                self.scalers[cache_key] = joblib.load(scaler_path)
            else:
                print(f"⚠️ WARNING: Scaler not found at {scaler_path}")
                print("   Using fallback scaler (predictions may be less accurate)")
                self.scalers[cache_key] = None
        
        return self.scalers[cache_key]
    
    def fetch_live_data(self, coin: str, timeframe: str, limit: int = None) -> dict:
        """
        ดึงข้อมูล real-time จาก Binance API พร้อม retry mechanism (FIXED)
        
        Args:
            coin: 'btc' หรือ 'eth'
            timeframe: '1h' หรือ '4h'
            limit: จำนวน candles (ถ้าไม่ระบุจะใช้ค่าสำหรับ prediction)
            
        Returns:
            dict: {'prices', 'current_price', 'timestamp', 'ohlcv'}
        """
        coin, timeframe, _ = self._validate_inputs(coin, timeframe)
        
        # เก็บไว้ว่า caller ระบุ limit เองหรือไม่ (สำหรับ validation ด้านล่าง)
        caller_specified_limit = limit is not None
        
        if limit is None:
            limit = config.WINDOW_SIZE + config.INDICATOR_WARMUP
            
        symbol = config.SYMBOL_MAP.get(coin)
        if not symbol:
            raise ValueError(f"Unknown coin: {coin}")
        
        interval_map = {"1h": "1h", "4h": "4h", "1d": "1d"}
        interval = interval_map.get(timeframe, timeframe)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        data = None
        try:
            response = self.session.get(
                config.BINANCE_API_URL, 
                params=params, 
                timeout=config.API_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            print(f"[SUCCESS] Fetched data from Global Binance API")
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Warning: Global Binance API failed ({str(e)}). Trying Binance US...")
            try:
                response = self.session.get(
                    config.BINANCE_US_API_URL,
                    params=params,
                    timeout=config.API_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()
                print(f"[SUCCESS] Fetched data from Binance US API")
                
            except requests.exceptions.RequestException as us_e:
                raise Exception(
                    f"Failed to fetch data from both Global and US Binance APIs.\n"
                    f"Global Error: {str(e)}\n"
                    f"US Error: {str(us_e)}"
                )

        # [BUG-02 FIX] Only enforce minimum for prediction mode (no explicit limit)
        if not caller_specified_limit and len(data) < config.WINDOW_SIZE:
            raise ValueError(
                f"Insufficient data from API: got {len(data)} candles, "
                f"need at least {config.WINDOW_SIZE}"
            )

        # [BUG-04 FIX] Correct exception types for data parsing
        try:
            prices = [float(candle[4]) for candle in data]
            
            ohlcv = []
            for candle in data:
                ohlcv.append({
                    "timestamp": datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })
            
            return {
                "prices": prices,
                "current_price": prices[-1],
                "timestamp": datetime.now().isoformat(),
                "ohlcv": ohlcv
            }
            
        except (IndexError, KeyError, TypeError, ValueError) as e:
            raise Exception(
                f"Failed to parse data from Binance API: {str(e)}"
            )
    
    def preprocess_data(self, ohlcv_list: list, coin: str, timeframe: str) -> tuple:
        """
        Preprocessing pipeline - Multi-Feature VERSION
        Uses shared indicator computation for consistency with training.
        
        Returns:
            tuple: (window, scaler)
        """
        if len(ohlcv_list) < config.WINDOW_SIZE:
            raise ValueError(
                f"Not enough data from Binance. "
                f"Need {config.WINDOW_SIZE} candles, but got {len(ohlcv_list)}"
            )

        # คำนวณ Technical Indicators (using SHARED function — DUP-01 FIX)
        df = pd.DataFrame(ohlcv_list)
        df = compute_technical_indicators(df)
        
        # เลือก features ที่ตรงกับ training (from shared config — DUP-04 FIX)
        feature_cols = [col for col in config.FEATURE_COLUMNS if col in df.columns]
        data = df[feature_cols].values
        
        # ใช้เฉพาะ WINDOW_SIZE อันล่าสุด
        recent_data = data[-config.WINDOW_SIZE:]
        
        # โหลด scaler ที่ train ไว้
        scaler = self._load_scaler(coin, timeframe)
        
        if scaler is not None:
            scaled_data = scaler.transform(recent_data)
        else:
            print("⚠️ WARNING: Using fallback scaler fitting on recent data")
            scaler = MinMaxScaler()
            scaler.fit(data)
            scaled_data = scaler.transform(recent_data)
        
        # Reshape: (1, WINDOW_SIZE, num_features)
        window = scaled_data.reshape(1, config.WINDOW_SIZE, len(feature_cols))
        
        return window, scaler
    
    def predict(self, coin: str, timeframe: str, model_type: str = "LSTM",
                _prefetched_data: dict = None) -> dict:
        """
        ทำ real-time prediction - FIXED VERSION
        
        Args:
            coin, timeframe, model_type: standard parameters
            _prefetched_data: (internal) pre-fetched live data to avoid redundant API calls
            
        Returns:
            dict: ผลลัพธ์ prediction พร้อม metadata
        """
        coin, timeframe, model_type = self._validate_inputs(coin, timeframe, model_type)
        
        try:
            # 1. ดึงข้อมูล (reuse if prefetched — CC-06 FIX)
            if _prefetched_data is not None:
                live_data = _prefetched_data
            else:
                print(f"[INFO] Fetching live data for {coin.upper()} {timeframe}...")
                live_data = self.fetch_live_data(coin, timeframe)
            
            ohlcv = live_data["ohlcv"]
            current_price = live_data["current_price"]
            
            # 2. Preprocessing (Multi-Feature + saved scaler)
            print(f"[INFO] Preprocessing data with {len(config.FEATURE_COLUMNS)} features...")
            window, scaler = self.preprocess_data(ohlcv, coin, timeframe)
            
            # 3. โหลดโมเดล
            print(f"[INFO] Loading {model_type} model...")
            model = self._load_model(model_type, coin, timeframe)
            
            # 4. Predict
            print(f"[INFO] Running prediction...")
            scaled_prediction = model.predict(window, verbose=0)
            
            # 5. Inverse transform (using shared utility — DUP-03/BUG-01 FIX)
            predicted_price_raw = inverse_transform_close(scaler, scaled_prediction)
            
            predicted_price = float(predicted_price_raw.flatten()[0])
            current_price = float(current_price)
            
            # 6. Change percentage
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # 7. Trend (using config threshold — CC-01 FIX)
            if price_change_pct > config.TREND_THRESHOLD:
                trend = "bullish"
            elif price_change_pct < -config.TREND_THRESHOLD:
                trend = "bearish"
            else:
                trend = "neutral"
            
            print(f"[SUCCESS] Prediction completed")
            
            # [BUG-03 FIX] Check file existence, not just path string
            scaler_path = self._get_scaler_path(coin, timeframe)
            
            return {
                "coin": coin.upper(),
                "timeframe": timeframe,
                "model_type": model_type,
                "current_price": round(current_price, 2),
                "predicted_price": round(predicted_price, 2),
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change_pct, 4),
                "trend": trend,
                "timestamp": live_data["timestamp"],
                "data_points_used": config.WINDOW_SIZE,
                "scaler_used": "trained" if os.path.exists(scaler_path) else "fallback"
            }
            
        except FileNotFoundError as e:
            print(f"❌ Model/Scaler not found: {str(e)}")
            raise
        except ValueError as e:
            print(f"❌ Invalid input: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ Error in predict({model_type}, {coin}, {timeframe}): {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_both_models(self, coin: str, timeframe: str) -> dict:
        """
        ทำ prediction ทั้ง LSTM และ GRU
        [CC-06 FIX] Fetches live data ONCE and shares between both models
        """
        # Fetch data once (avoid redundant API calls)
        coin_lower, tf_lower, _ = self._validate_inputs(coin, timeframe)
        print(f"[INFO] Fetching live data for {coin_lower.upper()} {tf_lower}...")
        live_data = self.fetch_live_data(coin_lower, tf_lower)
        
        lstm_result = self.predict(coin_lower, tf_lower, "LSTM", _prefetched_data=live_data)
        gru_result = self.predict(coin_lower, tf_lower, "GRU", _prefetched_data=live_data)
        
        # หาว่า model ไหนทำนายสูงกว่า
        if lstm_result["predicted_price"] > gru_result["predicted_price"]:
            consensus = "LSTM predicts higher"
        elif lstm_result["predicted_price"] < gru_result["predicted_price"]:
            consensus = "GRU predicts higher"
        else:
            consensus = "Both models agree"
        
        avg_predicted = (lstm_result["predicted_price"] + gru_result["predicted_price"]) / 2
        avg_change_pct = (lstm_result["price_change_pct"] + gru_result["price_change_pct"]) / 2
        
        return {
            "coin": coin_lower.upper(),
            "timeframe": tf_lower,
            "current_price": lstm_result["current_price"],
            "timestamp": lstm_result["timestamp"],
            "lstm": {
                "predicted_price": lstm_result["predicted_price"],
                "price_change": lstm_result["price_change"],
                "price_change_pct": lstm_result["price_change_pct"],
                "trend": lstm_result["trend"]
            },
            "gru": {
                "predicted_price": gru_result["predicted_price"],
                "price_change": gru_result["price_change"],
                "price_change_pct": gru_result["price_change_pct"],
                "trend": gru_result["trend"]
            },
            "consensus": consensus,
            "average_prediction": round(avg_predicted, 2),
            "average_change_pct": round(avg_change_pct, 4)
        }
    
    def get_available_models(self) -> list:
        """แสดงรายการโมเดลที่มีในระบบ"""
        models = []
        
        if not os.path.exists(config.MODEL_DIR):
            return models
        
        for filename in os.listdir(config.MODEL_DIR):
            if filename.endswith(".h5"):
                parts = filename.replace(".h5", "").split("_")
                if len(parts) == 3:
                    model_type, coin, timeframe = parts
                    
                    scaler_path = self._get_scaler_path(coin, timeframe)
                    has_scaler = os.path.exists(scaler_path)
                    
                    models.append({
                        "model_type": model_type.upper(),
                        "coin": coin.upper(),
                        "timeframe": timeframe,
                        "filename": filename,
                        "has_scaler": has_scaler,
                        "status": "ready" if has_scaler else "missing_scaler"
                    })
        return models
    
    def health_check(self) -> dict:
        """ตรวจสอบสถานะระบบ"""
        models = self.get_available_models()
        
        ready_models = [m for m in models if m["has_scaler"]]
        missing_scaler = [m for m in models if not m["has_scaler"]]
        
        return {
            "status": "healthy" if ready_models else "degraded",
            "total_models": len(models),
            "ready_models": len(ready_models),
            "models_missing_scaler": len(missing_scaler),
            "available_models": models
        }


# -------------------------
# Singleton Instance
# -------------------------
predictor = RealTimePredictor()


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    # ANSI Color Codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def get_trend_color(trend):
        """คืนค่าสีตาม trend"""
        if trend == "bullish":
            return GREEN
        elif trend == "bearish":
            return RED
        else:
            return YELLOW
    
    print(f"\n{CYAN}{BOLD}[TESTING] Real-time Predictor v3.0 (FIXED)...{RESET}\n")
    
    # Health check
    print(f"{MAGENTA}[HEALTH CHECK]{RESET}")
    health = predictor.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Total Models: {health['total_models']}")
    print(f"   Ready: {health['ready_models']}")
    print(f"   Missing Scaler: {health['models_missing_scaler']}")
    
    print("\n" + "="*60 + "\n")
    
    # Test available models
    print(f"{MAGENTA}[MODELS] Available Models:{RESET}")
    for m in predictor.get_available_models():
        status_color = GREEN if m['has_scaler'] else RED
        print(f"   - {m['model_type']} {m['coin']} {m['timeframe']} "
              f"[{status_color}{m['status']}{RESET}]")
    
    print("\n" + "="*60 + "\n")
    
    # Test prediction
    try:
        result = predictor.predict_both_models("btc", "1h")
        
        print(f"{CYAN}{BOLD}[PRICE] {result['coin']} ({result['timeframe']}){RESET}")
        print(f"   Current Price: {YELLOW}${result['current_price']:,.2f}{RESET}")
        
        # LSTM
        lstm_color = get_trend_color(result['lstm']['trend'])
        print(f"\n{CYAN}[LSTM Prediction]{RESET}")
        print(f"   Predicted: {YELLOW}${result['lstm']['predicted_price']:,.2f}{RESET}")
        print(f"   Change: {lstm_color}{result['lstm']['price_change_pct']:+.4f}%{RESET}")
        print(f"   Trend: {lstm_color}{result['lstm']['trend'].upper()}{RESET}")
        
        # GRU
        gru_color = get_trend_color(result['gru']['trend'])
        print(f"\n{CYAN}[GRU Prediction]{RESET}")
        print(f"   Predicted: {YELLOW}${result['gru']['predicted_price']:,.2f}{RESET}")
        print(f"   Change: {gru_color}{result['gru']['price_change_pct']:+.4f}%{RESET}")
        print(f"   Trend: {gru_color}{result['gru']['trend'].upper()}{RESET}")
        
        print(f"\n{MAGENTA}[CONSENSUS] {result['consensus']}{RESET}")
        print(f"{CYAN}[AVERAGE] Predicted: ${result['average_prediction']:,.2f} "
              f"({result['average_change_pct']:+.4f}%){RESET}")
        
    except Exception as e:
        print(f"{RED}[ERROR] {e}{RESET}")
        import traceback
        traceback.print_exc()
