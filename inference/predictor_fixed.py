"""
Real-time Prediction Engine - FIXED VERSION
=============================================
โหลดโมเดล LSTM/GRU ที่ train ไว้แล้ว และทำ prediction แบบ real-time
โดยดึงข้อมูลจาก Binance API

CHANGELOG:
- v2.0: Fixed scaler mismatch bug (CRITICAL)
- v2.0: Added retry logic for Binance API
- v2.0: Added input validation
- v2.0: Improved error handling
"""

import os
import numpy as np
import requests
import joblib
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# -------------------------
# Path Configuration
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# -------------------------
# Constants
# -------------------------
WINDOW_SIZE = 60  # จำนวน candles ที่ใช้ทำนาย (ต้องตรงกับตอน training)
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

# Mapping symbols
SYMBOL_MAP = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT"
}

# Valid inputs
VALID_COINS = ["btc", "eth"]
VALID_TIMEFRAMES = ["1h", "4h"]
VALID_MODELS = ["LSTM", "GRU"]


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
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP errors ที่จะ retry
            allowed_methods=["HEAD", "GET", "OPTIONS"]  # Retry เฉพาะ GET requests
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session
    
    def _validate_inputs(self, coin: str, timeframe: str, model_type: str = None) -> tuple:
        """
        ตรวจสอบความถูกต้องของ inputs
        
        Args:
            coin: ชื่อเหรียญ
            timeframe: ช่วงเวลา
            model_type: ประเภทโมเดล (optional)
            
        Returns:
            tuple: (coin_lower, timeframe_lower, model_type_upper)
            
        Raises:
            ValueError: ถ้า input ไม่ถูกต้อง
        """
        coin_lower = coin.lower()
        timeframe_lower = timeframe.lower()
        
        if coin_lower not in VALID_COINS:
            raise ValueError(
                f"Invalid coin: '{coin}'. "
                f"Must be one of: {VALID_COINS}"
            )
        
        if timeframe_lower not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe: '{timeframe}'. "
                f"Must be one of: {VALID_TIMEFRAMES}"
            )
        
        if model_type is not None:
            model_upper = model_type.upper()
            if model_upper not in VALID_MODELS:
                raise ValueError(
                    f"Invalid model_type: '{model_type}'. "
                    f"Must be one of: {VALID_MODELS}"
                )
            return coin_lower, timeframe_lower, model_upper
        
        return coin_lower, timeframe_lower, None
    
    def _get_model_path(self, model_type: str, coin: str, timeframe: str) -> str:
        """สร้าง path ของไฟล์โมเดล"""
        filename = f"{model_type.lower()}_{coin.lower()}_{timeframe}.h5"
        return os.path.join(MODEL_DIR, filename)
    
    def _get_scaler_path(self, coin: str, timeframe: str) -> str:
        """สร้าง path ของไฟล์ scaler"""
        filename = f"scaler_{coin.lower()}_{timeframe}.save"
        return os.path.join(MODEL_DIR, filename)
    
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
            # ใช้ compile=False เพื่อหลีกเลี่ยงปัญหา Keras version compatibility
            # เราไม่ต้องการ optimizer/metrics สำหรับ inference
            self.models[cache_key] = load_model(model_path, compile=False)
        
        return self.models[cache_key]
    
    def _load_scaler(self, coin: str, timeframe: str) -> MinMaxScaler:
        """
        โหลด scaler ที่ train ไว้ (CRITICAL FIX)
        
        แทนที่จะสร้าง scaler ใหม่ทุกครั้ง (fit_transform)
        เราโหลด scaler ที่ fit กับ training data มาใช้
        """
        cache_key = f"scaler_{coin}_{timeframe}"
        
        if cache_key not in self.scalers:
            scaler_path = self._get_scaler_path(coin, timeframe)
            
            if os.path.exists(scaler_path):
                # โหลด scaler ที่ train ไว้ (RECOMMENDED)
                print(f"[INFO] Loading scaler: {scaler_path}")
                self.scalers[cache_key] = joblib.load(scaler_path)
            else:
                # Fallback: ถ้าไม่มี scaler file (เช่น โมเดลเก่า)
                # ในกรณีนี้เราต้อง fit บน historical data ไม่ใช่แค่ 60 candles ล่าสุด
                print(f"⚠️ WARNING: Scaler not found at {scaler_path}")
                print("   Using fallback scaler (predictions may be less accurate)")
                # สร้าง scaler เปล่า จะ fit ใน preprocess_data
                self.scalers[cache_key] = None
        
        return self.scalers[cache_key]
    
    def fetch_live_data(self, coin: str, timeframe: str, limit: int = None, max_retries: int = 3) -> dict:
        """
        ดึงข้อมูล real-time จาก Binance API พร้อม retry mechanism (FIXED)
        
        Args:
            coin: 'btc' หรือ 'eth'
            timeframe: '1h' หรือ '4h'
            limit: จำนวน candles (default: WINDOW_SIZE + 10)
            max_retries: จำนวนครั้งที่ลองใหม่ถ้า fail (default: 3)
            
        Returns:
            dict: {
                'prices': list of close prices,
                'current_price': float,
                'timestamp': datetime,
                'ohlcv': list of OHLCV data
            }
            
        Raises:
            Exception: ถ้าดึงข้อมูลไม่สำเร็จหลังจาก retry หมด
        """
        # Validate inputs
        coin, timeframe, _ = self._validate_inputs(coin, timeframe)
        
        if limit is None:
            limit = WINDOW_SIZE + 10  # เผื่อ buffer
            
        symbol = SYMBOL_MAP.get(coin)
        if not symbol:
            raise ValueError(f"Unknown coin: {coin}")
        
        # Map timeframe to Binance interval format
        interval_map = {"1h": "1h", "4h": "4h", "1d": "1d"}
        interval = interval_map.get(timeframe, timeframe)
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            # ใช้ timeout ที่เหมาะสม (15 วินาที)
            response = self.session.get(
                BINANCE_API_URL, 
                params=params, 
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            # ตรวจสอบว่าได้ข้อมูลมาพอไหม (เฉพาะ prediction mode)
            # สำหรับ chart display (limit != None) ไม่ต้องตรวจสอบ
            if limit is None and len(data) < WINDOW_SIZE:
                raise ValueError(
                    f"Insufficient data from API: got {len(data)} candles, "
                    f"need at least {WINDOW_SIZE}"
                )
            
            # Extract close prices
            prices = [float(candle[4]) for candle in data]  # Index 4 = close price
            
            # OHLCV data
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
            
        except requests.exceptions.RequestException as e:
            raise Exception(
                f"Failed to fetch data from Binance after {max_retries} retries: {str(e)}"
            )
    
    def preprocess_data(self, prices: list, coin: str, timeframe: str) -> tuple:
        """
        Preprocessing pipeline - FIXED VERSION
        
        CRITICAL FIX: ใช้ scaler ที่ train ไว้ แทนที่จะสร้างใหม่ทุกครั้ง
        
        Args:
            prices: list ของราคา close
            coin: 'btc' หรือ 'eth' (ใช้โหลด scaler ที่ถูกต้อง)
            timeframe: '1h' หรือ '4h' (ใช้โหลด scaler ที่ถูกต้อง)
            
        Returns:
            tuple: (window, scaler)
                window: numpy array shape (1, WINDOW_SIZE, 1) พร้อมใส่โมเดล
                scaler: MinMaxScaler ที่ใช้
        """
        # ตรวจสอบว่ามีข้อมูลเพียงพอไหม
        if len(prices) < WINDOW_SIZE:
            raise ValueError(
                f"Not enough data from Binance. "
                f"Need {WINDOW_SIZE} candles, but got {len(prices)}"
            )

        # ใช้เฉพาะ 60 อันล่าสุด
        recent_prices = np.array(prices[-WINDOW_SIZE:]).reshape(-1, 1)
        
        # FIXED: โหลด scaler ที่ train ไว้ แทนที่จะสร้างใหม่
        scaler = self._load_scaler(coin, timeframe)
        
        if scaler is not None:
            # ใช้ scaler ที่ train ไว้ (RECOMMENDED)
            scaled_prices = scaler.transform(recent_prices)
        else:
            # Fallback: ถ้าไม่มี scaler file เก่า
            # ใช้ historical data มากกว่า 60 candles เพื่อ fit scaler
            print("⚠️ WARNING: Using fallback scaler fitting on recent data")
            scaler = MinMaxScaler()
            # ใช้ข้อมูลมากกว่า 60 ถ้ามี เพื่อให้ min/max แม่นยำกว่า
            available_data = np.array(prices).reshape(-1, 1)
            scaler.fit(available_data)
            scaled_prices = scaler.transform(recent_prices)
        
        # Reshape: (1, 60, 1) สำหรับ LSTM/GRU input
        window = scaled_prices.reshape(1, WINDOW_SIZE, 1)
        
        return window, scaler
    
    def predict(self, coin: str, timeframe: str, model_type: str = "LSTM") -> dict:
        """
        ทำ real-time prediction - FIXED VERSION
        
        Args:
            coin: 'btc' หรือ 'eth'
            timeframe: '1h' หรือ '4h'
            model_type: 'LSTM' หรือ 'GRU'
            
        Returns:
            dict: ผลลัพธ์ prediction พร้อม metadata
            
        Raises:
            ValueError: ถ้า inputs ไม่ถูกต้อง
            FileNotFoundError: ถ้าไม่มีโมเดล
            Exception: ถ้าเกิด error ระหว่าง prediction
        """
        # Validate inputs (FIXED: เพิ่ม validation)
        coin, timeframe, model_type = self._validate_inputs(coin, timeframe, model_type)
        
        try:
            # 1. ดึงข้อมูล real-time จาก Binance (พร้อม retry)
            print(f"[INFO] Fetching live data for {coin.upper()} {timeframe}...")
            live_data = self.fetch_live_data(coin, timeframe)
            prices = live_data["prices"]
            current_price = live_data["current_price"]
            
            # 2. Preprocessing (FIXED: ส่ง coin/timeframe เพื่อโหลด scaler ที่ถูกต้อง)
            print(f"[INFO] Preprocessing data using saved scaler...")
            window, scaler = self.preprocess_data(prices, coin, timeframe)
            
            # 3. โหลดโมเดล
            print(f"[INFO] Loading {model_type} model...")
            model = self._load_model(model_type, coin, timeframe)
            
            # 4. Predict (ได้ค่า scaled 0-1)
            print(f"[INFO] Running prediction...")
            scaled_prediction = model.predict(window, verbose=0)
            
            # 5. Inverse transform กลับเป็นราคาจริง (FIXED: ใช้ scaler ที่ถูกต้อง)
            predicted_price_raw = scaler.inverse_transform(
                scaled_prediction.reshape(-1, 1)
            )[0][0]
            
            # แปลงเป็น float เพื่อป้องกัน numpy type issues
            predicted_price = float(predicted_price_raw)
            current_price = float(current_price)
            
            # 6. คำนวณ change percentage
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # 7. Determine trend
            if price_change_pct > 0.5:
                trend = "bullish"
            elif price_change_pct < -0.5:
                trend = "bearish"
            else:
                trend = "neutral"
            
            print(f"[SUCCESS] Prediction completed")
            
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
                "data_points_used": WINDOW_SIZE,
                "scaler_used": "trained" if self._get_scaler_path(coin, timeframe) else "fallback"
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
        
        Args:
            coin: 'btc' หรือ 'eth'
            timeframe: '1h' หรือ '4h'
            
        Returns:
            dict: ผลลัพธ์จากทั้ง 2 โมเดล พร้อม consensus
        """
        lstm_result = self.predict(coin, timeframe, "LSTM")
        gru_result = self.predict(coin, timeframe, "GRU")
        
        # หาว่า model ไหนทำนายสูงกว่า
        if lstm_result["predicted_price"] > gru_result["predicted_price"]:
            consensus = "LSTM predicts higher"
        elif lstm_result["predicted_price"] < gru_result["predicted_price"]:
            consensus = "GRU predicts higher"
        else:
            consensus = "Both models agree"
        
        # คำนวณค่าเฉลี่ย
        avg_predicted = (lstm_result["predicted_price"] + gru_result["predicted_price"]) / 2
        avg_change_pct = (lstm_result["price_change_pct"] + gru_result["price_change_pct"]) / 2
        
        return {
            "coin": coin.upper(),
            "timeframe": timeframe,
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
        
        if not os.path.exists(MODEL_DIR):
            return models
        
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith(".h5"):
                parts = filename.replace(".h5", "").split("_")
                if len(parts) == 3:
                    model_type, coin, timeframe = parts
                    
                    # ตรวจสอบว่ามี scaler ด้วยไหม
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
        """
        ตรวจสอบสถานะระบบ
        
        Returns:
            dict: สถานะของระบบทั้งหมด
        """
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
    
    print(f"\n{CYAN}{BOLD}[TESTING] Real-time Predictor v2.0 (FIXED)...{RESET}\n")
    
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
