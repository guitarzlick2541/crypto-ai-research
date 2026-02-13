"""
train/config.py
===============
Centralized configuration for the crypto-ai-research project.
All shared constants are defined here to avoid duplication.
"""

import os


def _ensure_directories(*dirs):
    """สร้างโฟลเดอร์ที่จำเป็น (เรียกเมื่อต้องการเท่านั้นm ไม่ใช่ตอน import)"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# =====================================================================
# Base Paths (เส้นทางไฟล์พื้นฐาน)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "experiments")

# =====================================================================
# Application Settings (การตั้งค่าแอพพลิเคชัน)
# =====================================================================
COINS = ["btc", "eth"]
TIMEFRAMES = ["1h", "4h"]
MODEL_TYPES = ["LSTM", "GRU"]

# Binance API Symbol Mapping (ใช้ร่วมกันทั้ง download_data + predictor)
SYMBOL_MAP = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT"
}

# Binance API URLs
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
BINANCE_US_API_URL = "https://api.binance.us/api/v3/klines"

# =====================================================================
# Feature Engineering (คุณลักษณะที่ใช้ในการเทรน)
# =====================================================================
FEATURE_COLUMNS = [
    "close",       # ราคาปิด
    "open",        # ราคาเปิด
    "high",        # ราคาสูงสุด
    "low",         # ราคาต่ำสุด
    "volume",      # ปริมาณการซื้อขาย
    "sma_7",       # Simple Moving Average 7
    "sma_25",      # Simple Moving Average 25
    "ema_12",      # Exponential Moving Average 12
    "ema_26",      # Exponential Moving Average 26
    "rsi_14",      # Relative Strength Index 14
    "macd",        # MACD
    "macd_signal", # MACD Signal Line
    "bb_upper",    # Bollinger Band Upper
    "bb_lower",    # Bollinger Band Lower
    "returns",     # Price Returns (% change)
]
NUM_FEATURES = len(FEATURE_COLUMNS)

# =====================================================================
# Model Hyperparameters (ค่า Hyperparameters ของโมเดล)
# =====================================================================
WINDOW_SIZE = 60          # ขนาดของ Sliding Window (ระยะเวลาย้อนหลัง)
PREDICT_SIZE = 1          # ทำนายล่วงหน้า 1 step
EPOCHS = 200              # จำนวนรอบการเทรน (ใช้ร่วมกับ EarlyStopping)
BATCH_SIZE = 32           # ขนาด Batch
VALIDATION_SPLIT = 0.2    # สัดส่วน Validation
INDICATOR_WARMUP = 100    # จำนวน candles เพิ่มสำหรับคำนวณ indicators

# Dropout rates
DROPOUT_RATE_HIGH = 0.3
DROPOUT_RATE_LOW = 0.2

# Trend classification thresholds
TREND_THRESHOLD = 0.5     # % change threshold สำหรับ bullish/bearish

# API Rate Limiting
API_RATE_LIMIT_DELAY = 0.3  # seconds between API calls
API_TIMEOUT = 15            # seconds
