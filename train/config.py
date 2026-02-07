import os

# Base Paths (เส้นทางไฟล์พื้นฐาน)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "experiments")
# EXPERIMENT_DIR deprecated, use RESULT_DIR instead for consistency

# Ensure directories exist (ตรวจสอบว่ามีโฟลเดอร์อยู่หรือไม่)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Application Settings (การตั้งค่าแอพพลิเคชัน)
COINS = ["btc", "eth"]
TIMEFRAMES = ["1h", "4h"]

# Model Hyperparameters (ค่า Hyperparameters ของโมเดล)
WINDOW_SIZE = 60      # ขนาดของ Sliding Window (ระยะเวลาย้อนหลัง)
PREDICT_SIZE = 1      # ทำนายล่วงหน้า 1 step
EPOCHS = 50           # จำนวนรอบการเทรน
BATCH_SIZE = 32       # ขนาด Batch size
VALIDATION_SPLIT = 0.2
