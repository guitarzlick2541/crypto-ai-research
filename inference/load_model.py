
import os
from tensorflow.keras.models import load_model as keras_load_model

# -------------------------
# Path Configuration
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -------------------------
# แคชส่วนกลาง (Global Cache)
# -------------------------
_models_cache = {}

def get_model_path(model_type: str, coin: str, timeframe: str) -> str:
    """สร้าง path ของไฟล์โมเดล"""
    filename = f"{model_type.lower()}_{coin.lower()}_{timeframe}.h5"
    return os.path.join(MODEL_DIR, filename)

def load_model(model_type: str, coin: str, timeframe: str):
    """โหลดโมเดลจากไฟล์ .h5 (ใช้ cache)"""
    cache_key = f"{model_type}_{coin}_{timeframe}"
    
    if cache_key not in _models_cache:
        model_path = get_model_path(model_type, coin, timeframe)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"[INFO] Loading model: {model_path}")
        # ใช้ compile=False เพื่อหลีกเลี่ยงปัญหา Keras version compatibility
        # เราไม่ต้องการ optimizer/metrics สำหรับ inference
        _models_cache[cache_key] = keras_load_model(model_path, compile=False)
    
    return _models_cache[cache_key]

def get_available_models() -> list:
    """แสดงรายการโมเดลที่มี"""
    models = []
    if not os.path.exists(MODEL_DIR):
        return models

    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".h5"):
            parts = filename.replace(".h5", "").split("_")
            if len(parts) == 3:
                model_type, coin, timeframe = parts
                models.append({
                    "model_type": model_type.upper(),
                    "coin": coin.upper(),
                    "timeframe": timeframe,
                    "filename": filename
                })
    return models
