
import sys
import os

# เพิ่ม path ของ project root เพื่อให้สามารถ import modules ได้
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predictor_fixed import predictor
from inference import load_model

def test_inference_modules():
    print("\n" + "="*50)
    print("Testing Inference Modules")
    print("="*50)

    # ทดสอบฟังก์ชันโหลดโมเดล (Test load_model)
    print("\n[Testing load_model]")
    models = load_model.get_available_models()
    print(f"Found {len(models)} models.")
    for m in models:
        print(f" - {m['model_type']} ({m['coin']} {m['timeframe']})")

    # ทดสอบการทำนาย (Test predict)
    print("\n[Testing predict]")
    try:
        result = predictor.predict_both_models("btc", "1h")
        print(f"Prediction successful for BTC 1h:")
        print(f" - LSTM: ${result['lstm']['predicted_price']:,.2f} ({result['lstm']['trend']})")
        print(f" - GRU: ${result['gru']['predicted_price']:,.2f} ({result['gru']['trend']})")
        print(f" - Consensus: {result['consensus']}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference_modules()
