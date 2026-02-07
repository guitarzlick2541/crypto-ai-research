"""
Test Script for Fixed Predictor
=================================
ทดสอบการทำงานของ predictor_fixed.py

วิธีใช้:
    python test_fixed_predictor.py

หรือทดสอบเฉพาะบางฟังก์ชัน:
    python test_fixed_predictor.py --test health
    python test_fixed_predictor.py --test validation
    python test_fixed_predictor.py --test prediction
"""

import sys
import os
import argparse

# เพิ่ม path ให้หา module ได้
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from inference.predictor_fixed import RealTimePredictor, predictor


def test_health_check():
    """ทดสอบ health check"""
    print("\n" + "="*60)
    print("[TEST 1] Health Check")
    print("="*60)
    
    health = predictor.health_check()
    print(f"Status: {health['status']}")
    print(f"Total Models: {health['total_models']}")
    print(f"Ready: {health['ready_models']}")
    print(f"Missing Scaler: {health['models_missing_scaler']}")
    
    if health['total_models'] == 0:
        print("\n⚠️ WARNING: No models found!")
        print("   Please train models first: python train/run_training.py")
    
    return health['status'] != "error"


def test_input_validation():
    """ทดสอบ input validation"""
    print("\n" + "="*60)
    print("[TEST 2] Input Validation")
    print("="*60)
    
    test_cases = [
        ("btc", "1h", "LSTM", True),
        ("BTC", "1H", "lstm", True),  # Should work (normalized)
        ("eth", "4h", "GRU", True),
        ("xrp", "1h", "LSTM", False),  # Invalid coin
        ("btc", "5m", "LSTM", False),  # Invalid timeframe
        ("btc", "1h", "CNN", False),   # Invalid model
    ]
    
    all_passed = True
    for coin, tf, model, should_pass in test_cases:
        try:
            predictor._validate_inputs(coin, tf, model)
            result = "PASS"
            if not should_pass:
                result = "FAIL (should have raised error)"
                all_passed = False
        except ValueError as e:
            result = "PASS (validation caught error)" if not should_pass else f"FAIL: {e}"
            if should_pass:
                all_passed = False
        
        status = "✅" if ("PASS" in result) else "❌"
        print(f"   {status} {coin}/{tf}/{model}: {result}")
    
    return all_passed


def test_scaler_loading():
    """ทดสอบการโหลด scaler"""
    print("\n" + "="*60)
    print("[TEST 3] Scaler Loading (CRITICAL FIX)")
    print("="*60)
    
    # ตรวจสอบว่ามี scaler files หรือไม่
    models = predictor.get_available_models()
    
    if not models:
        print("⚠️ No models found. Skipping scaler test.")
        return True
    
    for model in models:
        coin = model['coin'].lower()
        tf = model['timeframe']
        has_scaler = model['has_scaler']
        
        scaler_path = predictor._get_scaler_path(coin, tf)
        status = "✅" if has_scaler else "❌"
        
        print(f"   {status} {model['model_type']} {coin.upper()}/{tf}: ", end="")
        
        if has_scaler:
            try:
                scaler = predictor._load_scaler(coin, tf)
                print(f"scaler loaded successfully (type: {type(scaler).__name__})")
            except Exception as e:
                print(f"ERROR loading scaler: {e}")
                return False
        else:
            print(f"MISSING - will use fallback (predictions may be inaccurate)")
    
    return True


def test_prediction():
    """ทดสอบการทำนาย"""
    print("\n" + "="*60)
    print("[TEST 4] Real-time Prediction (LIVE TEST)")
    print("="*60)
    
    # ตรวจสอบว่ามีโมเดลพร้อมใช้หรือไม่
    health = predictor.health_check()
    if health['ready_models'] == 0:
        print("⚠️ No ready models found. Skipping prediction test.")
        print("   Please train models first with: python train/run_training.py")
        return True
    
    try:
        print("Fetching data and making prediction for BTC/1h...")
        result = predictor.predict("btc", "1h", "LSTM")
        
        print(f"\n✅ Prediction Successful!")
        print(f"   Current Price: ${result['current_price']:,.2f}")
        print(f"   Predicted: ${result['predicted_price']:,.2f}")
        print(f"   Change: {result['price_change_pct']:+.4f}%")
        print(f"   Trend: {result['trend']}")
        print(f"   Scaler Used: {result.get('scaler_used', 'unknown')}")
        
        # ตรวจสอบว่าใช้ scaler ที่ train ไว้หรือไม่
        if result.get('scaler_used') == 'trained':
            print("   ✅ Using TRAINED scaler (FIXED!)")
        else:
            print("   ⚠️ Using FALLBACK scaler (predictions may be less accurate)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Prediction Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """ทดสอบ API endpoints (ต้องมี Flask server รันอยู่)"""
    print("\n" + "="*60)
    print("[TEST 5] API Endpoints (requires running server)")
    print("="*60)
    
    import requests
    
    base_url = "http://localhost:5000"
    
    endpoints = [
        ("/api/health", "Health Check"),
        ("/api/models", "Models List"),
        ("/api/predict?coin=btc&tf=1h&model=lstm", "Prediction"),
    ]
    
    all_passed = True
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"   {status} {name}: HTTP {response.status_code}")
            
            if response.status_code != 200:
                all_passed = False
                
        except requests.exceptions.ConnectionError:
            print(f"   ⚠️ {name}: Server not running (skip)")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
            all_passed = False
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test Fixed Predictor")
    parser.add_argument(
        "--test",
        choices=["all", "health", "validation", "scaler", "prediction", "api"],
        default="all",
        help="เลือก test ที่ต้องการรัน"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" CRYPTO-AI PREDICTOR - FIXED VERSION TEST")
    print("="*60)
    
    results = {}
    
    if args.test in ["all", "health"]:
        results["health"] = test_health_check()
    
    if args.test in ["all", "validation"]:
        results["validation"] = test_input_validation()
    
    if args.test in ["all", "scaler"]:
        results["scaler"] = test_scaler_loading()
    
    if args.test in ["all", "prediction"]:
        results["prediction"] = test_prediction()
    
    if args.test in ["all", "api"]:
        results["api"] = test_api_endpoints()
    
    # Summary
    print("\n" + "="*60)
    print("[SUMMARY]")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The fixed predictor is working correctly.")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
