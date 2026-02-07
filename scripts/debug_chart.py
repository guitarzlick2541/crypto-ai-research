"""
Debug script to test API price endpoint
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predictor_fixed import predictor

def test_fetch_live_data():
    print("=" * 60)
    print("Testing fetch_live_data() from predictor_fixed")
    print("=" * 60)
    
    try:
        print("\n[1] Testing BTC 1h...")
        result = predictor.fetch_live_data("btc", "1h", 50)
        
        print(f"   Success: {len(result['ohlcv'])} candles received")
        print(f"   Current Price: ${result['current_price']:,.2f}")
        
        # Check first 3 OHLCV entries
        print("\n   Sample OHLCV data:")
        for i, candle in enumerate(result['ohlcv'][:3]):
            print(f"   [{i}] Close: ${candle['close']:,.2f}")
        
        # Check if prices are in normal range
        close_prices = [c['close'] for c in result['ohlcv']]
        min_price = min(close_prices)
        max_price = max(close_prices)
        
        print(f"\n   Price Range: ${min_price:,.2f} - ${max_price:,.2f}")
        
        if max_price < 10:
            print("   ⚠️ WARNING: Prices seem scaled (0-1 range)!")
        else:
            print("   ✅ Prices look normal (real USD)")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fetch_live_data()
