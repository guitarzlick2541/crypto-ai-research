"""
scripts/download_data.py
========================
ดาวน์โหลดข้อมูล OHLCV จาก Binance API สำหรับ BTC และ ETH
ดึงข้อมูลย้อนหลัง 5,000 แถว เพื่อใช้ในการเทรนโมเดล

Usage:
    python scripts/download_data.py
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from train import config

# Config (from shared config — DUP-05 FIX)
DATA_DIR = config.DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)

TARGET_ROWS = 5000  # จำนวนแถวที่ต้องการ


def fetch_klines(symbol, interval, limit=1000, end_time=None):
    """
    ดึงข้อมูล klines จาก Binance API (ครั้งละ 1000 แถว)
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if end_time:
        params["endTime"] = end_time

    try:
        response = requests.get(
            config.BINANCE_API_URL, params=params, timeout=config.API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"   ⚠️ Global API failed: {e}. Trying Binance US...")
        try:
            response = requests.get(
                config.BINANCE_US_API_URL, params=params, timeout=config.API_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as us_e:
            raise Exception(f"Both APIs failed. Global: {e}, US: {us_e}")


def download_coin_data(coin, timeframe, target_rows=TARGET_ROWS):
    """
    ดาวน์โหลดข้อมูลย้อนหลังจำนวนมาก โดยใช้ pagination
    """
    symbol = config.SYMBOL_MAP[coin]
    all_data = []
    end_time = None
    batch_size = 1000  # Binance max per request

    print(f"\n📥 Downloading {coin.upper()} {timeframe} data ({target_rows} candles)...")

    while len(all_data) < target_rows:
        remaining = target_rows - len(all_data)
        limit = min(remaining, batch_size)

        data = fetch_klines(symbol, timeframe, limit=limit, end_time=end_time)

        if not data:
            print(f"   ⚠️ No more data available. Got {len(all_data)} candles.")
            break

        all_data = data + all_data  # Prepend (older data first)

        # Set end_time for next batch (1ms before the oldest candle)
        end_time = data[0][0] - 1

        print(f"   📊 Fetched batch: {len(data)} candles (total: {len(all_data)})")

        # Rate limiting (from config — CC-01 FIX)
        time.sleep(config.API_RATE_LIMIT_DELAY)

    # สร้าง DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # แปลง timestamp เป็น datetime
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

    # เลือกเฉพาะคอลัมน์ที่ต้องการ
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # แปลงเป็น float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # ลบ duplicates (ถ้ามี)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # บันทึก
    filename = f"{coin}_{timeframe}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)

    print(f"   ✅ Saved: {filepath}")
    print(f"   📊 Total rows: {len(df)}")
    print(f"   📅 Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"   💰 Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")

    return df


def main():
    print("=" * 60)
    print(" BINANCE DATA DOWNLOADER")
    print("=" * 60)
    print(f" Target: {TARGET_ROWS} candles per dataset")
    print(f" Output: {DATA_DIR}")

    results = []
    for coin in config.COINS:
        for tf in config.TIMEFRAMES:
            try:
                df = download_coin_data(coin, tf)
                results.append({
                    "Coin": coin.upper(),
                    "Timeframe": tf,
                    "Rows": len(df),
                    "Status": "✅ OK"
                })
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results.append({
                    "Coin": coin.upper(),
                    "Timeframe": tf,
                    "Rows": 0,
                    "Status": f"❌ {e}"
                })

    print("\n" + "=" * 60)
    print(" DOWNLOAD COMPLETE")
    print("=" * 60)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
