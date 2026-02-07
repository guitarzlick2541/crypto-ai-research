import requests
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)


def download(symbol, interval, filename, limit=1000):
    """ฟังก์ชันดาวน์โหลดข้อมูลจาก Binance"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    r = requests.get(url, params=params)
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
    ])

    # แปลง timestamp เป็น datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # แปลงเฉพาะคอลัมน์ตัวเลข
    price_cols = ["open", "high", "low", "close", "volume"]
    df[price_cols] = df[price_cols].astype(float)

    # เลือกเฉพาะคอลัมน์ที่จำเป็น
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df.to_csv(os.path.join(DATA_DIR, filename), index=False)
    # ANSI color codes สำหรับสีเขียว
    GREEN = "\033[92m"
    RESET = "\033[0m"
    print(f"{GREEN}[SUCCESS]{RESET} Saved {filename}")


if __name__ == "__main__":
    # ดาวน์โหลดข้อมูล (BTC, ETH)
    download("BTCUSDT", "1h", "btc_1h.csv")
    download("BTCUSDT", "4h", "btc_4h.csv")
    download("ETHUSDT", "1h", "eth_1h.csv")
    download("ETHUSDT", "4h", "eth_4h.csv")
