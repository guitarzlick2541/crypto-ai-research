import os
import pandas as pd
import config
from train_model import train_model
import time
import uuid
from datetime import datetime

# Define Log File
LOG_FILE = os.path.join(config.RESULT_DIR, "experiment_log.csv")

def log_experiment(model_type, coin, timeframe, mae, rmse, mape, latency):
    """Log experiment results"""
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp,experiment_id,model_type,coin,timeframe,accuracy,mae,rmse,latency_ms,action,status,notes\n")
            
        timestamp = datetime.now().isoformat()
        experiment_id = str(uuid.uuid4())[:8]
        accuracy = max(0, 100 - mape) if mape is not None else 0
        action = "training"
        status = "success"
        notes = f"MAE={mae:.4f} | RMSE={rmse:.4f}"
        
        f.write(f"{timestamp},{experiment_id},{model_type},{coin},{timeframe},{accuracy:.2f},{mae:.4f},{rmse:.4f},{latency:.2f},{action},{status},{notes}\n")
    
    print(f"   📝 Logged experiment: {experiment_id}")

def run_all_training():
    """Run training for all coins and timeframes"""
    print("=" * 60)
    print("🚀 STARTED: Full Model Training Pipeline (Consolidated)")
    print("=" * 60)
    
    results = []
    
    # Ensure results directory exists
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    
    models_to_train = ["LSTM", "GRU"]
    
    for coin in config.COINS:
        for tf in config.TIMEFRAMES:
            for model_type in models_to_train:
                
                start_time = time.time()
                mae, rmse, mape = train_model(coin, tf, model_type)
                end_time = time.time()
                latency = (end_time - start_time) * 1000 # ms
                
                if mae is not None:
                    results.append({
                        "Coin": coin.upper(),
                        "Timeframe": tf,
                        "Model": model_type,
                        "MAE": mae,
                        "RMSE": rmse,
                    })
                    
                    log_experiment(model_type, coin, tf, mae, rmse, mape, latency)
    
    # Save Results
    if results:
        df = pd.DataFrame(results)
        result_path = os.path.join(config.RESULT_DIR, "dual_model_results.csv")
        df.to_csv(result_path, index=False)
        
        print("\n" + "=" * 60)
        print(" TRAINING COMPLETE")
        print("=" * 60)
        print(df)
        print(f"\n Results saved to: {result_path}")
    else:
        print("\n❌ No models were trained (failed to load data?)")

if __name__ == "__main__":
    run_all_training()
