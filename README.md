# ระบบเว็บไซต์วิเคราะห์แนวโน้มราคาคริปโตเคอร์เรนซีด้วยปัญญาประดิษฐ์
**Cryptocurrency Price Trend Analysis System using Artificial Intelligence**

## 1. บทนำและภาพรวม (Project Overview)
โปรเจคนี้เป็นการพัฒนาระบบวิเคราะห์และพยากรณ์ราคาคริปโตเคอร์เรนซี (Cryptocurrency) ในระยะสั้น โดยมุ่งเน้นการเปรียบเทียบประสิทธิภาพของแบบจำลอง Deep Learning ประเภท **Recurrent Neural Networks (RNNs)** สองสถาปัตยกรรม ได้แก่ **Long Short-Term Memory (LSTM)** และ **Gated Recurrent Unit (GRU)**

ระบบถูกออกแบบมาให้ทำงานแบบ End-to-End ตั้งแต่การดึงข้อมูลราคาย้อนหลัง (Historical Data), การประมวลผลข้อมูล (Preprocessing), การคำนวณ Technical Indicators, การฝึกสอนโมเดล (Training), จนถึงการนำไปใช้งานจริงผ่านเว็บแอปพลิเคชัน (Web Interface) ที่แสดงผลการพยากรณ์แบบ Real-time

---

## 2. วัตถุประสงค์ของโครงการ (Objectives)
1. เพื่อศึกษาและเปรียบเทียบประสิทธิภาพระหว่างโมเดล LSTM และ GRU ในการพยากรณ์ราคา Cryptocurrency (BTC, ETH)
2. เพื่อพัฒนาระบบต้นแบบ (Prototype) ที่สามารถแสดงผลแนวโน้มราคาและคำแนะนำการลงทุนเบื้องต้นได้
3. เพื่อประยุกต์ใช้ความรู้ด้าน Deep Learning และ Software Engineering ในการแก้ปัญหาจริงทางเศรษฐศาสตร์ดิจิทัล

---

## 3. ขอบเขตของระบบ (Scope)
*   **ข้อมูลนำเข้า:** ข้อมูล OHLCV (Open, High, Low, Close, Volume) ย้อนหลัง 5,000+ แท่งเทียน จากกระดานเทรด Binance พร้อม Technical Indicators 10 ตัว
*   **เหรียญที่รองรับ:** Bitcoin (BTC) และ Ethereum (ETH)
*   **ช่วงเวลา (Timeframe):** 1 ชั่วโมง (1h) และ 4 ชั่วโมง (4h)
*   **แบบจำลอง:** LSTM และ GRU (สถาปัตยกรรม 128→64→32 Units พร้อม Dropout + BatchNormalization)
*   **คุณลักษณะ (Features):** 15 features ได้แก่ OHLCV, SMA(7,25), EMA(12,26), RSI(14), MACD, Bollinger Bands, Returns
*   **ผลลัพธ์:** ราคาที่คาดการณ์ในแท่งเทียนถัดไป (Next Candle Prediction) และแนวโน้ม (Bullish/Bearish/Neutral)

---

## 4. โครงสร้างโปรเจค (Project Structure)
ระบบถูกจัดเก็บและแยกส่วนการทำงานตามหลักการ Software Engineering ดังนี้:

```
crypto-ai-research/
├── data/                       # จัดการข้อมูลนำเข้า
│   └── raw/                    # เก็บไฟล์ CSV ข้อมูลราคาย้อนหลัง (btc_1h.csv, eth_4h.csv ฯลฯ)
├── train/                      # โมดูลสำหรับการฝึกสอนโมเดล (Machine Learning Pipeline)
│   ├── config.py               # ศูนย์รวมค่าคงที่ (Hyperparameters, Feature Columns, API Config)
│   ├── preprocessing.py        # การเตรียมข้อมูล, Feature Engineering, Normalization
│   ├── train_model.py          # โค้ดหลักในการสร้างและเทรนโมเดล LSTM/GRU
│   └── run_training.py         # สคริปต์ควบคุมการรัน Experiment ทั้งหมด
├── inference/                  # โมดูลสำหรับการนำโมเดลไปใช้งาน (Inference Engine)
│   ├── load_model.py           # จัดการการโหลดไฟล์ .h5 และ Scaler
│   └── predictor_fixed.py      # ระบบพยากรณ์ Real-time (พร้อม Retry Logic และ Validation)
├── evaluation/                 # การประเมินผล
│   ├── evaluate_model.py       # สคริปต์คำนวณค่า Error (MAE, RMSE, MAPE) จากชุดข้อมูลทดสอบ
│   └── metrics.py              # ฟังก์ชัน Metrics กลาง (MAE, RMSE, MAPE)
├── utils/                      # โมดูลยูทิลิตี้ที่ใช้ร่วมกัน (Shared Utilities)
│   ├── indicators.py           # การคำนวณ Technical Indicators (ใช้ร่วมระหว่าง Training + Inference)
│   └── scaling.py              # ฟังก์ชัน Inverse Transform สำหรับ Multi-Feature Scaler
├── web/                        # ส่วนต่อประสานผู้ใช้ (User Interface)
│   ├── app.py                  # Flask Server และ API Endpoints
│   ├── templates/              # HTML Files (Dashboard)
│   └── static/                 # CSS และ JavaScript (Chart.js)
├── scripts/                    # สคริปต์เสริมและเครื่องมือ
│   ├── download_data.py        # ดาวน์โหลดข้อมูล OHLCV จาก Binance API (5,000+ แถว)
│   ├── test_inference.py       # เทสต์ระบบ Inference
│   └── debug_chart.py          # เทสต์การดึงข้อมูลกราฟ
├── tests/                      # ชุดทดสอบอัตโนมัติ
│   └── test_fixed_predictor.py # Unit Test สำหรับ Predictor (Health, Validation, Scaler, Prediction)
├── models/                     # ที่เก็บไฟล์ Model (.h5) และ Scaler (.save) ที่ผ่านการเทรนแล้ว
├── experiments/                # บันทึกผลการทดลอง (Logs, Predictions, CSV Reports)
├── docs/                       # เอกสารประกอบโครงการ (System Architecture, Workflow Diagram)
├── requirements.txt            # รายการ Library ที่จำเป็น
├── render.yaml                 # การตั้งค่าสำหรับ Deploy บน Render
└── Procfile                    # กำหนดคำสั่งเริ่มต้นสำหรับ Web Server
```

---

## 5. หลักการทำงานของระบบ (System Workflow)

### Phase 1: Data Collection & Preparation
*   ดึงข้อมูล OHLCV (Open, High, Low, Close, Volume) 5,000+ แท่งเทียนผ่าน Binance API โดยใช้ Pagination
*   คำนวณ **Technical Indicators** จำนวน 10 ตัว ผ่านโมดูล `utils/indicators.py`:
    - **SMA** (Simple Moving Average: 7, 25 วัน)
    - **EMA** (Exponential Moving Average: 12, 26 วัน)
    - **RSI** (Relative Strength Index: 14 วัน)
    - **MACD** + Signal Line
    - **Bollinger Bands** (Upper, Lower)
    - **Price Returns** (% change)
*   ทำ **Data Normalization** โดยใช้ `MinMaxScaler` (fit เฉพาะ Training Set เพื่อป้องกัน Data Leakage)
*   ทำ **Sliding Window Technique** เพื่อแปลงข้อมูล Time-series เป็นรูปแบบ Supervised Learning
    - Input (X): 60 แท่งเทียนก่อนหน้า × 15 features
    - Output (y): ราคาปิดแท่งเทียนถัดไป
*   แบ่งข้อมูลเป็น Training Set (80%) และ Test Set (20%) โดยไม่มีการสลับลำดับเวลา (No Shuffle)

### Phase 2: Model Training
*   **สถาปัตยกรรมโมเดล (Enhanced Architecture):**
    - Layer 1: LSTM/GRU 128 Units + Dropout(0.3) + BatchNormalization
    - Layer 2: LSTM/GRU 64 Units + Dropout(0.3) + BatchNormalization
    - Layer 3: LSTM/GRU 32 Units + Dropout(0.2)
    - Dense Layer: 16 Units (ReLU) → 1 Unit (Linear Output)
*   **Optimizer:** Adam
*   **Loss Function:** Mean Squared Error (MSE)
*   **Callbacks:**
    - `EarlyStopping` (patience=15, restore_best_weights=True) — หยุดเทรนอัตโนมัติเมื่อไม่มีการปรับปรุง
    - `ReduceLROnPlateau` (factor=0.5, patience=5) — ลด Learning Rate อัตโนมัติ
*   **Epochs:** สูงสุด 200 รอบ (จำกัดด้วย EarlyStopping)

### Phase 3: Inference & Serving
*   ระบบ Frontend Request ข้อมูลไปยัง Flask API
*   Backend ดึงข้อมูลราคา Real-time ล่าสุด 160 แท่งเทียน (60 สำหรับ Window + 100 สำหรับ Indicator Warmup)
*   คำนวณ Technical Indicators 10 ตัว ด้วยโมดูล `utils/indicators.py` (สูตรเดียวกับตอนเทรน)
*   ทำการ Scale ข้อมูลเข้าสู่ Range 0-1 โดยใช้ Scaler ตัวเดียวกับที่ใช้ตอนเทรน (**Prevent Data Leakage**)
*   โมเดลประมวลผลและทำนายราคา
*   Inverse Transform ค่ากลับมาเป็นราคาจริง ($USD) ผ่านโมดูล `utils/scaling.py` และส่งกลับไปยัง Frontend

---

## 6. เทคโนโลยีและเครื่องมือ (Tech Stack)
*   **Programming Language:** Python 3.9+
*   **Deep Learning Framework:** TensorFlow / Keras
*   **Data Processing:** Pandas, NumPy, Scikit-learn
*   **Model Persistence:** Joblib (Scaler), HDF5 (Keras Model)
*   **Web Framework:** Flask
*   **Frontend Library:** HTML5, CSS3, JavaScript, Chart.js
*   **External API:** Binance Public Data API (Global + US Fallback)
*   **Deployment:** Render (render.yaml + Procfile)

---

## 7. วิธีการใช้งานระบบ (How to Run)

### 7.1 การติดตั้ง (Installation)
1. Clone โปรเจคและติดตั้ง Library ที่จำเป็น
   ```bash
   git clone https://github.com/guitarzlick2541/crypto-ai-research.git
   cd crypto-ai-research
   pip install -r requirements.txt
   ```

### 7.2 การเตรียมข้อมูลและฝึกสอนโมเดล (Data & Training)
หากต้องการเทรนโมเดลใหม่ด้วยข้อมูลล่าสุด:
1. ดาวน์โหลดข้อมูล (5,000+ แท่งเทียนต่อชุดข้อมูล):
   ```bash
   python scripts/download_data.py
   ```
2. รันกระบวนการเทรน (ระบบจะเทรนทั้ง LSTM และ GRU สำหรับทุกเหรียญและทุก Timeframe):
   ```bash
   python train/run_training.py
   ```
   *ไฟล์โมเดล (.h5) และ Scaler (.save) จะถูกบันทึกที่โฟลเดอร์ `models/`*

### 7.3 การรันเว็บแอปพลิเคชัน (Deployment)
1. รัน Flask Server:
   ```bash
   python web/app.py
   ```
2. เปิด Browser และเข้าใช้งานที่:
   `http://localhost:5000`

---

## 8. การประเมินผล (Evaluation Metrics)
ระบบใช้ตัวชี้วัดทางสถิติเพื่อวัดประสิทธิภาพความแม่นยำ (ฟังก์ชันทั้งหมดอยู่ใน `evaluation/metrics.py`):

1.  **MAE (Mean Absolute Error):** ค่าเฉลี่ยความคลาดเคลื่อนสัมบูรณ์ (ยิ่งน้อยยิ่งดี)
2.  **RMSE (Root Mean Squared Error):** รากที่สองของค่าเฉลี่ยความคลาดเคลื่อนกำลังสอง (เน้นการลงโทษ error ที่มีค่ามาก)
3.  **MAPE (Mean Absolute Percentage Error):** ค่าเฉลี่ยความผิดพลาดคิดเป็นร้อยละ

การประเมินผลด้วยชุดข้อมูลทดสอบ (Test Set) สามารถทำได้โดยรัน:
```bash
python evaluation/evaluate_model.py
```
*ผลลัพธ์จะถูกบันทึกที่ `experiments/evaluation_report.csv` และ `experiments/predictions/`*

---

## 9. ระบบทดสอบ (Testing & Verification)
เพื่อให้มั่นใจว่าระบบทำงานได้ถูกต้องและปราศจากข้อผิดพลาดร้ายแรง (Critical Bugs) โครงการนี้ได้จัดทำโมดูลทดสอบไว้ดังนี้:

### 9.1 การทดสอบระบบ Inference (Inference System Test)
ใช้สำหรับตรวจสอบความถูกต้องของการโหลดโมเดล Logic การพยากรณ์ และการเชื่อมต่อ Scaler:
```bash
python scripts/test_inference.py
```
*สิ่งที่ตรวจสอบ:* `Model Loading`, `Input Validation`, `Scaler Consistency`, `Prediction Output`

### 9.2 การทดสอบข้อมูลกราฟ (Chart Data Test)
ใช้สำหรับตรวจสอบการดึงข้อมูลจาก Binance API เพื่อแสดงผลกราฟ และยืนยันว่าราคาที่ได้เป็นราคาจริง (Real USD) ไม่ใช่ค่าที่ถูก Normalize:
```bash
python scripts/debug_chart.py
```
*สิ่งที่ตรวจสอบ:* `API Connectivity`, `Data Integrity`, `Price Scaling Logic`

### 9.3 การทดสอบเชิงลึก (Advanced Unit Test)
สำหรับนักพัฒนาที่ต้องการตรวจสอบทุกฟังก์ชันอย่างละเอียด (Unit Test) ครอบคลุมถึง Input Validation, Health Check และ API Endpoints:
```bash
python tests/test_fixed_predictor.py
```
*สิ่งที่ตรวจสอบ:* `Full System Health`, `Edge Cases`, `Scaler Integrity`, `API Response Status`

---

## 10. ข้อจำกัดของระบบ (Limitations)
1.  **การพึ่งพาข้อมูลในอดีต:** โมเดลเรียนรู้จากพฤติกรรมราคาในอดีต (Technical Analysis) เท่านั้น ไม่ได้นำปัจจัยข่าวสาร (Fundamental/Sentiment) มาวิเคราะห์
2.  **สภาวะตลาดผันผวนรุนแรง:** ในช่วงที่ตลาดมีความผันผวนสูงผิดปกติ (Market Crash/Pump) ความแม่นยำอาจลดลง
3.  **API Rate Limit:** การดึงข้อมูล Real-time อาจถูกจำกัดจำนวนครั้งจากทาง Binance หากมีการ Request ถี่เกินไป (ระบบมี Retry Logic + Fallback ไปยัง Binance US API)

---

## 11. หมายเหตุ (Note)
โครงการนี้เป็นส่วนหนึ่งของการศึกษาและวิจัยเพื่อการเรียนรู้ ไม่แนะนำให้ใช้อ้างอิงสำหรับการลงทุนจริงที่มีความเสี่ยงสูง ผู้พัฒนามิได้มีเจตนาชักชวนให้ลงทุนในสินทรัพย์ดิจิทัลแต่อย่างใด
