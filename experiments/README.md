# ผลการทดลอง (Experimental Results)

โฟลเดอร์นี้เก็บข้อมูลผลลัพธ์การทำนายดิบ (Raw Prediction) ของแต่ละโมเดล

## รูปแบบไฟล์ (File Format)
- **ชื่อไฟล์**: `{coin}_{model}_{timeframe}_predictions.csv`
- **คอลัมน์**:
  - `timestamp`: วันและเวลาของข้อมูล
  - `actual`: ราคาปิดจริง
  - `predicted`: ราคาที่ทำนายโดยโมเดล AI

## ตัวชี้วัดที่ใช้ (Evaluation Metrics)
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error (%)
- **DA**: Directional Accuracy (%) — ความแม่นยำในการทำนายทิศทางราคา

## Baseline Models
- **Naive Forecast**: ใช้ราคาปิดของแท่งเทียนก่อนหน้าเป็นค่าทำนาย
- **MA(7)**: ค่าเฉลี่ยเคลื่อนที่ 7 ช่วงเวลาก่อนหน้า
