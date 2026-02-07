# ผลการทดลอง (Experimental Results)

โฟลเดอร์นี้เก็บข้อมูลผลลัพธ์การทำนายดิบ (Raw Prediction) ของแต่ละโมเดล

## รูปแบบไฟล์ (File Format)
- **ชื่อไฟล์**: `{coin}_{model}_{timeframe}_predictions.csv`
- **คอลัมน์**:
  - `timestamp`: วันและเวลาของข้อมูล
  - `actual`: ราคาปิดจริง
  - `predicted`: ราคาที่ทำนายโดยโมเดล AI
