# 🔥 Industrial Fire & Smoke Detection System

A real-time AI-powered fire and smoke detection system built using **YOLOv8**, **OpenCV**, **Flask**, and **Python**.

This project detects fire and smoke from live camera feeds, video files, or RTSP streams and instantly triggers alarms, email notifications, dashboard updates, and logging systems.

---

## 🚀 Features

- 🔥 Real-time Fire Detection using a custom-trained YOLOv8 model
- 💨 Real-time Smoke Detection
- 📷 Webcam, Video File, and RTSP Stream Support
- 🌐 Live Flask Dashboard
- 📧 Email Alerts with Screenshots
- 🔔 Sound Alarm System
- 📊 CSV Logging
- 📑 Automatic HTML Session Reports
- 📷 Multi-Camera Monitoring
- 🛡️ False Positive Reduction using Scene Context Detection
- ⚡ CPU-Friendly Deployment

---

## 🏗️ System Workflow

```text
Camera Feed
    │
    ▼
OpenCV Processing
    │
    ▼
YOLOv8 Fire & Smoke Detection
    │
    ▼
Risk Classification
    │
    ├── Sound Alarm
    ├── Email Notification
    ├── Dashboard Update
    └── CSV Logging
```

---

## 🧠 Model Information

### Dataset

- D-Fire Dataset
- 20,320 Images

### Performance

| Metric | Fire | Smoke | Overall |
|----------|----------|----------|----------|
| mAP@50 | 66.0% | 77.2% | 71.6% |
| Precision | 69.4% | 75.9% | 72.0% |
| Recall | 59.2% | 71.6% | 65.6% |

---

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Flask
- PyTorch
- Pygame
- Pillow

---

## 📦 Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### Install Dependencies

```bash
pip install opencv-python ultralytics torch torchvision flask pygame pillow
```

---

## ▶️ Usage

### Webcam

```bash
python app.py 0
```

### Video File

```bash
python app.py fire_test.mp4
```

### RTSP Stream

```bash
python app.py rtsp://YOUR_STREAM_URL
```

### Disable Scene Detection

```bash
python app.py 0 --no-yolo
```

---

## ⚠️ Risk Levels

| Level | Description |
|---------|------------|
| 🟢 CLEAR | No fire or smoke detected |
| 🟡 CAUTION | Confidence > 35% |
| 🟠 WARNING | Confidence > 55% |
| 🔴 CRITICAL | Confidence > 75% |

---

## 📧 Email Alert Configuration

Configure your Gmail App Password inside the application:

```python
EMAIL_CFG = {
    "enabled": True,
    "sender_email": "your_email@gmail.com",
    "sender_password": "your_gmail_app_password",
    "recipients": ["recipient@gmail.com"]
}
```

---

## 🌐 Dashboard

After starting the application, open:

```text
http://localhost:5000
```

Dashboard Features:

- Live Camera Feed
- Detection Statistics
- Confidence Monitoring
- Alert Logs
- Risk Status Display

---

## 📂 Project Structure

```text
FireSmoke/
│
├── app.py
├── dashboard.html
├── fire_yolo.pt
├── yolov8n.pt
├── alarm.mp3
│
└── detection_logs/
    ├── detection_log.csv
    ├── alert_log.csv
    ├── report.html
    └── screenshots/
```

---

## 🎯 Future Improvements

- SMS Alerts
- Telegram Notifications
- Mobile Dashboard
- Cloud Deployment
- Edge AI Optimization
- Emergency Service Integration

---

## 👨‍💻 Author

**Bade Hari Preetham**

Computer Science Student | Computer Vision Enthusiast

### Skills

- Computer Vision
- Deep Learning
- YOLOv8
- OpenCV
- Python
- Flask

---

## 📜 License

This project is licensed under the MIT License.