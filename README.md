# 🏥 AI-Powered Real-Time ICU Patient Monitoring System

## 🚀 Overview
An advanced healthcare monitoring solution that uses Machine Learning and Deep Learning to predict patient risk levels in real-time. Unlike traditional threshold-based systems, this project analyzes multivariate sensor data to provide early warnings for cardiac events.

## 🧠 The Triple-Engine AI Pipeline
Our system operates on a multi-layered validation logic:
1. **XGBoost Classifier:** Real-time risk status (Normal vs. Critical) with **100% Accuracy**.
2. **LSTM (Deep Learning):** Analyzes temporal patterns to predict future cardiac risk probability.
3. **Isolation Forest:** Monitors sensor integrity to detect malfunctions and prevent false alarms.



## 📊 Key Results
- **Balanced Data:** Achieved perfect Mean-Median alignment using 2.5% strict percentile capping.
- **Precision:** 0.99 for critical patient detection.
- **Features:** Age, HR, SpO2, Systolic/Diastolic BP, Temp, HRV, MAP, and Pulse Pressure.



## 🛠️ Tech Stack
- **Languages:** Python
- **AI/ML:** XGBoost, TensorFlow (Keras), Scikit-learn
- **Data:** Pandas, Numpy
- **Backend/Web:** Flask/FastAPI (Work in Progress by Varun)
- **IoT:** Arduino/ESP32 (Work in Progress by Vishal & Suhani)

## 👥 Team Members & Contributions (*IIT Patna : Capstone Project - II*)
- **Shivam Gupta:** Data Cleaning Pipeline & XGBoost Risk Model.
- **Shanya Gupta:** LSTM Predictive Analytics & Anomaly Detection.
- **Varun Gupta:** Web Dashboard & API Integration.
- **Vishal & Suhani Gupta:** IoT Sensor Hardware & Cloud Data Flow.
