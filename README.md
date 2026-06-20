🚨 IoT Anomaly Detector

⚡ Machine-Learning Based Attack Detection for Bot-IoT & IoT-23 Datasets

Frontend Dashboard + Flask Backend + ML Models

📌 Overview

The IoT Anomaly Detector is a machine learning–powered system designed to detect malicious activity in IoT networks.
It supports:

Two ML models:

Bot-IoT Model (Random Forest)

IoT-23 Model (Isolation Forest / Random Forest)

Real-time anomaly detection

Interactive dashboard for prediction & visualization

The dashboard allows you to:

✔ Upload a CSV file
✔ Choose the ML model
✔ Auto-tune threshold
✔ View metrics, ROC curve, confusion matrix
✔ See device-wise anomaly summary

🖼 Project Screenshots


🔹 Dashboard – Run Prediction

<img width="1899" height="859" alt="image" src="https://github.com/user-attachments/assets/4ad6deed-c678-4599-92a7-6ceea09bc273" />




🔹 Confusion Matrix

<img width="817" height="555" alt="image" src="https://github.com/user-attachments/assets/88490514-62ad-442d-935c-246fb6a44a87" />




🔹 Device Summary Page

<img width="1886" height="855" alt="image" src="https://github.com/user-attachments/assets/e5af4eae-3cb9-4571-8878-26b752fd6dd8" />



🚀 Features

🟢 1. Multi-Model Support

Bot-IoT ML model

IoT-23 ML model

User can select model from dropdown

🟢 2. Threshold Auto-Tuning

Uses Youden’s Index (TPR − FPR) to find optimal threshold.

🟢 3. Real-time Prediction Dashboard

Displays:

Accuracy
Precision
Recall
F1-Score
FPR
AUC
MSE
Donut chart (Normal vs Attack)
ROC Curve
Confusion Matrix

🟢 4. Device-Wise Attack Tracking

Shows:

Device IP
Packet count
Bytes
Attack type
Anomaly percentage
Last seen time



⚙️ Installation & Setup

1️⃣ Clone the repository
git clone https://github.com/Ashutosh-s-6/IoT-Anomaly-Detector.git
cd IoT-Anomaly-Detector

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run backend
cd backend
python app.py

Backend URL:

http://127.0.0.1:5000

5️⃣ Open the dashboard

Open frontend/index.html in your browser.

🧠 Machine Learning Models Used
Bot-IoT Model

Model: Random Forest Classifier

Dataset: UNSW Bot-IoT 2018

Preprocessing:
Label encoding
Scaling
Attack mapping
Feature selection
IoT-23 Model

Model: Random Forest / Isolation Forest

Dataset: IoT-23 by Stratosphere Lab

Preprocessing:

PCAP → CSV conversion
Feature extraction
Attack labeling

📊 Evaluation Metrics

We evaluate using:

Accuracy
Precision
Recall (TPR)
F1-Score
False Positive Rate (FPR)
ROC Curve
AUC Score
Confusion Matrix

🎯 Threshold Explanation

Threshold determines the cutoff for classifying:

Anomaly if score ≥ threshold
Normal if score < threshold

Auto threshold uses:

⭐ Youden's Index (J = TPR − FPR)

This finds the threshold giving best balance between sensitivity and specificity.

📌 Future Improvements

Add LSTM/Deep Learning model
Deploy on cloud (AWS or Azure)
Add live packet capture
Add user authentication & role-based access
