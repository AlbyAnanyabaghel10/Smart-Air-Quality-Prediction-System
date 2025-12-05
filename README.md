# 🌍 Smart Air Quality Prediction System

A Python-based project using data analytics and machine learning to predict Air Quality Index (AQI) using real-like environmental parameters such as Temperature, Humidity, PM2.5, and PM10.

---

## 📌 Project Overview
This project simulates a real-world Air Quality Monitoring System by:
- Generating a realistic AQI dataset  
- Analyzing important environmental parameters  
- Cleaning & preprocessing data  
- Applying a Machine Learning model for prediction  
- Visualizing trends using graphs  
- Presenting results in a clear and meaningful way  

---

## 🛠️ Tools & Technologies Used
- **Python**
- **Pandas** – Data handling  
- **NumPy** – Calculations  
- **Matplotlib** – Visualizations  
- **Scikit-Learn** – Machine learning (Linear Regression)
- **CSV Dataset** – Realistic environment data  

---

## 📊 Dataset Information
The dataset contains the following columns:

| Feature | Description |
|--------|-------------|
| Temperature | Ambient temperature (°C) |
| Humidity | Relative humidity (%) |
| PM2.5 | Fine particulate matter |
| PM10 | Coarse particulate matter |
| AQI | Air Quality Index |

The final cleaned dataset used for training is stored as:  
**`real_aqi_dataset_clean.csv`**

---

## 🔍 Prediction Model
A **Linear Regression** model is used to predict AQI based on:
- Temperature  
- Humidity  
- PM2.5  
- PM10  

It provides:
- AQI prediction  
- Trend analysis  
- Comparison graphs  

---

## 📈 Visualizations Included
- AQI Trend Over Time  
- Temperature vs AQI  
- Humidity vs AQI  
- PM2.5 vs AQI  
- PM10 vs AQI  

All graphs are saved as `.png` files in the repository.

---

## 📂 Project Files
- `generate_real_aqi_dataset.py` – Creates realistic dataset  
- `real_aqi_analysis.py` – Main analysis + ML prediction  
- `real_aqi_dataset_clean.csv` – Final dataset  
- Graphs (`*.png`)  
- `Smart_air_quality_prediction_project.pptx` – Final project presentation  

---

## 🚀 How to Run the Project
1. Install required Python libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn

