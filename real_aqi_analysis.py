# real_aqi_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("real_aqi_dataset.csv")
print("Loaded dataset:\n", df.head())

# Basic stats
print("\nStatistics:")
print(df.describe())

# Save a cleaned copy (optional)
df.to_csv("real_aqi_dataset_clean.csv", index=False)

# Helper plotting function to keep style
def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename)
    print("Saved", filename)

# 1) AQI trend over days
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(df["Day"], df["AQI"], marker='o', linewidth=2)
ax.set_title("AQI Trend Over 30 Days")
ax.set_xlabel("Day")
ax.set_ylabel("AQI")
ax.grid(True)
save_plot(fig, "aqi_trend.png")
plt.close(fig)

# 2) Temperature vs AQI (scatter + fit line)
fig, ax = plt.subplots(figsize=(9,4))
ax.scatter(df["Temperature"], df["AQI"], s=60)
m, b = np.polyfit(df["Temperature"], df["AQI"], 1)
x_line = np.array([df["Temperature"].min(), df["Temperature"].max()])
ax.plot(x_line, m*x_line + b, color='orange', linestyle='--', linewidth=1.8)
ax.set_title("Temperature vs AQI")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("AQI")
ax.grid(True)
save_plot(fig, "temp_vs_aqi.png")
plt.close(fig)

# 3) Humidity vs AQI
fig, ax = plt.subplots(figsize=(9,4))
ax.scatter(df["Humidity"], df["AQI"], s=60)
m2, b2 = np.polyfit(df["Humidity"], df["AQI"], 1)
x_line2 = np.array([df["Humidity"].min(), df["Humidity"].max()])
ax.plot(x_line2, m2*x_line2 + b2, color='orange', linestyle='--', linewidth=1.8)
ax.set_title("Humidity vs AQI")
ax.set_xlabel("Humidity (%)")
ax.set_ylabel("AQI")
ax.grid(True)
save_plot(fig, "humidity_vs_aqi.png")
plt.close(fig)

# 4) PM2.5 vs AQI
fig, ax = plt.subplots(figsize=(9,4))
ax.scatter(df["PM2.5"], df["AQI"], s=60)
m3, b3 = np.polyfit(df["PM2.5"], df["AQI"], 1)
x_line3 = np.array([df["PM2.5"].min(), df["PM2.5"].max()])
ax.plot(x_line3, m3*x_line3 + b3, color='orange', linestyle='--', linewidth=1.8)
ax.set_title("PM2.5 vs AQI")
ax.set_xlabel("PM2.5 (µg/m³)")
ax.set_ylabel("AQI")
ax.grid(True)
save_plot(fig, "pm25_vs_aqi.png")
plt.close(fig)

# 5) AQI histogram
fig, ax = plt.subplots(figsize=(9,4))
ax.hist(df["AQI"], bins=8)
ax.set_title("AQI Distribution")
ax.set_xlabel("AQI")
ax.set_ylabel("Frequency")
ax.grid(True)
save_plot(fig, "aqi_histogram.png")
plt.close(fig)

# 6) Category counts
cat_counts = df["Category"].value_counts()
fig, ax = plt.subplots(figsize=(9,4))
cat_counts.plot(kind='bar', ax=ax)
ax.set_title("AQI Category Counts")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
save_plot(fig, "aqi_category_counts.png")
plt.close(fig)

# 7) Simple ML model using multiple features
X = df[["Temperature", "Humidity", "PM2.5", "PM10"]]
y = df["AQI"]
model = LinearRegression()
model.fit(X, y)
print("Model trained. Coefficients:", model.coef_, "Intercept:", model.intercept_)

# Example predict and save
sample = np.array([[31, 55, 70, 120]])  # example inputs: Temp, Humidity, PM2.5, PM10
pred = model.predict(sample)[0]
pd.DataFrame([{"Temperature": sample[0,0], "Humidity": sample[0,1], "PM2.5": sample[0,2], "PM10": sample[0,3], "Predicted_AQI": round(pred,2)}]).to_csv("prediction_output_real.csv", index=False)
print("Saved prediction_output_real.csv with predicted AQI:", round(pred,2))
