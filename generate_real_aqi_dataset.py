 # generate_real_aqi_dataset.py
import pandas as pd
import numpy as np

np.random.seed(42)

days = np.arange(1, 31)
temperature = np.random.randint(22, 39, size=30)   # 22–38 °C
humidity = np.random.randint(35, 85, size=30)      # 35–85 %
pm25 = np.random.randint(20, 180, size=30)         # realistic PM2.5 range
pm10 = np.random.randint(30, 250, size=30)         # realistic PM10 range

# Create AQI proxy (simple weighted combination — fine for academic demo)
aqi = np.round(0.6 * pm25 + 0.4 * pm10).astype(int)

df = pd.DataFrame({
    "Day": days,
    "Temperature": temperature,
    "Humidity": humidity,
    "PM2.5": pm25,
    "PM10": pm10,
    "AQI": aqi
})

# Optional category helper
def aqi_category(v):
    if v <= 50: return "Good"
    if v <= 100: return "Moderate"
    if v <= 150: return "Unhealthy for Sensitive Groups"
    if v <= 200: return "Unhealthy"
    if v <= 300: return "Very Unhealthy"
    return "Hazardous"

df["Category"] = df["AQI"].apply(aqi_category)

csv_path = "real_aqi_dataset.csv"
df.to_csv(csv_path, index=False)
print("Saved dataset to", csv_path)
print(df.head())
