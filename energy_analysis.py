import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("household_power.csv", sep=";")
print(df.columns)

print("Dataset Preview")
print(df.head())

# Clean data
df.dropna(inplace=True)

df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)

df.dropna(inplace=True)

# Graph 1-Energy Consumtion Trend

plt.figure(figsize=(10,5))
df["Global_active_power"].head(500).plot()

plt.title("Energy Consumption Trend")
plt.xlabel("Records")
plt.ylabel("Power")

plt.show()

# Graph 2-Avarage Energy Consumption by Hour
 
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

hourly_avg = df.groupby("Hour")["Global_active_power"].mean()

plt.figure(figsize=(10,5))

hourly_avg.plot(kind="bar")

plt.title("Average Energy Consumption by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Power")
plt.show()


# Graph 3-Power Consumption Distribution(Histogram)

plt.figure(figsize=(8,5))

df["Global_active_power"].hist(bins=30)

plt.title("Distribution of Power Consumption")
plt.xlabel("Power Consumption")
plt.ylabel("Frequency")

plt.show()



# Graph 4-Monthly Energy Consumtion(Pie Chart)

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

df["Month"] = df["Date"].dt.month_name()

monthly = df.groupby("Month")["Global_active_power"].sum()

plt.figure(figsize=(8,8))

monthly.head(6).plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Monthly Energy Consumption Share")
plt.ylabel("")

plt.show()


# Forecast
series = df["Global_active_power"].head(500)

model = ARIMA(series, order=(5,1,0))
result = model.fit()

forecast = result.forecast(steps=10)

print("\nForecast:")
print(forecast)

# Anomaly detection
detector = IsolationForest(
    contamination=0.02,
    random_state=42
)

df2 = df.head(500).copy()

df2["anomaly"] = detector.fit_predict(
    df2[["Global_active_power"]]
)


# Graph 5- Normal vs Anomalies

plt.figure(figsize=(10,5))

normal = df2[df2["anomaly"] == 1]
anomaly = df2[df2["anomaly"] == -1]

plt.scatter(normal.index,
            normal["Global_active_power"],
            label="Normal")

plt.scatter(anomaly.index,
            anomaly["Global_active_power"],
            label="Anomaly")

plt.legend()

plt.title("Anomaly Detection Result")

plt.show()

print("\nAnomalies:")
print(df2[df2["anomaly"] == -1])
