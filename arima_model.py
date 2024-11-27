import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv('sensor_data.csv',index_col=0)
data.rename(columns={'0':'Sensor Readings'},inplace=True)


plt.figure(figsize=(8,4))
plt.plot(data.index,data['Sensor Readings'], label='Daily Sensor Data')
plt.xlabel('Time Period')
plt.ylabel('Sensor Value')
plt.title('Daily Sensor Data Readings')
plt.legend()
plt.show()

train_size=int(len(data)*0.8)
train_data,test_data=data[:train_size],data[train_size:]

model=ARIMA(train_data,order=(0,2,2))
model_fit=model.fit()

forecast = model_fit.forecast(steps=len(test_data))

plt.figure(figsize=(12,6))
plt.plot(train_data.index,train_data,label='Train Data')
plt.plot(test_data.index,test_data,label='Test Data')
plt.plot(test_data.index,forecast,label='ARIMA Forecast')
plt.title('ARIMA Model Forecast')
plt.xlabel('Time')
plt.ylabel('Sensor Value')
plt.legend()

best_aic = float("inf")
best_order = None

# Grid search
for p in range(6):
    for d in range(3):
        for q in range(6):
            try:
                model = ARIMA(data, order=(p, d, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except:
                continue

print(f"Best ARIMA order: {best_order} with AIC: {best_aic}")
