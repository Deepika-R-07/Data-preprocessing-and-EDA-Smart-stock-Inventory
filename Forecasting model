import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle, os, warnings, joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

df = pd.read_csv("processed dataset.csv")

date_col = None
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        date_col = col
        break

if date_col is None:
    raise KeyError("❌ No date column found in processed dataset.csv")


df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")


df = df.dropna(subset=[date_col])


os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)



def train_lstm(series, n_lags=7, epochs=10):
    """Train a simple LSTM model on a univariate time series."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled) - n_lags):
        X.append(scaled[i:i+n_lags, 0])
        y.append(scaled[i+n_lags, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)

    return model, scaler


def forecast_lstm(model, scaler, series, steps=30, n_lags=7):
   
    data = scaler.transform(series.values.reshape(-1, 1)).flatten().tolist()
    preds = []
    for _ in range(steps):
        x_input = np.array(data[-n_lags:]).reshape((1, n_lags, 1))
        yhat = model.predict(x_input, verbose=0)
        data.append(yhat[0][0])
        preds.append(yhat[0][0])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds



forecast_list = []
all_products = df["Product ID"].unique()

for product_name in all_products:
    print(f"\n🔄 Training Prophet & LSTM for {product_name}...")

    product_df = df[df['Product ID'] == product_name][[date_col, 'Sales']]

   
    prophet_df = product_df.rename(columns={date_col: 'ds', 'Sales': 'y'})
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model_prophet.fit(prophet_df)

    future = model_prophet.make_future_dataframe(periods=30)
    forecast_p = model_prophet.predict(future)
    yhat_prophet = forecast_p['yhat'][-30:]

    sales_series = product_df.set_index(date_col)['Sales']
    train_size = int(len(sales_series) * 0.8)
    train_series = sales_series.iloc[:train_size]

    lstm_model, scaler = train_lstm(train_series)
    yhat_lstm = forecast_lstm(lstm_model, scaler, sales_series, steps=30)

   
    actual = sales_series[-30:] if len(sales_series) >= 30 else sales_series

    mae_prophet = mean_absolute_error(actual, yhat_prophet[:len(actual)])
    rmse_prophet = np.sqrt(mean_squared_error(actual, yhat_prophet[:len(actual)]))

    mae_lstm = mean_absolute_error(actual, yhat_lstm[:len(actual)])
    rmse_lstm = np.sqrt(mean_squared_error(actual, yhat_lstm[:len(actual)]))

    print(f"Prophet  → MAE: {mae_prophet:.2f} | RMSE: {rmse_prophet:.2f}")
    print(f"LSTM     → MAE: {mae_lstm:.2f} | RMSE: {rmse_lstm:.2f}")

    
    if rmse_lstm < rmse_prophet:
        print("✅ LSTM better — saving LSTM model.")
        best_forecast = yhat_lstm

        
        lstm_model.save(f"models/lstm_model_{product_name}.keras")
        joblib.dump(scaler, f"models/lstm_scaler_{product_name}.pkl")

    else:
        print("✅ Prophet better — saving Prophet model.")
        best_forecast = yhat_prophet
        with open(f"models/prophet_model_{product_name}.pkl", "wb") as f:
            pickle.dump(model_prophet, f)

  
    forecast_dates = pd.date_range(
        start=product_df[date_col].max() + pd.Timedelta(days=1),
        periods=30
    )

    temp = pd.DataFrame({
        'date': forecast_dates,
        'forecast_best': best_forecast,
        'Product ID': product_name
    })
    forecast_list.append(temp)

forecast_all = pd.concat(forecast_list)
forecast_all.to_csv("data/forecast_results.csv", index=False)
print("\n✅ Forecast results saved in data/forecast_results.csv")


try:
    sample_product = all_products[0]  # pick first product
    actual_df = df[df["Product ID"] == sample_product]
    forecast_df = forecast_all[forecast_all["Product ID"] == sample_product]

    plt.figure(figsize=(10, 5))
    plt.plot(actual_df[date_col], actual_df["Sales"], label="Actual Sales")
    plt.plot(forecast_df["date"], forecast_df["forecast_best"],
             label="Forecast (Next 30 Days)", linestyle="--")
    plt.title(f"Sales Forecast for {sample_product}")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("⚠️ Could not plot sample forecast:", e)
