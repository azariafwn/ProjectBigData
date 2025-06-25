import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta

df = pd.read_csv('test2/currency_rates.csv', parse_dates=['timestamp'])

def predict_currency(currency_pair='USDJPY', horizon_days=7, look_back=60):
    data = df[df['currency_pair'] == currency_pair].copy()
    data = data.sort_values('timestamp')
    prices = data[['price']].values

    # Optional: cek apakah log transform membantu (bisa aktifkan)
    # prices = np.log1p(prices)  # Uncomment jika ingin coba

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(look_back, len(scaled_prices) - horizon_days):
        X.append(scaled_prices[i - look_back:i, 0])
        y.append(scaled_prices[i + horizon_days, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        shuffle=False,
        verbose=1
    )
    print("Training selesai.")

    # Plot learning curve
    plt.figure(figsize=(10,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

    # Prediksi data test
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Baseline naive forecast (last observed price)
    naive_pred = scaler.inverse_transform(X_test[:, -1, :])
    naive_mse = mean_squared_error(y_true, naive_pred)
    naive_r2 = r2_score(y_true, naive_pred)

    print(f"Model MSE = {mse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}")
    print(f"Naive baseline MSE = {naive_mse:.4f}, R2 = {naive_r2:.4f}")

    # Plot actual vs prediksi model dan baseline
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'].iloc[split + look_back + horizon_days:], y_true, label='Actual Price (Test)')
    plt.plot(data['timestamp'].iloc[split + look_back + horizon_days:], y_pred, label='Predicted Price (Model)')
    plt.plot(data['timestamp'].iloc[split + look_back + horizon_days:], naive_pred, label='Naive Forecast (Last Obs)')
    plt.legend()
    plt.title(f'{currency_pair} - Evaluasi Model (Test Set)')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.show()

    # Recursive forecast prediksi horizon_days ke depan
    last_seq = scaled_prices[-look_back:].reshape(1, look_back, 1)
    future_preds_scaled = []

    for _ in range(horizon_days):
        pred_scaled = model.predict(last_seq)[0, 0]
        future_preds_scaled.append(pred_scaled)
        last_seq = np.append(last_seq[:, 1:, :], [[[pred_scaled]]], axis=1)

    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamp'], prices, label='Historical Prices')
    future_dates = [data['timestamp'].max() + timedelta(days=i + 1) for i in range(horizon_days)]
    plt.plot(future_dates, future_preds, label=f'Predicted Next {horizon_days} Days', linestyle='--')
    plt.legend()
    plt.title(f'{currency_pair} - Prediksi {horizon_days} Hari ke Depan')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.show()

    print(f"\n📈 Prediksi harga {currency_pair} pada {future_dates[-1].date()} adalah: {future_preds[-1][0]:.3f}")

print("Currency pair yang tersedia:", df['currency_pair'].unique())
pair = input("Masukkan currency pair (contoh: USDJPY): ").upper()
days = int(input("Masukkan jumlah hari ke depan untuk prediksi (contoh: 7): "))  # default 7 hari untuk coba dulu
predict_currency(pair, days)
