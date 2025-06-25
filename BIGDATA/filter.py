import pandas as pd

# Buka file CSV
df = pd.read_csv('test2/cleaned_currency_rates.csv', parse_dates=['timestamp'])

# Filter hanya data sebelum 1 Mei 2025
filtered_df = df[df['timestamp'] < '2025-05-01']

# Simpan ke file baru
filtered_df.to_csv('currency_rates_before_2025_05_01.csv', index=False)
