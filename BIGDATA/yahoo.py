from yahooquery import Ticker
import pandas as pd
from datetime import datetime

# Daftar pasangan mata uang
currency_pairs = [
    "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", 
    "EURJPY=X", "GBPJPY=X", "EURGBP=X", "EURCAD=X", "EURSEK=X",
    "EURCHF=X", "EURHUF=X", "CNY=X", "HKD=X", "SGD=X", "INR=X",
    "MXN=X", "PHP=X", "IDR=X", "THB=X", "MYR=X", "ZAR=X", "RUB=X"
]

# Ambil data dari Yahoo Finance
tickers = Ticker(currency_pairs)
prices = tickers.price

# Proses data
currency_data = []
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Waktu scraping

for pair in currency_pairs:
    if pair in prices:
        price = prices[pair].get("regularMarketPrice", "N/A")
        change = prices[pair].get("regularMarketChange", "N/A")
        change_percent = prices[pair].get("regularMarketChangePercent", "N/A")
        currency_data.append([timestamp, pair, price, change, change_percent])

# Konversi ke DataFrame
df = pd.DataFrame(currency_data, columns=["timestamp", "currency_pair", "price", "change_value", "change_percent"])


# Simpan ke CSV (append tanpa header jika sudah ada file)
df.to_csv("C:/zafaa/kuliah/SEMESTER6/BIGDATA/test2/currency_rates.csv", mode='a', header=not pd.io.common.file_exists("C:/zafaa/kuliah/SEMESTER6/BIGDATA/test2/currency_rates.csv"), index=False)

print("Scraping selesai! Data berhasil ditambahkan ke 'currency_rates.csv'.")
