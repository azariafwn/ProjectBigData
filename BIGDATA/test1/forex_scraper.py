import requests
import pandas as pd

# Ganti dengan API Key kamu
API_KEY = "f31d1578723d436b83902e9bc2fa3f01"
SYMBOLS_URL = f"https://api.forexrateapi.com/v1/symbols?api_key={API_KEY}"
LATEST_URL = "https://api.forexrateapi.com/v1/latest"

# Ambil daftar semua mata uang
response_symbols = requests.get(SYMBOLS_URL)
symbols_data = response_symbols.json()

if "symbols" in symbols_data:
    all_currencies = list(symbols_data["symbols"].keys())  # List semua currency
else:
    print("Gagal mengambil daftar currency!")
    exit()

# List untuk menyimpan semua hasil
all_data = []

# Loop setiap currency sebagai base
for base_currency in all_currencies:
    url = f"{LATEST_URL}?api_key={API_KEY}&base={base_currency}"
    response = requests.get(url)
    data = response.json()

    if "rates" in data:
        for currency, rate in data["rates"].items():
            all_data.append([base_currency, currency, rate])

# Simpan ke CSV untuk Tableau
df = pd.DataFrame(all_data, columns=["Base Currency", "Target Currency", "Exchange Rate"])
df.to_csv("forex_rates_all_base.csv", index=False)
print("Data berhasil disimpan ke forex_rates_all_base.csv")
