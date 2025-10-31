"""
Simple data downloader for initial setup
"""

import pandas as pd
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import os

API_KEY = "evKEUv2Kzywwm2dk6uv1eaS0gnChH0mT"
TICKERS = [
    "RGTI", "IREN", "QBTS", "IONQ", "APLD", "VRT", "TMC", "IDR", "MOD", "NPCE",
    "F", "GM", "T", "VZ", "WMT", "KO", "PEP", "PG", "JNJ", "HD",
    "XOM", "CVX", "JPM", "INTC", "CSCO",
    "SMCI", "NVDA", "ARM", "CELH", "PLTR", "TSLA", "ELF", "MSTR", "MARA", "RIOT",
    "NVCR", "AEHR", "SOFI", "CRWD", "SNOW", "CAT", "DE", "BA", "NEE", "ENPH",
    "FSLR", "LLY", "ABBV", "COST", "MSFT", "AMZN", "META", "GOOG", "AAPL", "AVGO",
    "GIS", "KHC", "CAG", "TAP", "K", "DG", "KR", "LOW", "PFE", "MRK",
    "NKLA", "LCID", "HOOD", "SNAP", "BYND", "APRN", "PTON", "BBBYQ", "CVNA"
]

def fetch_and_save_ticker(ticker, start_date="2022-01-01", end_date="2025-10-01"):
    """Pull daily OHLCV for one ticker and save to data/daily/{ticker}.parquet"""
    print(f"Fetching data for {ticker}...")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}
    
    try:
        r = requests.get(url, params=params)
        data = r.json()
        
        if data.get("status") != "OK":
            print(f"⚠ API error for {ticker}: {data.get('message', 'Unknown error')}")
            return False
        
        results = data.get("results", [])
        if not results:
            print(f"⚠ No data for {ticker}")
            return False
        
        df = pd.DataFrame(results)
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low", 
            "c": "close", "v": "volume"
        })
        
        # Save to parquet
        output_path = f"data/daily/{ticker}.parquet"
        pq.write_table(
            pa.Table.from_pandas(df[["ticker", "date", "open", "high", "low", "close", "volume"]]),
            output_path
        )
        
        print(f"✓ Saved {ticker}: {len(df)} records")
        return True
        
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Downloading stock data...")
    print("=" * 50)
    
    # Ensure directories exist
    os.makedirs("data/daily", exist_ok=True)
    
    success_count = 0
    for ticker in TICKERS:
        if fetch_and_save_ticker(ticker):
            success_count += 1
    
    print(f"\n📊 Download Summary:")
    print(f"   Successful: {success_count}/{len(TICKERS)}")
    print(f"   Failed: {len(TICKERS) - success_count}/{len(TICKERS)}")
