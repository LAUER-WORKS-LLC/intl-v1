"""
Stock Lists for Simulation Backtesting
50 NASDAQ stocks + 50 NYSE stocks
"""

# 50 NASDAQ stocks
NASDAQ_STOCKS = [
    # Tech Giants
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    # Semiconductors
    "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "LRCX", "AMAT",
    # Software/Cloud
    "CRM", "ADBE", "ORCL", "NOW", "SNOW", "DDOG", "ZS", "CRWD",
    # E-commerce/Retail
    "COST", "HD", "LOW", "TGT", "EBAY",
    # Biotech/Pharma
    "GILD", "AMGN", "REGN", "VRTX", "BIIB",
    # Finance
    "PYPL", "SQ", "HOOD", "SOFI", "AFRM",
    # Other Tech
    "UBER", "LYFT", "ZM", "SBUX", "NKE"
]

# 50 NYSE stocks
NYSE_STOCKS = [
    # Financial Services
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    # Consumer Goods
    "PG", "KO", "PEP", "WMT", "TGT", "HD",
    # Industrial
    "BA", "CAT", "GE", "HON", "MMM", "EMR", "ETN", "DE",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
    # Technology
    "IBM", "CSCO", "INTU", "ADP", "FIS", "FISV", "BR",
    # Consumer Services
    "MCD", "YUM", "CMG", "DPZ", "LULU", "TJX", "ROST",
    # Other
    "V", "MA", "DIS", "NFLX", "NKE", "SBUX"
]

# Remove duplicates and ensure exactly 50 each
NASDAQ_STOCKS = list(dict.fromkeys(NASDAQ_STOCKS))
NYSE_STOCKS = list(dict.fromkeys(NYSE_STOCKS))

# Remove any overlaps
overlap = set(NASDAQ_STOCKS) & set(NYSE_STOCKS)
NYSE_STOCKS = [s for s in NYSE_STOCKS if s not in overlap]

# Ensure exactly 50 each (pad if needed or trim)
if len(NASDAQ_STOCKS) < 50:
    # Add more NASDAQ stocks if needed
    additional_nasdaq = ["TEAM", "ESTC", "MDB", "SPLK", "OKTA", "FTNT", "PANW", "VRNS",
                         "ETSY", "ROST", "TJX", "BBY", "FIVE", "ILMN", "ALNY", "MRNA",
                         "BNTX", "LC", "UPST", "NU", "NUAN", "FITB", "FAST", "NDAQ",
                         "FDS", "PTON", "RBLX", "ROKU", "SPOT", "TWLO"]
    for stock in additional_nasdaq:
        if stock not in NASDAQ_STOCKS and len(NASDAQ_STOCKS) < 50:
            NASDAQ_STOCKS.append(stock)

if len(NYSE_STOCKS) < 50:
    # Add more NYSE stocks if needed
    additional_nyse = ["AXP", "COF", "USB", "PNC", "TFC", "KEY", "CFG", "HBAN",
                       "BK", "STT", "ZION", "MTB", "RF", "FITB", "CMA", "WTFC",
                       "WAL", "ONB", "FNB", "TCF", "UBSH", "HOMB", "UMBF", "BOKF",
                       "SNV", "CBSH", "FBNC", "FFIN", "HBNC", "UBSH"]
    for stock in additional_nyse:
        if stock not in NYSE_STOCKS and stock not in NASDAQ_STOCKS and len(NYSE_STOCKS) < 50:
            NYSE_STOCKS.append(stock)

NASDAQ_STOCKS = NASDAQ_STOCKS[:50]
NYSE_STOCKS = NYSE_STOCKS[:50]

# Combine all stocks
ALL_SIMULATION_STOCKS = NASDAQ_STOCKS + NYSE_STOCKS

print(f"NASDAQ stocks: {len(NASDAQ_STOCKS)}")
print(f"NYSE stocks: {len(NYSE_STOCKS)}")
print(f"Total stocks: {len(ALL_SIMULATION_STOCKS)}")

