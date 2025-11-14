"""
Stock Lists for Machine Learning Configuration and Testing
50 NASDAQ stocks for training/configuration, 50 different stocks for testing
"""

# 50 NASDAQ stocks for training/configuration (finding optimal weights and best model)
TRAINING_STOCKS = [
    # Tech Giants (8)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    # Semiconductors (8)
    "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "LRCX", "AMAT",
    # Software/Cloud (8)
    "CRM", "ADBE", "ORCL", "NOW", "SNOW", "DDOG", "ZS", "CRWD",
    # E-commerce/Retail (5)
    "COST", "HD", "LOW", "TGT", "EBAY",
    # Biotech/Pharma (5)
    "GILD", "AMGN", "REGN", "VRTX", "BIIB",
    # Finance (5)
    "PYPL", "SQ", "HOOD", "SOFI", "AFRM",
    # Other Tech (5)
    "UBER", "LYFT", "ZM", "DOCN", "NET",
    # Consumer (4)
    "SBUX", "NKE", "MCD", "CMG",
    # Healthcare Tech (4)
    "ISRG", "DXCM", "ALGN", "HOLX",
    # Other (3)
    "ANSS", "CDNS", "SNPS"
]
# Total: 8+8+8+5+5+5+5+4+4+3 = 55, need to remove 5

# Remove 5 to get exactly 50
TRAINING_STOCKS = [s for s in TRAINING_STOCKS if s not in ["DOCN", "NET", "ISRG", "DXCM", "ALGN"]]

# 50 Different NASDAQ stocks for testing (validating the optimal configuration)
TESTING_STOCKS = [
    # Tech (8)
    "GOOG", "MCHP", "MRVL", "SWKS", "QRVO", "ON", "WOLF", "ALGM",
    # Software (8)
    "TEAM", "ESTC", "MDB", "SPLK", "OKTA", "FTNT", "PANW", "VRNS",
    # E-commerce/Retail (5)
    "ETSY", "ROST", "TJX", "BBY", "FIVE",
    # Biotech/Pharma (5)
    "ILMN", "ALNY", "MRNA", "BNTX", "SGMO",
    # Finance (5)
    "LC", "UPST", "NU", "NUAN", "FITB",
    # Consumer (5)
    "YUM", "DPZ", "LULU", "WING", "SHAK",
    # Healthcare (5)
    "TECH", "TMO", "A", "BIO", "ALKS",
    # Industrial/Tech Services (5)
    "FAST", "NDAQ", "FISV", "FIS", "FDS",
    # Communication (4)
    "TMUS", "CHTR", "CMCSA", "LBRDK",
    # Other Tech (0 - will add to reach 50)
]

# Remove any overlaps
overlap = set(TRAINING_STOCKS) & set(TESTING_STOCKS)
TESTING_STOCKS = [s for s in TESTING_STOCKS if s not in overlap]

# Add stocks to reach exactly 50
additional_testing = [
    "PTON", "RBLX", "ROKU", "SPOT", "TWLO", "VEEV", "WDAY", "ZEN",
    "APPN", "BILL", "COUP", "DOCU", "FROG", "GTLB", "HUBS", "MNDY",
    "PLTR", "CLVT", "COIN", "DASH", "GRAB", "OPEN", "PINS", "SNAP",
    "TTD", "RPD", "ASAN", "INTU", "ADSK", "HOLX", "DOCN", "NET",
    "ISRG", "DXCM", "ALGN", "MELI", "SE", "SHOP", "LBRDA", "RPD"
]

while len(TESTING_STOCKS) < 50 and additional_testing:
    stock = additional_testing.pop(0)
    if stock not in TRAINING_STOCKS and stock not in TESTING_STOCKS:
        TESTING_STOCKS.append(stock)

# Trim to exactly 50
TESTING_STOCKS = TESTING_STOCKS[:50]

# Final verification
assert len(TRAINING_STOCKS) == 50, f"Training stocks should be 50, got {len(TRAINING_STOCKS)}"
assert len(TESTING_STOCKS) == 50, f"Testing stocks should be 50, got {len(TESTING_STOCKS)}"

final_overlap = set(TRAINING_STOCKS) & set(TESTING_STOCKS)
assert len(final_overlap) == 0, f"Found overlap between training and testing stocks: {final_overlap}"

print(f"Training stocks: {len(TRAINING_STOCKS)}")
print(f"Testing stocks: {len(TESTING_STOCKS)}")
print(f"Overlap: {len(final_overlap)}")
