-- Pre-calculated performance data for stocks
CREATE TABLE IF NOT EXISTS stock_performance_metrics (
    ticker VARCHAR(10) NOT NULL REFERENCES stocks(ticker),
    date DATE NOT NULL,
    daily_return DECIMAL(8,6),
    volatility_30d DECIMAL(8,6),
    beta DECIMAL(8,6),
    rsi DECIMAL(5,2),
    moving_avg_50d DECIMAL(10,4),
    moving_avg_200d DECIMAL(10,4),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date)
);
