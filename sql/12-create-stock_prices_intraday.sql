-- Intraday price data for stocks
CREATE TABLE IF NOT EXISTS stock_prices_intraday (
    ticker VARCHAR(10) NOT NULL REFERENCES stocks(ticker),
    datetime TIMESTAMPTZ NOT NULL,
    open DECIMAL(10,4),
    high DECIMAL(10,4),
    low DECIMAL(10,4),
    close DECIMAL(10,4),
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, datetime)
);
