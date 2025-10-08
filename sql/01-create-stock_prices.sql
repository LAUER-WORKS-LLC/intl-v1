CREATE TABLE IF NOT EXISTS stock_prices (
    ticker VARCHAR(10),
    timestamp TIMESTAMPTZ,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    PRIMARY KEY (ticker, timestamp)
);
