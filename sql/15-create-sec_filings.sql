-- SEC filing tracking
CREATE TABLE IF NOT EXISTS sec_filings (
    filing_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL REFERENCES stocks(ticker),
    filing_type VARCHAR(10) NOT NULL CHECK (filing_type IN ('10-K', '10-Q', '8-K', 'Press Release')),
    filing_date DATE NOT NULL,
    url TEXT,
    summary TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
