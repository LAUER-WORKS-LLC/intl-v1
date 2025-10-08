-- FRED series information and metadata
CREATE TABLE IF NOT EXISTS macro_indicator_metadata (
    indicator_name VARCHAR(100) PRIMARY KEY,
    fred_series_id VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    unit VARCHAR(50),
    frequency VARCHAR(20) DEFAULT 'daily',
    last_updated TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
