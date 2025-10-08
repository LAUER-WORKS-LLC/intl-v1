-- Company event calendar
CREATE TABLE IF NOT EXISTS corporate_events (
    event_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL REFERENCES stocks(ticker),
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN ('earnings', 'dividend', 'split', 'merger', 'guidance', 'filing')),
    event_date DATE NOT NULL,
    description TEXT,
    impact_level VARCHAR(20) DEFAULT 'medium' CHECK (impact_level IN ('low', 'medium', 'high')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
