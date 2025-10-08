-- Macro-economic indicators from FRED
CREATE TABLE IF NOT EXISTS macro_indicators (
    indicator_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    value DECIMAL(15,6),
    frequency VARCHAR(20) DEFAULT 'daily' CHECK (frequency IN ('daily', 'monthly', 'quarterly', 'annual')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (indicator_name, date)
);
