-- Sector-level performance aggregation
CREATE TABLE IF NOT EXISTS sector_performance (
    sector VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    avg_return DECIMAL(8,6),
    total_volume BIGINT,
    advancing_stocks INTEGER,
    declining_stocks INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (sector, date)
);
