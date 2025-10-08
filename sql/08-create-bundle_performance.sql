-- Calculated bundle NAV over time
CREATE TABLE IF NOT EXISTS bundle_performance (
    bundle_id INTEGER NOT NULL REFERENCES bundles(bundle_id),
    date DATE NOT NULL,
    nav_value DECIMAL(15,4) NOT NULL,
    daily_return DECIMAL(8,6),
    cumulative_return DECIMAL(8,6),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (bundle_id, date)
);
