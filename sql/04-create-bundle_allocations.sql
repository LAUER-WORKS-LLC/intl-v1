-- Hard-set allocations within each bundle
CREATE TABLE IF NOT EXISTS bundle_allocations (
    allocation_id SERIAL PRIMARY KEY,
    bundle_id INTEGER NOT NULL REFERENCES bundles(bundle_id),
    ticker VARCHAR(10) NOT NULL,
    allocation_pct DECIMAL(5,2) NOT NULL CHECK (allocation_pct >= 0 AND allocation_pct <= 100),
    effective_date DATE DEFAULT CURRENT_DATE,
    version INTEGER DEFAULT 1,
    UNIQUE(bundle_id, ticker, version)
);
