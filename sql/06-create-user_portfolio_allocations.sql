-- User's chosen bundle allocations
CREATE TABLE IF NOT EXISTS user_portfolio_allocations (
    allocation_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    bundle_id INTEGER NOT NULL REFERENCES bundles(bundle_id),
    allocation_pct DECIMAL(5,2) NOT NULL CHECK (allocation_pct >= 0 AND allocation_pct <= 100),
    effective_date DATE DEFAULT CURRENT_DATE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'pending')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
