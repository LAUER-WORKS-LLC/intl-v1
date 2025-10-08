-- Complete history of bundle allocation changes
CREATE TABLE IF NOT EXISTS bundle_allocation_history (
    history_id SERIAL PRIMARY KEY,
    bundle_id INTEGER NOT NULL REFERENCES bundles(bundle_id),
    old_allocation_id INTEGER REFERENCES bundle_allocations(allocation_id),
    new_allocation_id INTEGER REFERENCES bundle_allocations(allocation_id),
    change_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    reason TEXT
);
