-- Complete portfolio change history
CREATE TABLE IF NOT EXISTS user_portfolio_history (
    history_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    transaction_type VARCHAR(20) NOT NULL CHECK (transaction_type IN ('deposit', 'reallocate', 'withdraw')),
    amount DECIMAL(15,2),
    date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
