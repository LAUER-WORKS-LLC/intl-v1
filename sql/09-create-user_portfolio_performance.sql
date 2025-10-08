-- User's total portfolio performance
CREATE TABLE IF NOT EXISTS user_portfolio_performance (
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    date DATE NOT NULL,
    total_nav DECIMAL(15,2) NOT NULL,
    daily_return DECIMAL(8,6),
    cumulative_return DECIMAL(8,6),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, date)
);
