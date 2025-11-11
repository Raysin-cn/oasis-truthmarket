-- This is the schema definition for the new transactions table
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER,
    seller_id INTEGER,
    buyer_id INTEGER,
    round_number INTEGER,
    rating INTEGER,
    is_challenged BOOLEAN,
    -- Profit calculation results
    seller_profit REAL,
    buyer_utility REAL,
    -- Challenge related profits
    challenge_cost REAL,
    challenge_reward REAL,
    challenge_penalty REAL,
    -- Timestamp
    created_at INTEGER,
    FOREIGN KEY(post_id) REFERENCES post(post_id),
    FOREIGN KEY(seller_id) REFERENCES user(user_id),
    FOREIGN KEY(buyer_id) REFERENCES user(user_id)
);