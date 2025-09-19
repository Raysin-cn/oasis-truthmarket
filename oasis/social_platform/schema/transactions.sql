-- This is the schema definition for the new transactions table
CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER,
    seller_id INTEGER,
    buyer_id INTEGER,
    round_number INTEGER,
    rating INTEGER,
    is_challenged BOOLEAN,
    -- 收益计算结果
    seller_profit REAL,
    buyer_utility REAL,
    -- 挑战相关收益
    challenge_cost REAL,
    challenge_reward REAL,
    challenge_penalty REAL,
    -- 时间戳
    created_at INTEGER,
    FOREIGN KEY(post_id) REFERENCES post(post_id),
    FOREIGN KEY(seller_id) REFERENCES user(user_id),
    FOREIGN KEY(buyer_id) REFERENCES user(user_id)
);