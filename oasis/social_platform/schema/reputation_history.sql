CREATE TABLE reputation_history (
    run_id INTEGER,
    seed INTEGER,
    round INTEGER,
    seller_id INTEGER,
    public_reputation_score INTEGER,
    public_num_ratings INTEGER,
    FOREIGN KEY(seller_id) REFERENCES user(user_id)
);

CREATE INDEX IF NOT EXISTS idx_reputation_history_seller_round
ON reputation_history (seller_id, round);
