CREATE TABLE analysis_labels (
    run_id INTEGER,
    seed INTEGER,
    seller_id INTEGER,
    label_manipulator BOOLEAN,
    label_type TEXT,
    first_cheat_round INTEGER,
    last_cheat_round INTEGER,
    cheat_rate_pre REAL,
    cheat_rate_post REAL,
    FOREIGN KEY(seller_id) REFERENCES user(user_id)
);

CREATE INDEX IF NOT EXISTS idx_analysis_labels_seller
ON analysis_labels (seller_id);
