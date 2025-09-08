-- This is the schema definition for the post table
CREATE TABLE post (
    post_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    original_post_id INTEGER,
    content TEXT DEFAULT '',
    quote_content TEXT,
    created_at DATETIME,
    num_likes INTEGER DEFAULT 0,
    num_dislikes INTEGER DEFAULT 0,
    num_shares INTEGER DEFAULT 0,
    num_reports INTEGER DEFAULT 0,
    -- Extensions based on Design Proposal (as Product Listings)
    true_quality VARCHAR(10),
    advertised_quality VARCHAR(10),
    price DECIMAL(10, 2),
    cost DECIMAL(10, 2),
    has_warrant BOOLEAN,
    is_sold BOOLEAN,
    status VARCHAR(20),
    FOREIGN KEY(user_id) REFERENCES user(user_id),
    FOREIGN KEY(original_post_id) REFERENCES post(post_id)
);