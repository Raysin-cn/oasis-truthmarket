-- This is the schema definition for the product table
CREATE TABLE product (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    created_at INTEGER,
    -- Product quality and pricing
    true_quality VARCHAR(10),
    advertised_quality VARCHAR(10),
    price DECIMAL(10, 2),
    cost DECIMAL(10, 2),
    -- Warranty and sales status
    has_warrant BOOLEAN DEFAULT 0,
    is_sold BOOLEAN DEFAULT 0,
    -- Round tracking
    round_number INTEGER,
    -- Product status: 'on_sale', 'sold', 'challenged_success', 'challenged_fail'
    status VARCHAR(20) DEFAULT 'on_sale',
    FOREIGN KEY(user_id) REFERENCES user(user_id)
);
