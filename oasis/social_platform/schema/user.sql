-- This is the schema definition for the user table
CREATE TABLE user (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id INTEGER,
    user_name TEXT,
    name TEXT,
    bio TEXT,
    created_at DATETIME,
    num_followings INTEGER DEFAULT 0,
    num_followers INTEGER DEFAULT 0,
    -- Extensions based on Design Proposal
    role VARCHAR(10),
    reputation_score INTEGER DEFAULT 0,
    profit_utility_score DECIMAL(10, 2) DEFAULT 0.0,
    market_id VARCHAR(10)
);