import os
import sqlite3
from typing import Optional, Tuple

def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path)

def ensure_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    # reputation_history schema should already exist via SQL files,
    # but we defensively ensure it here to avoid runtime failures.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reputation_history (
            run_id INTEGER,
            seed INTEGER,
            round INTEGER,
            seller_id INTEGER,
            public_reputation_score INTEGER,
            public_num_ratings INTEGER,
            FOREIGN KEY(seller_id) REFERENCES user(user_id)
        );
        """
    )
    # Simple index for query speed
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reputation_history_seller_round
        ON reputation_history (seller_id, round);
        """
    )
    conn.commit()


def _get_previous_reputation(conn: sqlite3.Connection, seller_id: int) -> Tuple[int, int]:
    """Return (score, count) from latest history row if exists, else defaults (1, 0)."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT public_reputation_score, public_num_ratings
        FROM reputation_history
        WHERE seller_id = ?
        ORDER BY round DESC
        LIMIT 1
        """,
        (seller_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return 1, 0
    return int(row[0] or 1), int(row[1] or 0)


def _get_run_meta() -> Tuple[int, Optional[int]]:
    # In absence of a run manager, default run_id=1 and optional seed from env
    run_id = int(os.getenv("RUN_ID", "1"))
    seed_env = os.getenv("SEED")
    seed = int(seed_env) if seed_env is not None and seed_env.isdigit() else None
    return run_id, seed


def compute_and_update_reputation(conn: sqlite3.Connection, round_number: int, ratings_up_to_round: Optional[int] = None) -> None:
    """
    计算每个卖家的公共声誉：按评分的累计平均。
    考虑卖家的 enter_market 时间，只统计从该时间点开始的声誉累积。
    - round_number: 当前快照所属的回合（用于写入 reputation_history.round）
    - ratings_up_to_round: 评分聚合所考虑的最大回合（用于实现滞后显示）。
      若为 None，则等同于使用 round_number。
    """
    ensure_tables(conn)
    cursor = conn.cursor()

    effective_max_round = round_number if ratings_up_to_round is None else max(0, int(ratings_up_to_round))

    # 获取所有卖家及其 enter_market 时间
    cursor.execute("SELECT user_id, enter_market_round FROM user WHERE role = 'seller'")
    sellers_info = {row[0]: row[1] for row in cursor.fetchall()}

    run_id, seed = _get_run_meta()

    for seller_id, enter_market_time in sellers_info.items():
            # 只统计从 enter_market_time 开始的交易
        cursor.execute(
            """
            SELECT COUNT(t.rating) as cnt, COALESCE(SUM(t.rating), 0)
            FROM transactions t
            WHERE t.seller_id = ? AND t.rating IS NOT NULL 
            AND t.round_number <= ? AND t.round_number >= ?
            """,
            (seller_id, effective_max_round, enter_market_time),
        )
        
        result = cursor.fetchone()
        if result:
            num_ratings, sum_ratings = int(result[0]), int(result[1])
        else:
            num_ratings, sum_ratings = 0, 0

        if num_ratings > 0:
            avg_rating = round(sum_ratings)
        else:
            # Default initial reputation
            avg_rating = 0

        # Persist snapshot
        cursor.execute(
            """
            INSERT INTO reputation_history (
                run_id, seed, round, seller_id, public_reputation_score, public_num_ratings
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, seed, round_number, seller_id, int(avg_rating), int(num_ratings)),
        )

        # Update user table for live access in prompts
        cursor.execute(
            "UPDATE user SET reputation_score = ? WHERE user_id = ?",
            (int(avg_rating), seller_id),
        )

    conn.commit()


def backfill_reputation_for_all_rounds(max_round: int, db_path: Optional[str] = None) -> None:
    """Utility to recompute and snapshot reputation for rounds 1..max_round."""
    with _connect(db_path) as conn:
        for r in range(1, max_round + 1):
            compute_and_update_reputation(conn, r)
