import os
import sqlite3
from typing import Optional, Tuple

DATABASE_PATH = os.getenv("MARKET_DB_PATH", "market_sim.db")

def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or DATABASE_PATH)


def ensure_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_labels (
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
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analysis_labels_seller
        ON analysis_labels (seller_id);
        """
    )
    conn.commit()


def _get_run_meta() -> Tuple[int, Optional[int]]:
    run_id = int(os.getenv("RUN_ID", "1"))
    seed_env = os.getenv("SEED")
    seed = int(seed_env) if seed_env is not None and seed_env.isdigit() else None
    return run_id, seed


DEFAULT_FLAGS = {
    "enable_deception": True,              # LQ truth + HQ advertised
    "enable_price_gouging": True,          # price jumps while quality non-improving
    "enable_repeat_deception": True,       # consecutive deception rounds >= threshold
    "enable_warrant_masking": True,        # HQ advertised with warrant while truth LQ
}

DEFAULT_THRESHOLDS = {
    "deception_increase_delta": 0.4,       # post - pre deception rate
    "rating_drop_delta": 0.5,              # pre - post avg rating
    "price_jump_ratio": 1.5,               # price_post_avg / price_pre_avg
    "repeat_deception_min_streak": 2,      # consecutive deception rounds
    "warrant_masking_min_rate": 0.5,       # rate of deception when has_warrant
}


def _compute_round_series(cursor: sqlite3.Cursor, seller_id: int):
    cursor.execute(
        """
        SELECT p.round_number, p.advertised_quality, p.true_quality, p.price, p.has_warrant
        FROM post p
        WHERE p.user_id = ? AND p.round_number IS NOT NULL
        ORDER BY p.round_number ASC
        """,
        (seller_id,),
    )
    rows = cursor.fetchall()
    rounds = []
    adv = {}
    truth = {}
    price = {}
    warrant = {}
    for rnum, a, t, pr, w in rows:
        r = int(rnum)
        rounds.append(r)
        adv[r] = a
        truth[r] = t
        price[r] = float(pr) if pr is not None else None
        warrant[r] = bool(w) if w is not None else False
    return rounds, adv, truth, price, warrant


def _compute_round_ratings(cursor: sqlite3.Cursor, seller_id: int):
    cursor.execute(
        """
        SELECT round_number, AVG(rating) as avg_rating
        FROM transactions
        WHERE seller_id = ? AND rating IS NOT NULL
        GROUP BY round_number
        """,
        (seller_id,),
    )
    return {int(r): float(avg) for r, avg in cursor.fetchall()}


def detect_seller_manipulation(conn: sqlite3.Connection, max_round: int) -> None:
    """
    简单“漏洞/操纵”检测：
    - 基于广告质量与真实质量的偏差，若在若干回合后出现持续偏差升高且评级下降，标记为操纵者。
    - 统计操纵前后（分段）平均偏差与平均评分，生成分析标签。
    这是一个基础的启发式规则，便于后续替换为更复杂方法。
    """
    ensure_tables(conn)
    cursor = conn.cursor()

    # 收集卖家列表
    cursor.execute("SELECT user_id FROM user WHERE role = 'seller'")
    seller_ids = [r[0] for r in cursor.fetchall()]

    run_id, seed = _get_run_meta()

    for seller_id in seller_ids:
        # 每回合计算广告-真实偏差和本回合评分
        cursor.execute(
            """
            SELECT p.round_number, p.advertised_quality, p.true_quality
            FROM post p
            WHERE p.user_id = ? AND p.round_number IS NOT NULL
            ORDER BY p.round_number ASC
            """,
            (seller_id,),
        )
        posts = cursor.fetchall()

        # 连接当回合的交易评分（若存在）
        round_to_rating = {}
        cursor.execute(
            """
            SELECT round_number, AVG(rating) as avg_rating
            FROM transactions
            WHERE seller_id = ? AND rating IS NOT NULL
            GROUP BY round_number
            """,
            (seller_id,),
        )
        for rr, avg_rating in cursor.fetchall():
            round_to_rating[int(rr)] = float(avg_rating)

        # 计算欺骗：true_quality == 'LQ' 且 advertised_quality == 'HQ' 视为欺骗=1，否则=0
        round_to_gap = {}
        rounds = []
        for rnum, adv_q, true_q in posts:
            rounds.append(int(rnum))
            deception = 1.0 if (
                adv_q is not None and true_q is not None and str(true_q) == 'LQ' and str(adv_q) == 'HQ'
            ) else 0.0
            round_to_gap[int(rnum)] = deception

        if not rounds:
            continue

        first_half = [r for r in rounds if r <= max_round // 2]
        second_half = [r for r in rounds if r > max_round // 2]

        cheat_rate_pre = avg([round_to_gap[r] for r in first_half])
        cheat_rate_post = avg([round_to_gap[r] for r in second_half])

        # 如果后半段偏差显著增大，且后半段评分下降，则标记
        avg_rating_pre = avg([round_to_rating.get(r, 0.0) for r in first_half])
        avg_rating_post = avg([round_to_rating.get(r, 0.0) for r in second_half])

        manipulation = (cheat_rate_post - cheat_rate_pre >= 0.4) and (avg_rating_post < avg_rating_pre - 0.5)

        label_type = None
        first_cheat_round = None
        last_cheat_round = None
        if manipulation:
            label_type = "increasing_gap_and_rating_drop"
            cheat_rounds = [r for r in rounds if round_to_gap.get(r, 0.0) >= 1.0]
            if cheat_rounds:
                first_cheat_round = min(cheat_rounds)
                last_cheat_round = max(cheat_rounds)

        cursor.execute(
            """
            INSERT INTO analysis_labels (
                run_id, seed, seller_id, label_manipulator, label_type,
                first_cheat_round, last_cheat_round, cheat_rate_pre, cheat_rate_post
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                seed,
                seller_id,
                1 if manipulation else 0,
                label_type,
                first_cheat_round,
                last_cheat_round,
                cheat_rate_pre,
                cheat_rate_post,
            ),
        )

    conn.commit()


def run_detection(max_round: int, db_path: Optional[str] = None) -> None:
    with _connect(db_path) as conn:
        detect_seller_manipulation(conn, max_round)
