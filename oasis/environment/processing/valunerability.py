import os
import sqlite3
from typing import Optional, Tuple

DATABASE_PATH = os.getenv("MARKET_DB_PATH", "market_sim.db")

def avg(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or DATABASE_PATH)


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create analysis tables with enhanced behavioral pattern detection fields"""
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
            -- Enhanced pattern detection fields
            pattern_confidence REAL,          -- Detection confidence [0,1]
            deception_rate_overall REAL,      -- Overall deception rate
            adaptation_detected BOOLEAN,      -- Whether adaptation behavior found
            warrant_abuse_detected BOOLEAN,   -- Whether warrant abuse found
            exit_reentry_detected BOOLEAN,    -- Whether exit-reentry pattern found
            behavioral_pattern TEXT,          -- Primary pattern: systematic/opportunistic/adaptive/honest
            market_impact_score REAL,         -- Estimated impact on market efficiency
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
    # Add missing columns if they don't exist (for schema migration)
    missing_columns = [
        "pattern_confidence REAL",
        "deception_rate_overall REAL", 
        "adaptation_detected BOOLEAN",
        "warrant_abuse_detected BOOLEAN",
        "exit_reentry_detected BOOLEAN",
        "behavioral_pattern TEXT",
        "market_impact_score REAL"
    ]
    
    for column_def in missing_columns:
        try:
            cursor.execute(f"ALTER TABLE analysis_labels ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_analysis_labels_pattern
        ON analysis_labels (behavioral_pattern);
        """
    )
    conn.commit()


def _get_run_meta() -> Tuple[int, Optional[int]]:
    run_id = int(os.getenv("RUN_ID", "1"))
    seed_env = os.getenv("SEED")
    seed = int(seed_env) if seed_env is not None and seed_env.isdigit() else None
    return run_id, seed


# Detection flags for different behavioral patterns
DETECTION_FLAGS = {
    "enable_systematic_deception": True,    # High frequency deception (>70% rounds)
    "enable_opportunistic_deception": True, # Deception before reputation damage
    "enable_adaptive_learning": True,       # Initial deception then improvement
    "enable_exit_reentry": True,           # Exit after damage, reenter fresh
    "enable_warrant_abuse": True,          # False warranties with deception
}

# Configurable thresholds for pattern detection
DETECTION_THRESHOLDS = {
    "systematic_deception_rate": 0.7,      # Minimum rate for systematic pattern
    "opportunistic_reputation_drop": 0.3,  # Reputation drop threshold
    "adaptive_improvement_ratio": 0.5,     # Improvement ratio for learning
    "reentry_gap_min": 2,                  # Minimum rounds gap for reentry
    "warrant_abuse_rate": 0.5,             # Deception rate when offering warranty
    "deception_increase_delta": 0.4,       # Legacy: post - pre deception rate
    "rating_drop_delta": 0.5,              # Legacy: pre - post avg rating
}


def _compute_round_series(cursor: sqlite3.Cursor, seller_id: int):
    cursor.execute(
        """
        SELECT p.round_number, p.advertised_quality, p.true_quality, p.price, p.has_warrant
        FROM product p
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


def _get_reputation_history(cursor: sqlite3.Cursor, seller_id: int):
    """Get reputation history for the seller"""
    cursor.execute(
        """
        SELECT round, public_reputation_score
        FROM reputation_history
        WHERE seller_id = ?
        ORDER BY round ASC
        """,
        (seller_id,),
    )
    return {int(r): float(rep) for r, rep in cursor.fetchall()}


def _detect_systematic_deception(deception_pattern: list, threshold: float = 0.7) -> Tuple[bool, float]:
    """Detect systematic deception pattern (consistently high deception rate)"""
    if not deception_pattern:
        return False, 0.0
    
    deception_rate = sum(deception_pattern) / len(deception_pattern)
    is_systematic = deception_rate > threshold
    confidence = min(1.0, deception_rate / threshold) if is_systematic else 0.0
    
    return is_systematic, confidence


def _detect_opportunistic_deception(deception_pattern: list, reputation_history: dict, 
                                  reputation_drop_threshold: float = 0.3) -> Tuple[bool, float]:
    """Detect opportunistic deception (deception before reputation drops)"""
    if len(deception_pattern) < 3 or not reputation_history:
        return False, 0.0
    
    # Find reputation drops
    rep_values = list(reputation_history.values())
    reputation_drops = 0
    pre_drop_deceptions = 0
    
    for i in range(1, len(rep_values)):
        if rep_values[i] < rep_values[i-1] - reputation_drop_threshold:
            reputation_drops += 1
            # Check if there was deception in the 2 rounds before the drop
            if i >= 2:
                recent_deceptions = sum(deception_pattern[max(0, i-2):i])
                if recent_deceptions >= 1:
                    pre_drop_deceptions += 1
    
    if reputation_drops == 0:
        return False, 0.0
    
    opportunistic_ratio = pre_drop_deceptions / reputation_drops
    is_opportunistic = opportunistic_ratio > 0.5
    confidence = opportunistic_ratio if is_opportunistic else 0.0
    
    return is_opportunistic, confidence


def _detect_adaptive_learning(deception_pattern: list, improvement_threshold: float = 0.5) -> Tuple[bool, float]:
    """Detect adaptive learning (initial deception followed by improvement)"""
    if len(deception_pattern) < 4:
        return False, 0.0
    
    mid_point = len(deception_pattern) // 2
    early_deception = sum(deception_pattern[:mid_point]) / mid_point
    late_deception = sum(deception_pattern[mid_point:]) / (len(deception_pattern) - mid_point)
    
    # Adaptive learning: high early deception, significant improvement later
    improvement_ratio = (early_deception - late_deception) / early_deception if early_deception > 0 else 0
    is_adaptive = early_deception > 0.4 and improvement_ratio > improvement_threshold
    confidence = improvement_ratio if is_adaptive else 0.0
    
    return is_adaptive, confidence


def _detect_exit_reentry(rounds: list, min_gap: int = 2) -> Tuple[bool, float]:
    """Detect exit and reentry pattern (gaps in participation)"""
    if len(rounds) < 3:
        return False, 0.0
    
    gaps = []
    for i in range(1, len(rounds)):
        gap = rounds[i] - rounds[i-1]
        if gap > min_gap:
            gaps.append(gap)
    
    # Exit-reentry pattern if there are significant gaps
    is_exit_reentry = len(gaps) > 0
    confidence = min(1.0, len(gaps) * 0.5) if is_exit_reentry else 0.0
    
    return is_exit_reentry, confidence


def _detect_warrant_abuse(deception_pattern: list, warrant_pattern: list, 
                         abuse_threshold: float = 0.5) -> Tuple[bool, float]:
    """Detect warrant abuse (deception while offering warranties)"""
    if not warrant_pattern or len(warrant_pattern) != len(deception_pattern):
        return False, 0.0
    
    warrant_deceptions = 0
    total_warrants = 0
    
    for i, has_warrant in enumerate(warrant_pattern):
        if has_warrant:
            total_warrants += 1
            if deception_pattern[i]:
                warrant_deceptions += 1
    
    if total_warrants == 0:
        return False, 0.0
    
    abuse_rate = warrant_deceptions / total_warrants
    is_abuse = abuse_rate > abuse_threshold
    confidence = abuse_rate if is_abuse else 0.0
    
    return is_abuse, confidence


def _classify_behavioral_pattern(deception_pattern: list, reputation_history: dict, 
                               rounds: list, warrant_pattern: list) -> Tuple[str, float]:
    """Classify the primary behavioral pattern and return confidence"""
    
    # Test all patterns
    systematic, sys_conf = _detect_systematic_deception(deception_pattern, 
                                                       DETECTION_THRESHOLDS["systematic_deception_rate"])
    opportunistic, opp_conf = _detect_opportunistic_deception(deception_pattern, reputation_history,
                                                             DETECTION_THRESHOLDS["opportunistic_reputation_drop"])
    adaptive, adapt_conf = _detect_adaptive_learning(deception_pattern,
                                                    DETECTION_THRESHOLDS["adaptive_improvement_ratio"])
    exit_reentry, exit_conf = _detect_exit_reentry(rounds, DETECTION_THRESHOLDS["reentry_gap_min"])
    warrant_abuse, warrant_conf = _detect_warrant_abuse(deception_pattern, warrant_pattern,
                                                       DETECTION_THRESHOLDS["warrant_abuse_rate"])
    
    # Find the strongest pattern
    patterns = [
        ("systematic", sys_conf if systematic else 0),
        ("opportunistic", opp_conf if opportunistic else 0),
        ("adaptive", adapt_conf if adaptive else 0),
        ("exit_reentry", exit_conf if exit_reentry else 0),
        ("warrant_abuse", warrant_conf if warrant_abuse else 0)
    ]
    
    # Sort by confidence
    patterns.sort(key=lambda x: x[1], reverse=True)
    
    # If no strong pattern, classify as honest or opportunistic based on overall deception
    if patterns[0][1] == 0:
        overall_deception = sum(deception_pattern) / len(deception_pattern) if deception_pattern else 0
        if overall_deception == 0:
            return "honest", 1.0
        else:
            return "opportunistic", overall_deception
    
    return patterns[0][0], patterns[0][1]


def detect_seller_manipulation(conn: sqlite3.Connection, max_round: int) -> None:
    """
    Enhanced behavioral pattern detection for sellers.
    Analyzes multiple patterns: systematic deception, opportunistic behavior, 
    adaptive learning, exit-reentry, and warrant abuse.
    Maintains backward compatibility with legacy detection method.
    """
    ensure_tables(conn)
    cursor = conn.cursor()

    # Get all sellers
    cursor.execute("SELECT user_id FROM user WHERE role = 'seller'")
    seller_ids = [r[0] for r in cursor.fetchall()]

    run_id, seed = _get_run_meta()

    for seller_id in seller_ids:
        # Get seller's posting data
        rounds, adv_quality, true_quality, prices, warrant_pattern = _compute_round_series(cursor, seller_id)
        
        if not rounds:
            continue

        # Get ratings and reputation history
        round_ratings = _compute_round_ratings(cursor, seller_id)
        reputation_history = _get_reputation_history(cursor, seller_id)

        # Compute deception pattern (HQ advertised but LQ delivered)
        deception_pattern = []
        warrant_list = []
        for round_num in rounds:
            adv_q = adv_quality.get(round_num)
            true_q = true_quality.get(round_num)
            has_warrant = warrant_pattern.get(round_num, False)
            
            is_deception = (adv_q == 'HQ' and true_q == 'LQ')
            deception_pattern.append(1 if is_deception else 0)
            warrant_list.append(has_warrant)

        # Legacy detection for backward compatibility
        first_half = [r for r in rounds if r <= max_round // 2]
        second_half = [r for r in rounds if r > max_round // 2]
        
        if first_half and second_half:
            cheat_rate_pre = avg([deception_pattern[rounds.index(r)] for r in first_half])
            cheat_rate_post = avg([deception_pattern[rounds.index(r)] for r in second_half])
            avg_rating_pre = avg([round_ratings.get(r, 0.0) for r in first_half])
            avg_rating_post = avg([round_ratings.get(r, 0.0) for r in second_half])
        else:
            cheat_rate_pre = cheat_rate_post = 0.0
            avg_rating_pre = avg_rating_post = 0.0

        # Legacy manipulation detection
        legacy_manipulation = (
            (cheat_rate_post - cheat_rate_pre >= DETECTION_THRESHOLDS["deception_increase_delta"]) and 
            (avg_rating_post < avg_rating_pre - DETECTION_THRESHOLDS["rating_drop_delta"])
        )

        # Enhanced pattern analysis
        behavioral_pattern, pattern_confidence = _classify_behavioral_pattern(
            deception_pattern, reputation_history, rounds, warrant_list
        )

        # Detect specific patterns for detailed analysis
        _, adaptation_conf = _detect_adaptive_learning(deception_pattern, 
                                                     DETECTION_THRESHOLDS["adaptive_improvement_ratio"])
        _, warrant_abuse_conf = _detect_warrant_abuse(deception_pattern, warrant_list,
                                                     DETECTION_THRESHOLDS["warrant_abuse_rate"])
        _, exit_reentry_conf = _detect_exit_reentry(rounds, DETECTION_THRESHOLDS["reentry_gap_min"])

        # Calculate overall metrics
        overall_deception_rate = sum(deception_pattern) / len(deception_pattern) if deception_pattern else 0.0
        
        # Estimate market impact (simple heuristic: deception rate * transaction volume)
        total_transactions = len([r for r in rounds if r in round_ratings])
        market_impact = min(1.0, overall_deception_rate * (total_transactions / max_round))

        # Determine label type for legacy compatibility
        label_type = None
        first_cheat_round = None
        last_cheat_round = None
        
        if legacy_manipulation or pattern_confidence > 0.5:
            if behavioral_pattern == "systematic":
                label_type = "systematic_deception"
            elif behavioral_pattern == "opportunistic":
                label_type = "opportunistic_deception"
            elif behavioral_pattern == "adaptive":
                label_type = "adaptive_learning"
            elif behavioral_pattern == "exit_reentry":
                label_type = "exit_reentry_pattern"
            elif behavioral_pattern == "warrant_abuse":
                label_type = "warrant_abuse"
            else:
                label_type = "increasing_gap_and_rating_drop"  # Legacy
            
            # Find first and last deception rounds
            cheat_rounds = [rounds[i] for i, dec in enumerate(deception_pattern) if dec == 1]
            if cheat_rounds:
                first_cheat_round = min(cheat_rounds)
                last_cheat_round = max(cheat_rounds)

        # Insert enhanced analysis results
        cursor.execute(
            """
            INSERT INTO analysis_labels (
                run_id, seed, seller_id, label_manipulator, label_type,
                first_cheat_round, last_cheat_round, cheat_rate_pre, cheat_rate_post,
                pattern_confidence, deception_rate_overall, adaptation_detected,
                warrant_abuse_detected, exit_reentry_detected, behavioral_pattern,
                market_impact_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, seed, seller_id,
                1 if (legacy_manipulation or pattern_confidence > 0.5) else 0,
                label_type,
                first_cheat_round, last_cheat_round,
                cheat_rate_pre, cheat_rate_post,
                pattern_confidence, overall_deception_rate,
                1 if adaptation_conf > 0.3 else 0,
                1 if warrant_abuse_conf > 0.3 else 0,
                1 if exit_reentry_conf > 0.3 else 0,
                behavioral_pattern,
                market_impact
            ),
        )

    conn.commit()
    print(f"Enhanced behavioral analysis completed for {len(seller_ids)} sellers")


def run_detection(max_round: int, db_path: Optional[str] = None) -> None:
    with _connect(db_path) as conn:
        detect_seller_manipulation(conn, max_round)
