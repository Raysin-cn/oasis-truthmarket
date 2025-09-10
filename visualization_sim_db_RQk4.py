import os
import sqlite3
import json
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except Exception:  # pragma: no cover
    HAS_PLOT = False

# -----------------------------
# Config
# -----------------------------
DB_PATH = os.path.abspath("market_sim.db")
OUTPUT_DIR = os.path.abspath("analysis_output_k4")

# -----------------------------
# Utilities
# -----------------------------

def ensure_output_dir(path: str = OUTPUT_DIR) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _safe_read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()


def load_all_tables(db_path: str = DB_PATH) -> Dict[str, pd.DataFrame]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        user_df = _safe_read_table(conn, "user")
        post_df = _safe_read_table(conn, "post")
        tx_df = _safe_read_table(conn, "transactions")
        trace_df = _safe_read_table(conn, "trace")
    finally:
        conn.close()

    # dtype normalization
    if not post_df.empty:
        for col in ["has_warrant", "is_sold"]:
            if col in post_df.columns:
                post_df[col] = post_df[col].astype("float").fillna(0).astype(int)
        # normalize qualities to uppercase short tokens
        for col in ["true_quality", "advertised_quality"]:
            if col in post_df.columns:
                post_df[col] = post_df[col].astype(str).str.upper().str.strip()

    if not tx_df.empty:
        for col in ["is_challenged", "rating"]:
            if col in tx_df.columns:
                tx_df[col] = tx_df[col].astype("float").fillna(0).astype(int)

    return {
        "user": user_df,
        "post": post_df,
        "transactions": tx_df,
        "trace": trace_df,
    }


def enrich_transactions_with_post(tx_df: pd.DataFrame, post_df: pd.DataFrame) -> pd.DataFrame:
    if tx_df.empty:
        return tx_df.copy()
    if post_df.empty:
        out = tx_df.copy()
        out["tx_profit_naive"] = np.nan
        return out

    merged = tx_df.merge(
        post_df[["post_id", "user_id", "true_quality", "advertised_quality", "price", "cost", "has_warrant", "round_number"]],
        how="left",
        on="post_id",
        suffixes=("", "_post"),
    )
    # naive per-transaction profit = price - cost (penalties not represented in schema)
    merged["tx_profit_naive"] = (merged["price"].fillna(0) - merged["cost"].fillna(0))

    # flags
    merged["dishonest_ad"] = (
        (merged["true_quality"].str.upper() == "LQ") & (merged["advertised_quality"].str.upper() == "HQ")
    ).astype(int)
    merged["honest_ad"] = (
        (merged["true_quality"].str.upper() == merged["advertised_quality"].str.upper())
    ).astype(int)

    return merged


def build_round_index(df: pd.DataFrame, round_col: str = "prog_round") -> List[int]:
    if df.empty or round_col not in df.columns:
        return []
    rounds = (
        df[round_col].dropna().astype(int).sort_values().unique().tolist()
    )
    return rounds
# -----------------------------
# Round reconstruction & pretty print
# -----------------------------
from dataclasses import dataclass

@dataclass
class RoundSummary:
    round_number: int
    listings: List[Dict]
    sales: List[Dict]


def reconstruct_rounds(post_df: pd.DataFrame, tx_df: pd.DataFrame) -> List[RoundSummary]:
    summaries: List[RoundSummary] = []
    if post_df.empty and tx_df.empty:
        return summaries

    post_df = post_df.copy()
    tx_df = tx_df.copy()

    # Map DB round -> program round (prog_round)
    if not post_df.empty and "round_number" in post_df.columns:
        post_df["prog_round"] = ((post_df["round_number"].astype(float) + 1) // 2).astype(int)
    if not tx_df.empty and "round_number" in tx_df.columns:
        tx_df["prog_round"] = ((tx_df["round_number"].astype(float) + 1) // 2).astype(int)

    # Build per program-round
    all_rounds = sorted(set(build_round_index(post_df) + build_round_index(tx_df)))

    for r in all_rounds:
        # listings created in program round r
        listings_r = post_df[post_df.get("prog_round", pd.Series([np.nan]*len(post_df))).fillna(-1) == r]
        listings_view = []
        for _, row in listings_r.iterrows():
            listings_view.append({
                "post_id": int(row.get("post_id", -1)),
                "seller_id": int(row.get("user_id", -1)),
                "true_quality": str(row.get("true_quality", "")),
                "advertised_quality": str(row.get("advertised_quality", "")),
                "price": float(row.get("price", 0) or 0),
                "has_warrant": int(row.get("has_warrant", 0) or 0),
                "is_sold": int(row.get("is_sold", 0) or 0),
                "status": str(row.get("status", "")),
            })

        tx_r = tx_df[tx_df.get("prog_round", pd.Series([np.nan]*len(tx_df))).fillna(-1) == r]
        sales_view = []
        for _, row in tx_r.iterrows():
            sales_view.append({
                "transaction_id": int(row.get("transaction_id", -1)),
                "post_id": int(row.get("post_id", -1)),
                "seller_id": int(row.get("seller_id", -1)),
                "buyer_id": int(row.get("buyer_id", -1)),
                "rating": int(row.get("rating", 0) or 0),
                "is_challenged": int(row.get("is_challenged", 0) or 0),
            })

        summaries.append(RoundSummary(round_number=r, listings=listings_view, sales=sales_view))

    return summaries


def print_rounds(summaries: List[RoundSummary]) -> None:
    for s in summaries:
        print(f"\n==================== Round {s.round_number} ====================")
        print("-- Listings --")
        if not s.listings:
            print("(none)")
        for it in s.listings:
            warr = " (Warranted)" if it["has_warrant"] else ""
            sold = " [SOLD]" if it["is_sold"] else ""
            print(f"- Post {it['post_id']}: Seller {it['seller_id']}, True {it['true_quality']}, Adv {it['advertised_quality']}, Price ${it['price']:.2f}{warr}{sold}")
        print("-- Sales --")
        if not s.sales:
            print("(none)")
        for it in s.sales:
            ch = ", challenged" if it["is_challenged"] else ""
            print(f"- Tx {it['transaction_id']}: Post {it['post_id']}, Seller {it['seller_id']}, Buyer {it['buyer_id']}, Rating {it['rating']}{ch}")
# -----------------------------
# Metrics for RQ1–RQ4
# -----------------------------

# RQ4: deception metrics

def compute_deception_metrics(post_df: pd.DataFrame, tx_enriched: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if post_df.empty:
        out["by_round"] = pd.DataFrame()
        out["overall"] = pd.DataFrame()
        return out

    post = post_df.copy()
    post["dishonest_ad"] = ((post["true_quality"].str.upper()=="LQ") & (post["advertised_quality"].str.upper()=="HQ")).astype(int)

    # per program-round listings
    # ensure prog_round exists
    if "prog_round" not in post.columns and "round_number" in post.columns:
        post["prog_round"] = ((post["round_number"].astype(float) + 1) // 2).astype(int)
    grp = post.groupby("prog_round").agg(
        n_listings=("post_id", "count"),
        n_hq=("true_quality", lambda s: (s.str.upper()=="HQ").sum()),
        n_lq=("true_quality", lambda s: (s.str.upper()=="LQ").sum()),
        n_dishonest_ad=("dishonest_ad", "sum"),
    ).reset_index().rename(columns={"prog_round":"round_number"})

    # dishonest sales: via transactions enriched (group by prog_round)
    if tx_enriched is not None and not tx_enriched.empty:
        ds = tx_enriched.copy()
        if "prog_round" not in ds.columns and "round_number" in ds.columns:
            ds["prog_round"] = ((ds["round_number"].astype(float) + 1) // 2).astype(int)
        ds["dishonest_sale"] = ((ds["true_quality"].str.upper()=="LQ") & (ds["advertised_quality"].str.upper()=="HQ")).astype(int)
        ds_round = ds.groupby("prog_round").agg(
            n_sales=("transaction_id", "count"),
            n_dishonest_sale=("dishonest_sale", "sum"),
        ).reset_index().rename(columns={"prog_round":"round_number"})
        grp = grp.merge(ds_round, on="round_number", how="left").fillna(0)
    else:
        grp["n_sales"] = 0
        grp["n_dishonest_sale"] = 0

    grp["prop_dishonest_ad"] = grp["n_dishonest_ad"].astype(float) / grp["n_listings"].replace(0, np.nan)
    grp["prop_dishonest_sale"] = grp["n_dishonest_sale"].astype(float) / grp["n_sales"].replace(0, np.nan)

    overall = pd.DataFrame({
        "n_listings": [int(post.shape[0])],
        "n_hq": [int((post["true_quality"].str.upper()=="HQ").sum())],
        "n_lq": [int((post["true_quality"].str.upper()=="LQ").sum())],
        "n_dishonest_ad": [int(post["dishonest_ad"].sum())],
        "n_sales": [int(0 if tx_enriched is None or tx_enriched.empty else tx_enriched.shape[0])],
        "n_dishonest_sale": [int(0 if tx_enriched is None or tx_enriched.empty else (((tx_enriched["true_quality"].str.upper()=="LQ") & (tx_enriched["advertised_quality"].str.upper()=="HQ")).sum()))],
    })

    out["by_round"] = grp
    out["overall"] = overall
    return out


# RQ1: profit vs dishonesty

def compute_profit_metrics(user_df: pd.DataFrame, tx_enriched: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if user_df.empty:
        out["seller_profit"] = pd.DataFrame()
        out["tx_panel"] = pd.DataFrame()
        return out

    sellers = user_df[user_df.get("role", "").astype(str).str.lower()=="seller"].copy()

    # cumulative profit_utility_score as reported in user table (end-of-sim state)
    seller_profit = sellers[["user_id", "agent_id", "budget", "profit_utility_score", "reputation_score"]].copy()

    # per-transaction panel with dishonesty flag
    tx_panel = pd.DataFrame()
    if tx_enriched is not None and not tx_enriched.empty:
        tx_panel = tx_enriched.copy()
        # attach program round
        if "prog_round" not in tx_panel.columns and "round_number" in tx_panel.columns:
            tx_panel["prog_round"] = ((tx_panel["round_number"].astype(float) + 1) // 2).astype(int)
        tx_panel["dishonest_sale"] = ((tx_panel["true_quality"].str.upper()=="LQ") & (tx_panel["advertised_quality"].str.upper()=="HQ")).astype(int)
        tx_panel["tx_profit_naive"] = tx_panel["tx_profit_naive"].astype(float)

    out["seller_profit"] = seller_profit
    out["tx_panel"] = tx_panel
    return out


# RQ2: inequality metrics

def gini(array: np.ndarray) -> float:
    x = np.array(array, dtype=float)
    if x.size == 0:
        return np.nan
    x = x.flatten()
    if np.amin(x) < 0:
        x = x - np.amin(x)
    mean_x = np.mean(x)
    if mean_x == 0:
        return 0.0
    diff_sum = np.abs(np.subtract.outer(x, x)).mean()
    return diff_sum / (2 * mean_x)


def compute_inequality_metrics(user_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if user_df.empty:
        return out
    sellers = user_df[user_df.get("role", "").astype(str).str.lower()=="seller"].copy()
    buyers = user_df[user_df.get("role", "").astype(str).str.lower()=="buyer"].copy()

    seller_profit = sellers.get("profit_utility_score", pd.Series(dtype=float)).astype(float).fillna(0).values
    buyer_utility = buyers.get("profit_utility_score", pd.Series(dtype=float)).astype(float).fillna(0).values

    out["gini_seller_profit"] = gini(seller_profit)
    out["var_seller_profit"] = float(np.var(seller_profit))
    out["p90_p50_seller_profit"] = float(np.percentile(seller_profit, 90) - np.percentile(seller_profit, 50)) if seller_profit.size>0 else np.nan

    out["gini_buyer_utility"] = gini(buyer_utility)
    out["var_buyer_utility"] = float(np.var(buyer_utility))
    return out


# RQ3: short-term vs long-term reasoning (heuristic from trace)
SHORT_PATTERNS = [
    "immediate profit", "quick profit", "one-off", "arbitrage", "short-term",
    "ignore reputation", "no future", "maximize now"
]
LONG_PATTERNS = [
    "reputation", "future", "long-term", "build trust", "warrant",
    "challenge", "penalty", "future sales", "sustainable"
]


def annotate_reasoning(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame()
    df = trace_df.copy()
    text = df.get("info", pd.Series([""]*len(df))).astype(str).str.lower()
    def count_hits(patterns, s):
        return sum(s.str.contains(p, regex=False) for p in patterns)
    df["short_hits"] = count_hits(SHORT_PATTERNS, text)
    df["long_hits"] = count_hits(LONG_PATTERNS, text)
    df["long_share"] = (df["long_hits"].astype(float) / (df["short_hits"] + df["long_hits"]).replace(0, np.nan))
    return df
# -----------------------------
# Plotting & Export
# -----------------------------

def export_csv(df: pd.DataFrame, name: str) -> None:
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)


def plot_line(df: pd.DataFrame, x: str, y: str, title: str, name: str) -> None:
    if not HAS_PLOT or df.empty or x not in df.columns or y not in df.columns:
        return
    ensure_output_dir()
    plt.figure(figsize=(7,4))
    sns.lineplot(data=df, x=x, y=y, marker="o")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    plt.close()


def plot_bars(df: pd.DataFrame, x: str, y: str, title: str, name: str) -> None:
    if not HAS_PLOT or df.empty or x not in df.columns or y not in df.columns:
        return
    ensure_output_dir()
    plt.figure(figsize=(7,4))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    plt.close()

# -----------------------------
# Entry
# -----------------------------

def main(db_path: str = DB_PATH) -> None:
    ensure_output_dir()
    tables = load_all_tables(db_path)
    user_df, post_df, tx_df, trace_df = tables["user"], tables["post"], tables["transactions"], tables["trace"]

    # reconstruct rounds
    tx_enriched = enrich_transactions_with_post(tx_df, post_df)
    round_summaries = reconstruct_rounds(post_df, tx_df)
    print_rounds(round_summaries)

    # RQ4 deception (grouped by program rounds)
    deception = compute_deception_metrics(post_df, tx_enriched)
    by_round = deception["by_round"]
    if not by_round.empty:
        export_csv(by_round, "rq4_deception_by_prog_round")
        plot_line(by_round, "round_number", "prop_dishonest_ad", "Dishonest Advertisements by Program Round", "rq4_prop_dishonest_ad_prog")
        plot_line(by_round, "round_number", "prop_dishonest_sale", "Dishonest Sales by Program Round", "rq4_prop_dishonest_sale_prog")

    # RQ1 profit metrics (report tables; regression left for notebook/advanced step)
    profit = compute_profit_metrics(user_df, tx_enriched)
    if not profit["seller_profit"].empty:
        export_csv(profit["seller_profit"], "rq1_seller_profit")
    if not profit["tx_panel"].empty:
        export_csv(profit["tx_panel"], "rq1_tx_panel")

    # RQ2 inequality
    ineq = compute_inequality_metrics(user_df)
    with open(os.path.join(OUTPUT_DIR, "rq2_inequality.json"), "w", encoding="utf-8") as f:
        json.dump(ineq, f, ensure_ascii=False, indent=2)

    # RQ3 reasoning annotations
    reasoning = annotate_reasoning(trace_df)
    if not reasoning.empty:
        export_csv(reasoning, "rq3_reasoning_annotations")
        # optional: aggregate by round if trace has round info in info text or separate column

    print(f"Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
