import os
import sqlite3
from typing import Dict, List

import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

DB_PATH = os.path.abspath("market_sim.db")
OUTPUT_DIR = os.path.abspath(os.path.join("analysis_output_e", "rqe"))


def ensure_output_dir(path: str = OUTPUT_DIR) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _safe_read(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()


def load_tables(db_path: str = DB_PATH) -> Dict[str, pd.DataFrame]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        user = _safe_read(conn, "user")
        post = _safe_read(conn, "post")
        tx = _safe_read(conn, "transactions")
    finally:
        conn.close()

    # normalize
    if not post.empty:
        for col in ["true_quality", "advertised_quality"]:
            if col in post.columns:
                post[col] = post[col].astype(str).str.upper().str.strip()
        for col in ["has_warrant", "is_sold"]:
            if col in post.columns:
                post[col] = post[col].astype("float").fillna(0).astype(int)
        if "round_number" in post.columns:
            post["prog_round"] = ((post["round_number"].astype(float) + 1) // 2).astype(int)
    if not tx.empty:
        for col in ["rating", "is_challenged"]:
            if col in tx.columns:
                tx[col] = tx[col].astype("float").fillna(0).astype(int)
        if "round_number" in tx.columns:
            tx["prog_round"] = ((tx["round_number"].astype(float) + 1) // 2).astype(int)

    return {"user": user, "post": post, "transactions": tx}


# ---------------------- RQe1 ----------------------
# 卖家的不诚实行为与最后Reputation_score是否相关

def rqe1_seller_dishonesty_vs_reputation(user: pd.DataFrame, post: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    if user.empty or post.empty:
        return pd.DataFrame()

    # 定义卖家维度的不诚实：其上架中 LQ且广告HQ 的listing数量与比例，以及成交中的不诚实销售数量与比例
    post_seller = post.copy()
    post_seller["dishonest_ad"] = ((post_seller["true_quality"]=="LQ") & (post_seller["advertised_quality"]=="HQ")).astype(int)
    agg_list = post_seller.groupby("user_id").agg(
        listings=("post_id", "count"),
        dishonest_ads=("dishonest_ad", "sum"),
        warrant_rate=("has_warrant", "mean"),
    ).reset_index().rename(columns={"user_id":"seller_id"})
    agg_list["dishonest_ad_rate"] = agg_list["dishonest_ads"].astype(float) / agg_list["listings"].replace(0, np.nan)

    # 成交维度的欺骗：基于交易表关联post的真伪信息
    tx_en = pd.DataFrame()
    if not tx.empty:
        tx_en = tx.merge(post[["post_id", "true_quality", "advertised_quality", "has_warrant", "user_id"]], on="post_id", how="left")
        tx_en = tx_en.rename(columns={"user_id":"seller_id_post"})
        tx_en["dishonest_sale"] = ((tx_en["true_quality"]=="LQ") & (tx_en["advertised_quality"]=="HQ")).astype(int)
        agg_tx = tx_en.groupby("seller_id").agg(
            sales=("transaction_id", "count"),
            dishonest_sales=("dishonest_sale", "sum"),
        ).reset_index()
        agg_tx["dishonest_sale_rate"] = agg_tx["dishonest_sales"].astype(float) / agg_tx["sales"].replace(0, np.nan)
    else:
        agg_tx = pd.DataFrame({"seller_id": [], "sales": [], "dishonest_sales": [], "dishonest_sale_rate": []})

    sellers = user[user.get("role", "").astype(str).str.lower()=="seller"][
        ["user_id", "agent_id", "reputation_score", "profit_utility_score"]
    ].rename(columns={"user_id":"seller_id"})

    out = sellers.merge(agg_list, on="seller_id", how="left").merge(agg_tx, on="seller_id", how="left")
    return out


# ---------------------- RQe2 ----------------------
# 四类post的rating均值与标准差

def rqe2_post_type_rating_stats(post: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    if post.empty:
        return pd.DataFrame()
    post = post.copy()
    # 定义四类
    def label_row(r):
        tq = str(r.get("true_quality", "")).upper()
        aq = str(r.get("advertised_quality", "")).upper()
        if tq=="LQ" and aq=="LQ":
            return "LQ->LQ"
        if tq=="LQ" and aq=="HQ":
            return "LQ->HQ"
        if tq=="HQ" and aq=="HQ":
            return "HQ->HQ"
        if tq=="HQ" and aq=="LQ":
            return "HQ->LQ"
        return "OTHER"
    post["type"] = post.apply(label_row, axis=1)

    if tx.empty:
        agg = post.groupby("type").size().to_frame("n_posts").reset_index()
        agg["rating_mean"] = np.nan
        agg["rating_std"] = np.nan
        return agg

    # 按post聚合rating（可能一个post多次交易，这里用交易rating均值）
    tx_post = tx.groupby("post_id").agg(rating_mean=("rating", "mean"), rating_std=("rating", "std"), n_tx=("transaction_id", "count")).reset_index()
    merged = post.merge(tx_post, on="post_id", how="left")
    agg = merged.groupby("type").agg(
        n_posts=("post_id", "count"),
        rating_mean=("rating_mean", "mean"),
        rating_std=("rating_std", "mean"),
        n_tx_total=("n_tx", "sum")
    ).reset_index()
    return agg


# ---------------------- RQe3 ----------------------
# 三个卖家的欺骗率与warrant率

def rqe3_top3_seller_deception_and_warrant(post: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    if post.empty:
        return pd.DataFrame()
    post = post.copy()
    post["dishonest_ad"] = ((post["true_quality"]=="LQ") & (post["advertised_quality"]=="HQ")).astype(int)
    # 选择按listings排序的前3个卖家
    sellers = post.groupby("user_id").size().sort_values(ascending=False).head(3).index.tolist()
    sub = post[post["user_id"].isin(sellers)]
    agg_list = sub.groupby("user_id").agg(
        listings=("post_id", "count"),
        dishonest_ads=("dishonest_ad", "sum"),
        warrant_rate=("has_warrant", "mean")
    ).reset_index().rename(columns={"user_id":"seller_id"})
    agg_list["dishonest_ad_rate"] = agg_list["dishonest_ads"].astype(float) / agg_list["listings"].replace(0, np.nan)

    # 交易维度的欺骗率（若无交易，则为NaN）
    if tx.empty:
        agg_tx = pd.DataFrame({"seller_id": [], "sales": [], "dishonest_sales": [], "dishonest_sale_rate": []})
    else:
        tx_en = tx.merge(post[["post_id", "true_quality", "advertised_quality", "user_id"]], on="post_id", how="left")
        tx_en = tx_en.rename(columns={"user_id":"seller_id_post"})
        tx_en["dishonest_sale"] = ((tx_en["true_quality"]=="LQ") & (tx_en["advertised_quality"]=="HQ")).astype(int)
        agg_tx = tx_en.groupby("seller_id").agg(
            sales=("transaction_id", "count"),
            dishonest_sales=("dishonest_sale", "sum"),
        ).reset_index()
        agg_tx["dishonest_sale_rate"] = agg_tx["dishonest_sales"].astype(float) / agg_tx["sales"].replace(0, np.nan)

    out = agg_list.merge(agg_tx, on="seller_id", how="left")
    return out


# ---------------------- RQe4 ----------------------
# 三个买家的rating与其购买warrant概况

def rqe4_top3_buyer_rating_and_warrant(tx: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    if tx.empty:
        return pd.DataFrame()
    # 选择交易次数最多的前三买家
    buyers = tx.groupby("buyer_id").size().sort_values(ascending=False).head(3).index.tolist()
    sub = tx[tx["buyer_id"].isin(buyers)]
    merged = sub.merge(post[["post_id", "has_warrant"]], on="post_id", how="left")
    agg = merged.groupby("buyer_id").agg(
        n_tx=("transaction_id", "count"),
        rating_mean=("rating", "mean"),
        rating_std=("rating", "std"),
        warrant_rate=("has_warrant", "mean")
    ).reset_index()
    return agg


# ---------------------- Plot & Export ----------------------

def export_csv(df: pd.DataFrame, name: str) -> None:
    if df is None or df.empty:
        return
    ensure_output_dir()
    df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)


def barplot_df(df: pd.DataFrame, x: str, y: str, title: str, name: str) -> None:
    if not HAS_PLOT or df.empty:
        return
    ensure_output_dir()
    plt.figure(figsize=(7,4))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    plt.close()


def main(db_path: str = DB_PATH) -> None:
    ensure_output_dir()
    tbl = load_tables(db_path)
    user, post, tx = tbl["user"], tbl["post"], tbl["transactions"]

    # RQe1
    rq1 = rqe1_seller_dishonesty_vs_reputation(user, post, tx)
    export_csv(rq1, "rqe1_seller_dishonesty_vs_reputation")
    if not rq1.empty:
        # 简单相关散点图：dishonest_ad_rate vs reputation
        if HAS_PLOT:
            plt.figure(figsize=(5,4))
            plt.scatter(rq1.get("dishonest_ad_rate", pd.Series(dtype=float)), rq1.get("reputation_score", pd.Series(dtype=float)))
            plt.xlabel("Dishonest Ad Rate")
            plt.ylabel("Reputation Score")
            plt.title("Seller Dishonesty vs Reputation")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "rqe1_scatter_dishonesty_vs_reputation.png"))
            plt.close()

    # RQe2
    rq2 = rqe2_post_type_rating_stats(post, tx)
    export_csv(rq2, "rqe2_post_type_rating_stats")
    if not rq2.empty and HAS_PLOT:
        barplot_df(rq2, x="type", y="rating_mean", title="Avg Rating by Post Type", name="rqe2_rating_mean_by_type")

    # RQe3
    rq3 = rqe3_top3_seller_deception_and_warrant(post, tx)
    export_csv(rq3, "rqe3_top3_seller_deception_and_warrant")
    if not rq3.empty and HAS_PLOT:
        barplot_df(rq3, x="seller_id", y="dishonest_ad_rate", title="Top3 Sellers Dishonest Ad Rate", name="rqe3_sellers_dishonest_ad_rate")
        barplot_df(rq3, x="seller_id", y="warrant_rate", title="Top3 Sellers Warrant Rate", name="rqe3_sellers_warrant_rate")

    # RQe4
    rq4 = rqe4_top3_buyer_rating_and_warrant(tx, post)
    export_csv(rq4, "rqe4_top3_buyer_rating_and_warrant")
    if not rq4.empty and HAS_PLOT:
        barplot_df(rq4, x="buyer_id", y="rating_mean", title="Top3 Buyers Rating Mean", name="rqe4_buyers_rating_mean")
        barplot_df(rq4, x="buyer_id", y="warrant_rate", title="Top3 Buyers Warrant Rate", name="rqe4_buyers_warrant_rate")

    print(f"RQe artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
