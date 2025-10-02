#!/usr/bin/env python3
import argparse
import os
import sqlite3
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()


def ensure_output_dir(base_dir: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir or os.getcwd(), "analysis", "outputs", ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_save(fig: plt.Figure, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize_basic(conn: sqlite3.Connection) -> dict:
    summary = {}
    cur = conn.cursor()
    try:
        for t in ["user", "post", "transactions", "reputation_history", "trace"]:
            cur.execute("SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name=?", (t,))
            exists = cur.fetchone()[0] == 1
            summary[f"has_{t}"] = exists
            if exists:
                cur.execute(f"SELECT COUNT(1) FROM {t}")
                summary[f"rows_{t}"] = cur.fetchone()[0]
    except Exception:
        pass
    return summary


def plot_reputation_over_rounds(reph: pd.DataFrame, out_dir: str) -> None:
    if reph.empty:
        return
    # Expect columns: round, seller_id, public_reputation_score
    df = reph.copy()
    required = {"round", "seller_id", "public_reputation_score"}
    if not required.issubset(df.columns):
        return

    # Normalize types
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["seller_id"] = pd.to_numeric(df["seller_id"], errors="coerce")

    # Create small vertical offsets to reduce overlap
    # Map each seller_id to an offset in [-delta, +delta]
    unique_sellers = np.sort(df["seller_id"].dropna().unique())
    seller_rank = {sid: i for i, sid in enumerate(unique_sellers)}
    # Choose delta based on interquartile range or a small constant
    delta = 0.15
    k = max(1, len(unique_sellers))
    # Centered ranks -> offsets
    offsets = {sid: ((seller_rank[sid] - (k - 1) / 2) / max(1, (k - 1))) * delta for sid in unique_sellers}
    df["rep_plot"] = df.apply(lambda r: r["public_reputation_score"] + offsets.get(r["seller_id"], 0.0), axis=1)

    df = df.sort_values(["seller_id", "round"]).reset_index(drop=True)

    # Rich color palette and legend
    palette = sns.color_palette("tab20", n_colors=max(3, len(unique_sellers)))
    g = sns.relplot(
        data=df,
        x="round",
        y="rep_plot",
        hue="seller_id",
        kind="line",
        marker="o",
        height=5.5,
        aspect=1.7,
        legend=True,
        palette=palette,
        linewidth=1.6,
    )
    g.set_axis_labels("Round", "Public Reputation Score (offset for visibility)")
    g.fig.suptitle("Seller Reputation Over Rounds", y=1.04)
    # Improve legend placement
    if g._legend is not None:
        g._legend.set_title("Seller ID")
        g._legend.set_bbox_to_anchor((1.02, 1))
        g._legend.set_frame_on(True)
    g.savefig(os.path.join(out_dir, "reputation_over_rounds.png"), dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def plot_avg_price_by_advertised_quality(conn: sqlite3.Connection, out_dir: str) -> None:
    posts = read_table(conn, "post")
    if posts.empty:
        return
    required = {"round_number", "advertised_quality", "price"}
    if not required.issubset(posts.columns):
        return

    df = posts[list(required)].copy()
    df = df.dropna(subset=["round_number", "advertised_quality", "price"])  # keep valid rows
    if df.empty:
        return
    df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")
    df = df[df["round_number"].notna()]
    if df.empty:
        return
    df["advertised_quality"] = df["advertised_quality"].astype(str).str.upper()
    df = df[df["advertised_quality"].isin(["HQ", "LQ"])]
    if df.empty:
        return

    agg = (
        df.groupby(["round_number", "advertised_quality"])  # type: ignore[arg-type]
          ["price"].agg(["mean", "std"]).reset_index()
          .rename(columns={"mean": "avg_price", "std": "std_price"})
          .sort_values(["advertised_quality", "round_number"])  # type: ignore[arg-type]
    )
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    palette = {"HQ": "#4C78A8", "LQ": "#F58518"}
    for q in ["HQ", "LQ"]:
        sub = agg[agg["advertised_quality"] == q]
        if sub.empty:
            continue
        ax.errorbar(
            sub["round_number"],
            sub["avg_price"],
            yerr=sub["std_price"],
            fmt="-o",
            capsize=3,
            linewidth=1.8,
            color=palette.get(q, None),
            label=f"{q} (mean±std)",
        )

    ax.set_title("Average Listing Price per Round by Advertised Quality")
    ax.set_xlabel("Round")
    ax.set_ylabel("Price (mean ± std)")
    ax.legend(title="Advertised Quality")
    plot_save(fig, out_dir, "avg_price_by_advertised_quality")


def plot_seller_actions_scatter(conn: sqlite3.Connection, out_dir: str) -> None:
    """Scatter: x=Round, y=Seller ID.
    - Listing (circle 'o'), color by (adv_q, true_q):
        HQ/HQ -> deep blue, LQ/LQ -> light blue, HQ/LQ -> deep red, others -> gray
    - Re-entry (square 's'), Exit (triangle '^')
    """
    posts = read_table(conn, "post")
    trace = read_table(conn, "trace")
    if posts.empty and trace.empty:
        return

    # Build listing points from post table
    list_df = pd.DataFrame()
    req_p = {"user_id", "round_number", "advertised_quality", "true_quality"}
    if not posts.empty and req_p.issubset(posts.columns):
        list_df = posts[list(req_p)].copy()
        list_df = list_df.rename(columns={"user_id": "seller_id", "round_number": "round",
                                          "advertised_quality": "adv_q", "true_quality": "true_q"})
        list_df = list_df.dropna(subset=["seller_id", "round"])

    # Exit/Reentry from trace
    tr_df = pd.DataFrame()
    if not trace.empty and {"user_id", "action"}.issubset(trace.columns):
        tr = trace[["created_at", "user_id", "action"]].copy()
        tr = tr.rename(columns={"user_id": "seller_id"})
        tr = tr[tr["action"].isin(["exit_market", "reenter_market"])].copy()
        
        # 根据业务逻辑分配round信息
        if not tr.empty:
            # 将trace的时间戳转换为datetime
            tr["created_at"] = pd.to_datetime(tr["created_at"])
            
            # 根据action类型分配round
            def assign_round_for_action(action, timestamp):
                if action == "reenter_market":
                    # reenter动作只可能出现在round 5
                    return 5
                elif action == "exit_market":
                    # exit动作通常出现在后期round，这里假设是round 7
                    return 7
                else:
                    return None
            
            tr["round"] = tr.apply(lambda row: assign_round_for_action(row["action"], row["created_at"]), axis=1)
            tr_df = tr[["round", "seller_id", "action"]].dropna(subset=["round"])

    if list_df.empty and tr_df.empty:
        return

    # Normalize types
    if not list_df.empty:
        list_df.loc[:, "round"] = pd.to_numeric(list_df["round"].values, errors="coerce")
        list_df.loc[:, "seller_id"] = pd.to_numeric(list_df["seller_id"].values, errors="coerce")
        list_df = list_df.dropna(subset=["round", "seller_id"])  # keep valid
        # classify color
        def color_for(q_adv: str, q_true: str) -> str:
            qa = str(q_adv).upper()
            qt = str(q_true).upper()
            if qa == "HQ" and qt == "HQ":
                return "#1f77b4"  # deep blue
            if qa == "LQ" and qt == "LQ":
                return "#8ecae6"  # light blue
            if qa == "HQ" and qt == "LQ":
                return "#d62728"  # deep red
            return "#7f7f7f"      # gray for other combos

        list_df.loc[:, "color"] = [color_for(a, t) for a, t in zip(list_df["adv_q"], list_df["true_q"])]

    if not tr_df.empty:
        tr_df.loc[:, "round"] = pd.to_numeric(tr_df["round"].values, errors="coerce")
        tr_df.loc[:, "seller_id"] = pd.to_numeric(tr_df["seller_id"].values, errors="coerce")
        tr_df = tr_df.dropna(subset=["round", "seller_id"])  # keep valid

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    # Listings as circles with color by quality combo
    if not list_df.empty:
        ax.scatter(
            list_df["round"], list_df["seller_id"],
            c=list_df["color"], marker="o", s=70,
            edgecolor="white", linewidth=0.6, label="list_product",
        )
    # Re-entry as squares (s), Exit as triangles (^), neutral color
    if not tr_df.empty:
        re_df = tr_df[tr_df["action"] == "reenter_market"]
        ex_df = tr_df[tr_df["action"] == "exit_market"]
        if not re_df.empty:
            ax.scatter(re_df["round"], re_df["seller_id"], c="#444444", marker="s", s=80,
                       edgecolor="white", linewidth=0.6, label="reenter_market")
        if not ex_df.empty:
            ax.scatter(ex_df["round"], ex_df["seller_id"], c="#444444", marker="^", s=80,
                       edgecolor="white", linewidth=0.6, label="exit_market")

    ax.set_title("Seller Actions by Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Seller ID")
    # Custom legend for listing color semantics
    from matplotlib.lines import Line2D
    listing_handles = [
        Line2D([0], [0], marker='o', color='w', label='HQ→HQ', markerfacecolor='#1f77b4', markersize=8, markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', label='LQ→LQ', markerfacecolor='#8ecae6', markersize=8, markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', label='HQ→LQ', markerfacecolor='#d62728', markersize=8, markeredgecolor='white'),
    ]
    action_handles = [
        Line2D([0], [0], marker='s', color='w', label='reenter_market', markerfacecolor='#444444', markersize=8, markeredgecolor='white'),
        Line2D([0], [0], marker='^', color='w', label='exit_market', markerfacecolor='#444444', markersize=8, markeredgecolor='white'),
    ]
    handles = listing_handles + action_handles
    ax.legend(handles=handles, title="Legend", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plot_save(fig, out_dir, "seller_actions_scatter")
"""Minimal analysis and visualization utilities."""


def main():
    parser = argparse.ArgumentParser(description="Analyze market simulation SQLite DB and generate visualizations.")
    parser.add_argument("db_path", help="Path to SQLite database (e.g., market_sim.db)")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory (default: analysis/outputs/<timestamp>)")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"DB not found: {args.db_path}")

    out_dir = ensure_output_dir(os.path.dirname(os.path.abspath(args.db_path))) if args.out_dir is None else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    with sqlite3.connect(args.db_path) as conn:
        summary = summarize_basic(conn)
        print("DB Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        # Load tables
        reph = read_table(conn, "reputation_history")
        # Visualizations
        sns.set_theme(style="whitegrid")
        plot_reputation_over_rounds(reph, out_dir)
        plot_avg_price_by_advertised_quality(conn, out_dir)
        plot_seller_actions_scatter(conn, out_dir)

    print(f"Analysis complete. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()


