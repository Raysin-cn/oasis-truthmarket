#!/usr/bin/env python3
"""
Buyer Communication Analysis Module
Analyzes posts with useful_info field containing FRAUD/HONEST records from buyer communication phase.
"""

import argparse
import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set font for matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Read table from database"""
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()


def ensure_output_dir(base_dir: str | None = None) -> str:
    """Create output directory with timestamp"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir or os.getcwd(), "analysis", "outputs", f"bc_analysis_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_save(fig: plt.Figure, out_dir: str, name: str) -> None:
    """Save figure to file"""
    path = os.path.join(out_dir, f"{name}.png")
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved: {name}.png")
    except Exception as e:
        print(f"    ✗ Error saving {name}.png: {e}")
        plt.close(fig)


def load_config_from_db_path(db_path: str) -> dict:
    """Load configuration from config.json based on database path"""
    db_dir = os.path.dirname(os.path.abspath(db_path))
    config_file = os.path.join(db_dir, 'config.json')
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    return {}


def get_title_suffix(config: dict) -> str:
    """Generate title suffix based on configuration"""
    parts = []
    market_type = config.get('MARKET_TYPE', 'unknown')
    comm_type = config.get('COMMUNICATION_TYPE', 'none')
    
    if market_type == 'reputation_only':
        parts.append('Reputation-Only')
    elif market_type == 'reputation_warrant':
        parts.append('Reputation+Warrant')
    else:
        parts.append(market_type.replace('_', ' ').title())
    
    if comm_type and comm_type != 'none':
        parts.append(f'{comm_type.title()} Comm.')
    
    return ' | '.join(parts) if parts else ''


def parse_useful_info(posts_df: pd.DataFrame, conn: sqlite3.Connection = None) -> pd.DataFrame:
    """Parse useful_info JSON field and extract FRAUD/HONEST records
    
    Args:
        posts_df: DataFrame containing posts table
        conn: Optional database connection to extract round information via SQL
    """
    if posts_df.empty or 'useful_info' not in posts_df.columns:
        return pd.DataFrame()
    
    # Create a mapping from post_id to round_number using SQL query
    post_to_round = {}
    matched_count = 0
    if conn is not None:
        try:
            cursor = conn.cursor()
            # Strategy: Communication happens after purchase in each round
            # For each post, find the round by matching buyer's transactions
            for idx, post_row in posts_df.iterrows():
                post_id = post_row.get('post_id')
                user_id = post_row.get('user_id')
                post_created_at = post_row.get('created_at')
                
                if pd.isna(post_id) or pd.isna(user_id):
                    continue
                
                try:
                    # Method 1: Find the most recent transaction for this buyer before the post
                    # Note: transactions.created_at is INTEGER (timestamp), post.created_at is DATETIME
                    if not pd.isna(post_created_at):
                        # Convert post datetime to timestamp for comparison
                        try:
                            post_dt = pd.to_datetime(post_created_at)
                            post_timestamp = int(post_dt.timestamp())
                            
                            cursor.execute("""
                                SELECT round_number 
                                FROM transactions 
                                WHERE buyer_id = ? 
                                  AND created_at <= ?
                                ORDER BY created_at DESC 
                                LIMIT 1
                            """, (int(user_id), post_timestamp))
                            
                            result = cursor.fetchone()
                            if result and result[0] is not None:
                                post_to_round[int(post_id)] = int(result[0])
                                matched_count += 1
                                continue
                        except (ValueError, TypeError, OverflowError):
                            # If timestamp conversion fails, try datetime comparison
                            cursor.execute("""
                                SELECT round_number 
                                FROM transactions 
                                WHERE buyer_id = ? 
                                  AND datetime(created_at, 'unixepoch') <= datetime(?)
                                ORDER BY created_at DESC 
                                LIMIT 1
                            """, (int(user_id), str(post_created_at)))
                            
                            result = cursor.fetchone()
                            if result and result[0] is not None:
                                post_to_round[int(post_id)] = int(result[0])
                                matched_count += 1
                                continue
                    
                    # Method 2: If time-based matching fails, use the buyer's most recent transaction round
                    cursor.execute("""
                        SELECT round_number 
                        FROM transactions 
                        WHERE buyer_id = ?
                        ORDER BY round_number DESC, datetime(created_at) DESC
                        LIMIT 1
                    """, (int(user_id),))
                    
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        post_to_round[int(post_id)] = int(result[0])
                        matched_count += 1
                except (sqlite3.Error, ValueError, TypeError) as e:
                    continue
            
            if matched_count > 0:
                print(f"Successfully matched {matched_count}/{len(posts_df)} posts to rounds")
                # Show round distribution
                round_dist = {}
                for round_num in post_to_round.values():
                    round_dist[round_num] = round_dist.get(round_num, 0) + 1
                print(f"Round distribution: {round_dist}")
            else:
                print(f"Warning: Could not match any posts to rounds")
                # Debug: Check if transactions table has data
                try:
                    cursor.execute("SELECT COUNT(*) FROM transactions WHERE buyer_id IS NOT NULL")
                    trans_count = cursor.fetchone()[0]
                    print(f"Debug: Found {trans_count} transactions with buyer_id")
                    
                    # Check sample buyer_id values
                    cursor.execute("SELECT DISTINCT buyer_id FROM transactions LIMIT 5")
                    sample_buyers = [row[0] for row in cursor.fetchall()]
                    print(f"Debug: Sample buyer_ids in transactions: {sample_buyers}")
                    
                    # Check sample user_ids in posts
                    sample_post_users = posts_df['user_id'].dropna().unique()[:5].tolist()
                    print(f"Debug: Sample user_ids in posts: {sample_post_users}")
                except Exception as debug_e:
                    print(f"Debug query failed: {debug_e}")
        except Exception as e:
            print(f"Warning: Could not extract round information from database: {e}")
            import traceback
            traceback.print_exc()
    
    records = []
    for idx, row in posts_df.iterrows():
        useful_info = row.get('useful_info', '')
        if not useful_info or useful_info.strip() == '':
            continue
        
        try:
            info_dict = json.loads(useful_info)
            if isinstance(info_dict, dict) and 'outcome' in info_dict:
                post_id = row.get('post_id')
                record = {
                    'post_id': post_id,
                    'user_id': row.get('user_id'),
                    'content': row.get('content', ''),
                    'created_at': row.get('created_at'),
                    'seller': info_dict.get('seller', ''),
                    'outcome': info_dict.get('outcome', ''),
                    'num_likes': row.get('num_likes', 0),
                    'num_dislikes': row.get('num_dislikes', 0),
                    'round': post_to_round.get(post_id, None),
                }
                records.append(record)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            continue
    
    if not records:
        return pd.DataFrame()
    
    return pd.DataFrame(records)


def plot_fraud_vs_honest_by_round(records_df: pd.DataFrame, out_dir: str, title_suffix: str = '') -> None:
    """Plot FRAUD vs HONEST records by round"""
    if records_df.empty:
        print("  [plot_fraud_vs_honest_by_round] No records to plot")
        return
    
    if 'round' not in records_df.columns:
        print("  [plot_fraud_vs_honest_by_round] No 'round' column in records_df")
        return
    
    # Filter out records without round information
    records_with_round = records_df[records_df['round'].notna()].copy()
    if records_with_round.empty:
        print("  [plot_fraud_vs_honest_by_round] No records with round information found (all rounds are NaN)")
        return
    
    print(f"  [plot_fraud_vs_honest_by_round] Plotting {len(records_with_round)} records with round info")
    
    records_with_round['round'] = records_with_round['round'].astype(int)
    rounds = sorted(records_with_round['round'].unique())
    
    # Count by round and outcome
    round_outcome_counts = records_with_round.groupby(['round', 'outcome']).size().unstack(fill_value=0)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    title = f"FRAUD vs HONEST Records by Round ({title_suffix})" if title_suffix else "FRAUD vs HONEST Records by Round"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Stacked bar chart
    fraud_counts = [round_outcome_counts.loc[r, 'FRAUD'] if 'FRAUD' in round_outcome_counts.columns else 0 for r in rounds]
    honest_counts = [round_outcome_counts.loc[r, 'HONEST'] if 'HONEST' in round_outcome_counts.columns else 0 for r in rounds]
    
    axes[0].bar(rounds, honest_counts, label='HONEST', color='#2ca02c', alpha=0.8, edgecolor='black')
    axes[0].bar(rounds, fraud_counts, bottom=honest_counts, label='FRAUD', color='#d62728', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Number of Records')
    axes[0].set_title('Stacked Bar Chart: FRAUD vs HONEST by Round', fontweight='bold')
    axes[0].set_xticks(rounds)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Line chart showing trends
    axes[1].plot(rounds, fraud_counts, 'o-', label='FRAUD', color='#d62728', linewidth=2, markersize=8)
    axes[1].plot(rounds, honest_counts, 'o-', label='HONEST', color='#2ca02c', linewidth=2, markersize=8)
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Number of Records')
    axes[1].set_title('Trend Analysis: FRAUD vs HONEST Over Rounds', fontweight='bold')
    axes[1].set_xticks(rounds)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plot_save(fig, out_dir, "fraud_vs_honest_by_round")


def plot_seller_fraud_by_round(records_df: pd.DataFrame, out_dir: str, title_suffix: str = '') -> None:
    """Plot FRAUD records by seller and round (heatmap)"""
    if records_df.empty:
        print("  [plot_seller_fraud_by_round] No records to plot")
        return
    
    if 'round' not in records_df.columns:
        print("  [plot_seller_fraud_by_round] No 'round' column in records_df")
        return
    
    records_with_round = records_df[records_df['round'].notna()].copy()
    fraud_records = records_with_round[records_with_round['outcome'] == 'FRAUD'].copy()
    
    if fraud_records.empty:
        print("  [plot_seller_fraud_by_round] No FRAUD records with round information found")
        return
    
    print(f"  [plot_seller_fraud_by_round] Plotting {len(fraud_records)} FRAUD records")
    
    fraud_records['round'] = fraud_records['round'].astype(int)
    
    # Create pivot table: seller vs round
    seller_round_counts = fraud_records.groupby(['seller', 'round']).size().unstack(fill_value=0)
    
    if seller_round_counts.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(seller_round_counts) * 0.5)))
    title = f"FRAUD Records by Seller and Round ({title_suffix})" if title_suffix else "FRAUD Records by Seller and Round"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Create heatmap
    sns.heatmap(seller_round_counts, annot=True, fmt='d', cmap='Reds', 
                cbar_kws={'label': 'Number of FRAUD Records'}, ax=ax, linewidths=0.5)
    ax.set_xlabel('Round', fontweight='bold')
    ax.set_ylabel('Seller ID', fontweight='bold')
    ax.set_title('Heatmap: FRAUD Records Distribution', fontweight='bold', pad=20)
    
    plot_save(fig, out_dir, "seller_fraud_by_round")


def plot_communication_activity_by_round(records_df: pd.DataFrame, out_dir: str, title_suffix: str = '') -> None:
    """Plot communication activity (total records) by round"""
    if records_df.empty:
        print("  [plot_communication_activity_by_round] No records to plot")
        return
    
    if 'round' not in records_df.columns:
        print("  [plot_communication_activity_by_round] No 'round' column in records_df")
        return
    
    records_with_round = records_df[records_df['round'].notna()].copy()
    if records_with_round.empty:
        print("  [plot_communication_activity_by_round] No records with round information found")
        return
    
    print(f"  [plot_communication_activity_by_round] Plotting {len(records_with_round)} records with round info")
    
    records_with_round['round'] = records_with_round['round'].astype(int)
    
    # Count total records per round
    round_counts = records_with_round.groupby('round').size()
    rounds = sorted(round_counts.index)
    counts = [round_counts[r] for r in rounds]
    
    # Count unique buyers per round
    unique_buyers_per_round = records_with_round.groupby('round')['user_id'].nunique()
    buyer_counts = [unique_buyers_per_round[r] for r in rounds]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title = f"Communication Activity by Round ({title_suffix})" if title_suffix else "Communication Activity by Round"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Total records per round
    bars1 = axes[0].bar(rounds, counts, color='#1f77b4', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Total Number of Records')
    axes[0].set_title('Total Communication Records per Round', fontweight='bold')
    axes[0].set_xticks(rounds)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, counts):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Unique buyers per round
    bars2 = axes[1].bar(rounds, buyer_counts, color='#ff7f0e', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Number of Unique Buyers')
    axes[1].set_title('Active Buyers per Round', fontweight='bold')
    axes[1].set_xticks(rounds)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars2, buyer_counts):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plot_save(fig, out_dir, "communication_activity_by_round")


def generate_summary_statistics(records_df: pd.DataFrame, out_dir: str) -> Dict[str, Any]:
    """Generate and save summary statistics"""
    if records_df.empty:
        return {}
    
    stats = {
        'total_records': len(records_df),
        'fraud_count': len(records_df[records_df['outcome'] == 'FRAUD']),
        'honest_count': len(records_df[records_df['outcome'] == 'HONEST']),
        'unique_sellers_mentioned': records_df['seller'].nunique(),
        'unique_buyers': records_df['user_id'].nunique(),
        'sellers_with_fraud': records_df[records_df['outcome'] == 'FRAUD']['seller'].nunique(),
        'sellers_with_honest': records_df[records_df['outcome'] == 'HONEST']['seller'].nunique(),
    }
    
    if stats['total_records'] > 0:
        stats['fraud_percentage'] = (stats['fraud_count'] / stats['total_records']) * 100
        stats['honest_percentage'] = (stats['honest_count'] / stats['total_records']) * 100
    
    # Add round-based statistics if available
    if 'round' in records_df.columns:
        records_with_round = records_df[records_df['round'].notna()]
        if not records_with_round.empty:
            stats['records_with_round'] = len(records_with_round)
            stats['rounds_covered'] = sorted(records_with_round['round'].unique().tolist())
            stats['fraud_by_round'] = records_with_round[records_with_round['outcome'] == 'FRAUD'].groupby('round').size().to_dict()
            stats['honest_by_round'] = records_with_round[records_with_round['outcome'] == 'HONEST'].groupby('round').size().to_dict()
    
    # Save to JSON
    stats_file = os.path.join(out_dir, 'summary_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*60)
    print("Buyer Communication Analysis Summary")
    print("="*60)
    print(f"Total Records: {stats['total_records']}")
    print(f"FRAUD Records: {stats['fraud_count']} ({stats.get('fraud_percentage', 0):.1f}%)")
    print(f"HONEST Records: {stats['honest_count']} ({stats.get('honest_percentage', 0):.1f}%)")
    print(f"Unique Sellers Mentioned: {stats['unique_sellers_mentioned']}")
    print(f"Unique Buyers Sharing: {stats['unique_buyers']}")
    print(f"Sellers with FRAUD Records: {stats['sellers_with_fraud']}")
    print(f"Sellers with HONEST Records: {stats['sellers_with_honest']}")
    
    if 'rounds_covered' in stats:
        print(f"\nRound-based Statistics:")
        print(f"  Records with Round Info: {stats.get('records_with_round', 0)}")
        print(f"  Rounds Covered: {stats['rounds_covered']}")
        if stats.get('fraud_by_round'):
            print(f"  FRAUD by Round: {stats['fraud_by_round']}")
        if stats.get('honest_by_round'):
            print(f"  HONEST by Round: {stats['honest_by_round']}")
    
    print("="*60 + "\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze buyer communication posts with useful_info field (FRAUD/HONEST records)"
    )
    parser.add_argument("db_path", help="Path to SQLite database (e.g., market_sim.db)")
    parser.add_argument("--out", dest="out_dir", default=None, 
                       help="Output directory (default: analysis/outputs/bc_analysis_<timestamp>)")
    args = parser.parse_args()
    
    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"DB not found: {args.db_path}")
    
    out_dir = ensure_output_dir(os.path.dirname(os.path.abspath(args.db_path))) if args.out_dir is None else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Load configuration
    config = load_config_from_db_path(args.db_path)
    title_suffix = get_title_suffix(config)
    
    with sqlite3.connect(args.db_path) as conn:
        # Load posts table
        posts_df = read_table(conn, "post")
        
        if posts_df.empty:
            print("No posts found in database")
            return
        
        print(f"Found {len(posts_df)} posts in database")
        
        # Load transactions table for reference
        transactions_df = read_table(conn, "transactions")
        print(f"Found {len(transactions_df)} transactions in database")
        
        # Parse useful_info with round information from database
        records_df = parse_useful_info(posts_df, conn)
        
        if records_df.empty:
            print("No valid useful_info records found (FRAUD/HONEST)")
            print("Note: useful_info should contain JSON like: {\"seller\": \"<id>\", \"outcome\": \"FRAUD\" or \"HONEST\"}")
            return
        
        print(f"Parsed {len(records_df)} valid records from useful_info field")
        
        # Check if we have round information
        records_with_round = records_df[records_df['round'].notna()]
        print(f"Records with round info: {len(records_with_round)} / {len(records_df)}")
        
        if records_with_round.empty:
            print("Warning: No round information found in records.")
            print("Attempting to generate visualizations anyway (may show empty charts)...")
        else:
            print(f"Round distribution: {records_with_round['round'].value_counts().to_dict()}")
        
        # Generate visualizations (only round-based)
        print("Generating round-based visualizations...")
        sns.set_theme(style="whitegrid")
        
        plot_fraud_vs_honest_by_round(records_df, out_dir, title_suffix)
        plot_seller_fraud_by_round(records_df, out_dir, title_suffix)
        plot_communication_activity_by_round(records_df, out_dir, title_suffix)
        
        print("Visualization generation completed. Check output directory for images.")
        
        # Generate summary statistics
        generate_summary_statistics(records_df, out_dir)
        
        # Save raw data
        csv_path = os.path.join(out_dir, 'parsed_records.csv')
        records_df.to_csv(csv_path, index=False)
        print(f"Raw data saved to: {csv_path}")
    
    print(f"\nAnalysis complete. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()

