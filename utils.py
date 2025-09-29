import sqlite3
import os

DATABASE_PATH = 'market_sim.db'

def print_round_statistics(round_num: int):
    """Print current round profit statistics."""
    if not os.path.exists(DATABASE_PATH):
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        print(f"\n--- Round {round_num} Statistics ---")
        
        # Seller statistics
        cursor.execute("""
            SELECT COUNT(DISTINCT seller_id), SUM(seller_profit), AVG(seller_profit)
            FROM transactions WHERE round_number = ?
        """, (round_num,))
        seller_stats = cursor.fetchone()
        if seller_stats and seller_stats[0] > 0:
            print(f"Sellers: {seller_stats[0]} active, Total Profit: {seller_stats[1]:.2f}, Avg Profit: {seller_stats[2]:.2f}")
        
        # Buyer statistics
        cursor.execute("""
            SELECT COUNT(DISTINCT buyer_id), SUM(buyer_utility), AVG(buyer_utility)
            FROM transactions WHERE round_number = ?
        """, (round_num,))
        buyer_stats = cursor.fetchone()
        if buyer_stats and buyer_stats[0] > 0:
            print(f"Buyers: {buyer_stats[0]} active, Total Utility: {buyer_stats[1]:.2f}, Avg Utility: {buyer_stats[2]:.2f}")
        
        # Transaction statistics
        cursor.execute("""
            SELECT COUNT(*), SUM(p.price), AVG(p.price)
            FROM transactions t 
            JOIN post p ON t.post_id = p.post_id 
            WHERE t.round_number = ?
        """, (round_num,))
        transaction_stats = cursor.fetchone()
        if transaction_stats and transaction_stats[0] > 0:
            print(f"Transactions: {transaction_stats[0]}, Total Value: {transaction_stats[1]:.2f}, Avg Price: {transaction_stats[2]:.2f}")
        
        # Challenge statistics
        cursor.execute("""
            SELECT COUNT(*), SUM(CASE WHEN challenge_reward > 0 THEN 1 ELSE 0 END)
            FROM transactions WHERE round_number = ? AND is_challenged = 1
        """, (round_num,))
        challenge_stats = cursor.fetchone()
        if challenge_stats and challenge_stats[0] > 0:
            print(f"Challenges: {challenge_stats[0]} total, {challenge_stats[1]} successful")
            
    except sqlite3.Error as e:
        print(f"Database error (print_round_statistics): {e}")
    finally:
        conn.close()

def clear_market():
    """Update all products on sale status to 'expired' to clear the market."""
    if not os.path.exists(DATABASE_PATH):
        return
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        # Update all products with status 'on_sale' to 'expired'
        cursor.execute("UPDATE post SET status = 'expired' WHERE status = 'on_sale'")
        conn.commit()
        # Get number of affected rows for debugging
        changes = conn.total_changes
        print(f"Market cleared: {changes} unsold products have been removed from sale.")
    except sqlite3.Error as e:
        print(f"Database error (clear_market): {e}")
    finally:
        conn.close()

def print_simulation_summary():
    """Print summary statistics for the entire simulation."""
    if not os.path.exists(DATABASE_PATH):
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        print(f"\n{'='*50}")
        print("SIMULATION SUMMARY")
        print(f"{'='*50}")
        
        # Overall transaction statistics
        cursor.execute("""
            SELECT COUNT(*), SUM(p.price), AVG(p.price)
            FROM transactions t 
            JOIN post p ON t.post_id = p.post_id
        """)
        total_stats = cursor.fetchone()
        if total_stats and total_stats[0] > 0:
            print(f"Total Transactions: {total_stats[0]}")
            print(f"Total Market Value: {total_stats[1]:.2f}")
            print(f"Average Transaction Value: {total_stats[2]:.2f}")
        
        # Seller performance statistics
        cursor.execute("""
            SELECT COUNT(DISTINCT seller_id), SUM(seller_profit), AVG(seller_profit), 
                   MAX(seller_profit), MIN(seller_profit)
            FROM transactions
        """)
        seller_performance = cursor.fetchone()
        if seller_performance and seller_performance[0] > 0:
            print(f"\nSeller Performance:")
            print(f"  Active Sellers: {seller_performance[0]}")
            print(f"  Total Profit: {seller_performance[1]:.2f}")
            print(f"  Average Profit: {seller_performance[2]:.2f}")
            print(f"  Best Seller Profit: {seller_performance[3]:.2f}")
            print(f"  Worst Seller Profit: {seller_performance[4]:.2f}")
        
        # Buyer performance statistics
        cursor.execute("""
            SELECT COUNT(DISTINCT buyer_id), SUM(buyer_utility), AVG(buyer_utility),
                   MAX(buyer_utility), MIN(buyer_utility)
            FROM transactions
        """)
        buyer_performance = cursor.fetchone()
        if buyer_performance and buyer_performance[0] > 0:
            print(f"\nBuyer Performance:")
            print(f"  Active Buyers: {buyer_performance[0]}")
            print(f"  Total Utility: {buyer_performance[1]:.2f}")
            print(f"  Average Utility: {buyer_performance[2]:.2f}")
            print(f"  Best Buyer Utility: {buyer_performance[3]:.2f}")
            print(f"  Worst Buyer Utility: {buyer_performance[4]:.2f}")
        
        # Challenge statistics
        cursor.execute("""
            SELECT COUNT(*), SUM(CASE WHEN challenge_reward > 0 THEN 1 ELSE 0 END),
                   AVG(CASE WHEN challenge_reward > 0 THEN 1.0 ELSE 0.0 END)
            FROM transactions WHERE is_challenged = 1
        """)
        challenge_performance = cursor.fetchone()
        if challenge_performance and challenge_performance[0] > 0:
            print(f"\nChallenge Performance:")
            print(f"  Total Challenges: {challenge_performance[0]}")
            print(f"  Successful Challenges: {challenge_performance[1]}")
            print(f"  Success Rate: {challenge_performance[2]*100:.1f}%")
        
        print(f"{'='*50}")
            
    except sqlite3.Error as e:
        print(f"Database error (print_simulation_summary): {e}")
    finally:
        conn.close()
