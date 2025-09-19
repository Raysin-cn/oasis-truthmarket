import sqlite3
import os

DATABASE_PATH = 'market_sim.db'

def print_round_statistics(round_num: int):
    """打印当前回合的收益统计信息。"""
    if not os.path.exists(DATABASE_PATH):
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        print(f"\n--- Round {round_num} Statistics ---")
        
        # 卖家统计
        cursor.execute("""
            SELECT COUNT(DISTINCT seller_id), SUM(seller_profit), AVG(seller_profit)
            FROM transactions WHERE round_number = ?
        """, (round_num,))
        seller_stats = cursor.fetchone()
        if seller_stats and seller_stats[0] > 0:
            print(f"Sellers: {seller_stats[0]} active, Total Profit: {seller_stats[1]:.2f}, Avg Profit: {seller_stats[2]:.2f}")
        
        # 买家统计
        cursor.execute("""
            SELECT COUNT(DISTINCT buyer_id), SUM(buyer_utility), AVG(buyer_utility)
            FROM transactions WHERE round_number = ?
        """, (round_num,))
        buyer_stats = cursor.fetchone()
        if buyer_stats and buyer_stats[0] > 0:
            print(f"Buyers: {buyer_stats[0]} active, Total Utility: {buyer_stats[1]:.2f}, Avg Utility: {buyer_stats[2]:.2f}")
        
        # 交易统计
        cursor.execute("""
            SELECT COUNT(*), SUM(p.price), AVG(p.price)
            FROM transactions t 
            JOIN post p ON t.post_id = p.post_id 
            WHERE t.round_number = ?
        """, (round_num,))
        transaction_stats = cursor.fetchone()
        if transaction_stats and transaction_stats[0] > 0:
            print(f"Transactions: {transaction_stats[0]}, Total Value: {transaction_stats[1]:.2f}, Avg Price: {transaction_stats[2]:.2f}")
        
        # 挑战统计
        cursor.execute("""
            SELECT COUNT(*), SUM(CASE WHEN challenge_reward > 0 THEN 1 ELSE 0 END)
            FROM transactions WHERE round_number = ? AND is_challenged = 1
        """, (round_num,))
        challenge_stats = cursor.fetchone()
        if challenge_stats and challenge_stats[0] > 0:
            print(f"Challenges: {challenge_stats[0]} total, {challenge_stats[1]} successful")
            
    except sqlite3.Error as e:
        print(f"数据库错误 (print_round_statistics): {e}")
    finally:
        conn.close()

def clear_market():
    """将所有在售商品的状态更新为'expired'，实现市场清空。"""
    if not os.path.exists(DATABASE_PATH):
        return
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        # 将所有状态为 'on_sale' 的商品更新为 'expired'
        cursor.execute("UPDATE post SET status = 'expired' WHERE status = 'on_sale'")
        conn.commit()
        # 获取被影响的行数，用于调试
        changes = conn.total_changes
        print(f"Market cleared: {changes} unsold products have been removed from sale.")
    except sqlite3.Error as e:
        print(f"数据库错误 (clear_market): {e}")
    finally:
        conn.close()

def print_simulation_summary():
    """打印整个模拟的总结统计信息。"""
    if not os.path.exists(DATABASE_PATH):
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    try:
        print(f"\n{'='*50}")
        print("SIMULATION SUMMARY")
        print(f"{'='*50}")
        
        # 总体交易统计
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
        
        # 卖家表现统计
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
        
        # 买家表现统计
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
        
        # 挑战统计
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
        print(f"数据库错误 (print_simulation_summary): {e}")
    finally:
        conn.close()
