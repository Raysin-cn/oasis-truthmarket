import sqlite3
import os
import json
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
from config import SimulationConfig


def print_round_statistics(round_num: int, db_path: str = ""):
    """Print current round profit statistics."""
    if not os.path.exists(db_path):
        return
    
    conn = sqlite3.connect(db_path)
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
            JOIN product p ON t.product_id = p.product_id 
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

def clear_market(db_path: str = ""):
    """Update all products on sale status to 'expired' to clear the market."""
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Update all products with status 'on_sale' to 'expired'
        cursor.execute("UPDATE product SET status = 'expired' WHERE status = 'on_sale'")
        conn.commit()
        # Get number of affected rows for debugging
        changes = conn.total_changes
        print(f"Market cleared: {changes} unsold products have been removed from sale.")
    except sqlite3.Error as e:
        print(f"Database error (clear_market): {e}")
    finally:
        conn.close()

def print_simulation_summary(db_path: str = ""):
    """Print summary statistics for the entire simulation."""
    if not os.path.exists(db_path):
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        print(f"\n{'='*50}")
        print("SIMULATION SUMMARY")
        print(f"{'='*50}")
        
        # Overall transaction statistics
        cursor.execute("""
            SELECT COUNT(*), SUM(p.price), AVG(p.price)
            FROM transactions t 
            JOIN product p ON t.product_id = p.product_id
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


# ================== Multi-Run Experiment Management ==================

class ExperimentManager:
    """Multi-run experiment manager"""
    
    def __init__(self, experiment_id: Optional[str] = None):
        """Initialize experiment manager
        
        Args:
            experiment_id: Experiment ID, auto-generated if None
        """
        self.experiment_id = experiment_id or SimulationConfig.get_experiment_id()
        self.paths = SimulationConfig.get_experiment_paths(self.experiment_id)
        self.config = SimulationConfig()
        
        # Create necessary directories
        for path in [self.paths['experiment_dir'], 
                    self.paths['analysis_dir'],
                    self.paths['individual_analysis_dir'],
                    self.paths['aggregated_analysis_dir']]:
            os.makedirs(path, exist_ok=True)
    
    def prepare_experiment(self) -> str:
        """Prepare experiment environment and save configuration
        
        Returns:
            Experiment ID
        """
        print(f"Preparing experiment environment: {self.experiment_id}")
        print(f"Experiment directory: {self.paths['experiment_dir']}")
        print(f"Analysis directory: {self.paths['analysis_dir']}")
        
        # Save configuration file
        SimulationConfig.save_config(self.experiment_id)
        print(f"Configuration saved to: {self.paths['config_file']}")
        
        return self.experiment_id
    
    def get_run_database_path(self, run_id: int) -> str:
        """Get database path for specified run"""
        return SimulationConfig.get_run_db_path(self.experiment_id, run_id)
    
    def cleanup_run_database(self, run_id: int):
        """Clean up database file for specified run (if exists)"""
        db_path = self.get_run_database_path(run_id)
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Database cleaned up: {db_path}")
    
    def collect_run_results(self) -> Dict[str, Any]:
        """Collect statistics from all runs"""
        results = {
            'experiment_id': self.experiment_id,
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'run_details': {}
        }
        
        for run_id in range(1, self.config.RUNS + 1):
            db_path = self.get_run_database_path(run_id)
            if os.path.exists(db_path):
                results['total_runs'] += 1
                try:
                    run_stats = self._get_run_statistics(db_path, run_id)
                    results['run_details'][f'run_{run_id}'] = run_stats
                    results['successful_runs'] += 1
                except Exception as e:
                    print(f"Analysis of Run {run_id} failed: {e}")
                    results['failed_runs'] += 1
                    results['run_details'][f'run_{run_id}'] = {'error': str(e)}
        
        return results
    
    def _get_run_statistics(self, db_path: str, run_id: int) -> Dict[str, Any]:
        """Get statistics for single run"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        stats = {'run_id': run_id, 'db_path': db_path}
        
        try:
            # Total transaction statistics
            cursor.execute("SELECT COUNT(*), SUM(seller_profit), SUM(buyer_utility) FROM transactions")
            total_stats = cursor.fetchone()
            stats.update({
                'total_transactions': total_stats[0] if total_stats else 0,
                'total_seller_profit': total_stats[1] if total_stats and total_stats[1] else 0,
                'total_buyer_utility': total_stats[2] if total_stats and total_stats[2] else 0
            })
            
            # Seller statistics
            cursor.execute("""
                SELECT COUNT(DISTINCT seller_id), AVG(seller_profit), 
                       MAX(seller_profit), MIN(seller_profit)
                FROM transactions
            """)
            seller_stats = cursor.fetchone()
            if seller_stats and seller_stats[0]:
                stats.update({
                    'active_sellers': seller_stats[0],
                    'avg_seller_profit': seller_stats[1],
                    'max_seller_profit': seller_stats[2],
                    'min_seller_profit': seller_stats[3]
                })
            
            # Buyer statistics
            cursor.execute("""
                SELECT COUNT(DISTINCT buyer_id), AVG(buyer_utility),
                       MAX(buyer_utility), MIN(buyer_utility)
                FROM transactions
            """)
            buyer_stats = cursor.fetchone()
            if buyer_stats and buyer_stats[0]:
                stats.update({
                    'active_buyers': buyer_stats[0],
                    'avg_buyer_utility': buyer_stats[1],
                    'max_buyer_utility': buyer_stats[2],
                    'min_buyer_utility': buyer_stats[3]
                })
            
            # Challenge statistics (if applicable)
            cursor.execute("""
                SELECT COUNT(*), SUM(CASE WHEN challenge_reward > 0 THEN 1 ELSE 0 END)
                FROM transactions WHERE is_challenged = 1
            """)
            challenge_stats = cursor.fetchone()
            if challenge_stats and challenge_stats[0]:
                stats.update({
                    'total_challenges': challenge_stats[0],
                    'successful_challenges': challenge_stats[1],
                    'challenge_success_rate': challenge_stats[1] / challenge_stats[0] if challenge_stats[0] > 0 else 0
                })
            
        except sqlite3.Error as e:
            print(f"Database query error (run {run_id}): {e}")
            stats['error'] = str(e)
        finally:
            conn.close()
        
        return stats
    
    def save_experiment_results(self, results: Dict[str, Any]):
        """Save experiment results to JSON file"""
        results_file = os.path.join(self.paths['aggregated_analysis_dir'], 'experiment_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Experiment results saved to: {results_file}")
    
    def print_experiment_summary(self, results: Dict[str, Any]):
        """Print experiment summary"""
        print(f"\n{'='*60}")
        print(f"Experiment Summary: {self.experiment_id}")
        print(f"{'='*60}")
        print(f"Total runs: {results['total_runs']}")
        print(f"Successful runs: {results['successful_runs']}")
        print(f"Failed runs: {results['failed_runs']}")
        
        if results['successful_runs'] > 0:
            # Calculate cross-run statistics
            all_seller_profits = []
            all_buyer_utilities = []
            all_transactions = []
            
            for run_key, run_data in results['run_details'].items():
                if 'error' not in run_data:
                    all_seller_profits.append(run_data.get('total_seller_profit', 0))
                    all_buyer_utilities.append(run_data.get('total_buyer_utility', 0))
                    all_transactions.append(run_data.get('total_transactions', 0))
            
            if all_seller_profits:
                print(f"\nCross-run statistics:")
                print(f"  Average seller total profit: {sum(all_seller_profits)/len(all_seller_profits):.2f}")
                print(f"  Average buyer total utility: {sum(all_buyer_utilities)/len(all_buyer_utilities):.2f}")
                print(f"  Average transaction count: {sum(all_transactions)/len(all_transactions):.1f}")
        
        print(f"{'='*60}")


def setup_single_run_environment(experiment_id: str, run_id: int) -> str:
    """Set up environment for single run
    
    Args:
        experiment_id: Experiment ID
        run_id: Run ID
        
    Returns:
        Database path
    """
    manager = ExperimentManager(experiment_id)
    db_path = manager.get_run_database_path(run_id)
    
    # Set environment variables for oasis to use
    os.environ['MARKET_DB_PATH'] = db_path
    
    return db_path


def print_run_header(experiment_id: str, run_id: int, total_runs: int):
    """Print run start header"""
    print(f"\n{'='*80}")
    print(f"Experiment {experiment_id} - Run {run_id}/{total_runs}")
    print(f"Database: {SimulationConfig.get_run_db_path(experiment_id, run_id)}")
    print(f"{'='*80}")


def print_run_footer(run_id: int, total_runs: int):
    """Print run completion information"""
    print(f"\n{'='*80}")
    print(f"Run {run_id}/{total_runs} completed")
    print(f"{'='*80}")
