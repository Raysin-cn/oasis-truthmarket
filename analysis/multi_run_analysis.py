#!/usr/bin/env python3
"""
Multi-run Aggregated Analysis Module
Used for analyzing results of repeated experiments, generating aggregated statistics and visualizations.
"""

import os
import sqlite3
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SimulationConfig
from analysis.analyze_market import read_table, ensure_output_dir, plot_save


class MultiRunAnalyzer:
    """Multi-run Analyzer"""
    
    def __init__(self, experiment_id: str):
        """Initialize the analyzer
        
        Args:
            experiment_id: Experiment ID
        """
        self.experiment_id = experiment_id
        self.paths = SimulationConfig.get_experiment_paths(experiment_id)
        self.run_data = {}
        self.aggregated_data = {}
        
    def load_experiment_data(self):
        """Load experiment data"""
        print(f"Loading experiment data: {self.experiment_id}")
        
        # Find all run database files
        experiment_dir = self.paths['experiment_dir']
        if not os.path.exists(experiment_dir):
            print(f"Experiment directory does not exist: {experiment_dir}")
            return
        
        db_files = [f for f in os.listdir(experiment_dir) if f.startswith('run_') and f.endswith('.db')]
        print(f"Found {len(db_files)} run database(s)")
        
        for db_file in db_files:
            run_id = int(db_file.replace('run_', '').replace('.db', ''))
            db_path = os.path.join(experiment_dir, db_file)
            
            try:
                run_data = self._load_single_run_data(db_path, run_id)
                self.run_data[run_id] = run_data
                print(f"Loaded Run {run_id}")
            except Exception as e:
                print(f"Failed to load Run {run_id}: {e}")
        
        print(f"Successfully loaded data for {len(self.run_data)} run(s)")
    
    def _load_single_run_data(self, db_path: str, run_id: int) -> Dict[str, pd.DataFrame]:
        """Load data for a single run"""
        conn = sqlite3.connect(db_path)
        
        data = {
            'run_id': run_id,
            'transactions': read_table(conn, 'transactions'),
            'user': read_table(conn, 'user'),
            'post': read_table(conn, 'post'),
            'reputation_history': read_table(conn, 'reputation_history'),
            'trace': read_table(conn, 'trace')
        }
        
        conn.close()
        return data
    
    def generate_aggregated_statistics(self) -> Dict[str, Any]:
        """Generate aggregated statistics"""
        stats = {
            'experiment_id': self.experiment_id,
            'total_runs': len(self.run_data),
            'summary_stats': {},
            'round_stats': {},
            'cross_run_variance': {}
        }
        
        if not self.run_data:
            return stats
        
        # Aggregate basic statistics
        total_transactions = []
        total_seller_profits = []
        total_buyer_utilities = []
        seller_profit_by_run = {}
        buyer_utility_by_run = {}
        
        for run_id, data in self.run_data.items():
            transactions = data['transactions']
            if not transactions.empty:
                total_trans = len(transactions)
                total_seller_profit = transactions['seller_profit'].sum()
                total_buyer_utility = transactions['buyer_utility'].sum()
                
                total_transactions.append(total_trans)
                total_seller_profits.append(total_seller_profit)
                total_buyer_utilities.append(total_buyer_utility)
                
                seller_profit_by_run[run_id] = total_seller_profit
                buyer_utility_by_run[run_id] = total_buyer_utility
        
        # Compute cross-run statistics
        if total_transactions:
            stats['summary_stats'] = {
                'avg_transactions_per_run': np.mean(total_transactions),
                'std_transactions_per_run': np.std(total_transactions),
                'avg_seller_profit_per_run': np.mean(total_seller_profits),
                'std_seller_profit_per_run': np.std(total_seller_profits),
                'avg_buyer_utility_per_run': np.mean(total_buyer_utilities),
                'std_buyer_utility_per_run': np.std(total_buyer_utilities),
                'total_transactions_all_runs': sum(total_transactions),
                'total_seller_profit_all_runs': sum(total_seller_profits),
                'total_buyer_utility_all_runs': sum(total_buyer_utilities)
            }
        
        # Aggregate statistics by round
        round_stats = {}
        for round_num in range(1, SimulationConfig.SIMULATION_ROUNDS + 1):
            round_transactions = []
            round_seller_profits = []
            round_buyer_utilities = []
            
            for run_id, data in self.run_data.items():
                transactions = data['transactions']
                if not transactions.empty and 'round_number' in transactions.columns:
                    round_trans = transactions[transactions['round_number'] == round_num]
                    if not round_trans.empty:
                        round_transactions.append(len(round_trans))
                        round_seller_profits.append(round_trans['seller_profit'].sum())
                        round_buyer_utilities.append(round_trans['buyer_utility'].sum())
            
            if round_transactions:
                round_stats[round_num] = {
                    'avg_transactions': np.mean(round_transactions),
                    'std_transactions': np.std(round_transactions),
                    'avg_seller_profit': np.mean(round_seller_profits),
                    'std_seller_profit': np.std(round_seller_profits),
                    'avg_buyer_utility': np.mean(round_buyer_utilities),
                    'std_buyer_utility': np.std(round_buyer_utilities)
                }
        
        stats['round_stats'] = round_stats
        stats['seller_profit_by_run'] = seller_profit_by_run
        stats['buyer_utility_by_run'] = buyer_utility_by_run
        
        self.aggregated_data = stats
        return stats
    
    def plot_cross_run_comparison(self, out_dir: str):
        """Plot cross-run comparison chart"""
        if not self.run_data:
            return
        
        # Prepare data
        run_ids = []
        seller_profits = []
        buyer_utilities = []
        transaction_counts = []
        
        for run_id, data in self.run_data.items():
            transactions = data['transactions']
            if not transactions.empty:
                run_ids.append(run_id)
                seller_profits.append(transactions['seller_profit'].sum())
                buyer_utilities.append(transactions['buyer_utility'].sum())
                transaction_counts.append(len(transactions))
        
        if not run_ids:
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Cross-Run Comparison Analysis - Experiment {self.experiment_id}', fontsize=14)
        
        # 1. Seller total profit comparison
        axes[0, 0].bar(run_ids, seller_profits, alpha=0.7, color='skyblue')
        axes[0, 0].axhline(y=np.mean(seller_profits), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(seller_profits):.2f}')
        axes[0, 0].set_title('Total Seller Profits')
        axes[0, 0].set_xlabel('Run ID')
        axes[0, 0].set_ylabel('Total Profit')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Buyer total utility comparison
        axes[0, 1].bar(run_ids, buyer_utilities, alpha=0.7, color='lightgreen')
        axes[0, 1].axhline(y=np.mean(buyer_utilities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(buyer_utilities):.2f}')
        axes[0, 1].set_title('Total Buyer Utilities')
        axes[0, 1].set_xlabel('Run ID')
        axes[0, 1].set_ylabel('Total Utility')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Transaction count comparison
        axes[1, 0].bar(run_ids, transaction_counts, alpha=0.7, color='orange')
        axes[1, 0].axhline(y=np.mean(transaction_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(transaction_counts):.1f}')
        axes[1, 0].set_title('Transaction Counts')
        axes[1, 0].set_xlabel('Run ID')
        axes[1, 0].set_ylabel('Number of Transactions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Profit vs Utility scatter plot
        axes[1, 1].scatter(seller_profits, buyer_utilities, alpha=0.7, s=60)
        axes[1, 1].set_title('Seller Profits vs Buyer Utilities')
        axes[1, 1].set_xlabel('Total Seller Profit')
        axes[1, 1].set_ylabel('Total Buyer Utility')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add run ID labels
        for i, run_id in enumerate(run_ids):
            axes[1, 1].annotate(f'R{run_id}', (seller_profits[i], buyer_utilities[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plot_save(fig, out_dir, 'cross_run_comparison')
    
    def plot_round_progression(self, out_dir: str):
        """Plot round progression chart"""
        if not self.run_data or not self.aggregated_data.get('round_stats'):
            return
        
        round_stats = self.aggregated_data['round_stats']
        rounds = sorted(round_stats.keys())
        
        # Prepare data
        avg_seller_profits = [round_stats[r]['avg_seller_profit'] for r in rounds]
        std_seller_profits = [round_stats[r]['std_seller_profit'] for r in rounds]
        avg_buyer_utilities = [round_stats[r]['avg_buyer_utility'] for r in rounds]
        std_buyer_utilities = [round_stats[r]['std_buyer_utility'] for r in rounds]
        avg_transactions = [round_stats[r]['avg_transactions'] for r in rounds]
        std_transactions = [round_stats[r]['std_transactions'] for r in rounds]
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Round Progression Analysis - Experiment {self.experiment_id}', fontsize=14)
        
        # 1. Seller profit progression
        axes[0].errorbar(rounds, avg_seller_profits, yerr=std_seller_profits, 
                        fmt='-o', capsize=5, linewidth=2, markersize=6)
        axes[0].set_title('Seller Profit Progression')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Average Profit ± Std Dev')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Buyer utility progression
        axes[1].errorbar(rounds, avg_buyer_utilities, yerr=std_buyer_utilities,
                        fmt='-o', capsize=5, linewidth=2, markersize=6, color='green')
        axes[1].set_title('Buyer Utility Progression')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Average Utility ± Std Dev')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Transaction activity progression
        axes[2].errorbar(rounds, avg_transactions, yerr=std_transactions,
                        fmt='-o', capsize=5, linewidth=2, markersize=6, color='orange')
        axes[2].set_title('Transaction Activity Progression')
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('Average Transactions ± Std Dev')
        axes[2].grid(True, alpha=0.3)
        
        plot_save(fig, out_dir, 'round_progression')
    
    def plot_distribution_analysis(self, out_dir: str):
        """Plot distribution analysis charts"""
        if not self.run_data:
            return
        
        # Gather data from all runs
        all_seller_profits = []
        all_buyer_utilities = []
        all_transaction_counts = []
        
        for run_id, data in self.run_data.items():
            transactions = data['transactions']
            if not transactions.empty:
                all_seller_profits.append(transactions['seller_profit'].sum())
                all_buyer_utilities.append(transactions['buyer_utility'].sum())
                all_transaction_counts.append(len(transactions))
        
        if not all_seller_profits:
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Distribution Analysis - Experiment {self.experiment_id}', fontsize=14)
        
        # First row: Histograms
        # Seller profit distribution
        axes[0, 0].hist(all_seller_profits, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(all_seller_profits), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_seller_profits):.2f}')
        axes[0, 0].set_title('Total Seller Profits Distribution')
        axes[0, 0].set_xlabel('Total Profit')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Buyer utility distribution
        axes[0, 1].hist(all_buyer_utilities, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(np.mean(all_buyer_utilities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_buyer_utilities):.2f}')
        axes[0, 1].set_title('Total Buyer Utilities Distribution')
        axes[0, 1].set_xlabel('Total Utility')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Transaction count distribution
        axes[0, 2].hist(all_transaction_counts, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].axvline(np.mean(all_transaction_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_transaction_counts):.1f}')
        axes[0, 2].set_title('Transaction Counts Distribution')
        axes[0, 2].set_xlabel('Number of Transactions')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Second row: Box plots
        axes[1, 0].boxplot(all_seller_profits, labels=['Seller Profits'])
        axes[1, 0].set_title('Seller Profits Box Plot')
        axes[1, 0].set_ylabel('Total Profit')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].boxplot(all_buyer_utilities, labels=['Buyer Utilities'])
        axes[1, 1].set_title('Buyer Utilities Box Plot')
        axes[1, 1].set_ylabel('Total Utility')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].boxplot(all_transaction_counts, labels=['Transaction Counts'])
        axes[1, 2].set_title('Transaction Counts Box Plot')
        axes[1, 2].set_ylabel('Number of Transactions')
        axes[1, 2].grid(True, alpha=0.3)
        
        plot_save(fig, out_dir, 'distribution_analysis')
    
    def _calculate_seller_deception_stats(self) -> Dict[str, Any]:
        """Calculate statistics for seller deception behavior
        
        Deception behavior definition: seller advertises HQ but actually provides LQ products
        
        Returns:
            A dictionary containing deception statistics
        """
        stats = {
            'total_deceptions_by_run': {},
            'deception_rate_by_run': {},
            'seller_deception_count': {},
            'deception_pattern_distribution': {},
            'overall_deception_stats': {}
        }
        
        for run_id, data in self.run_data.items():
            posts = data.get('post')
            if posts is None or posts.empty:
                continue
            
            # Count number of deceptions (advertised HQ but actual LQ)
            deceptions = posts[
                (posts['advertised_quality'] == 'HQ') & 
                (posts['true_quality'] == 'LQ')
            ]
            total_deceptions = len(deceptions)
            
            # Calculate deception rate
            hq_advertised = len(posts[posts['advertised_quality'] == 'HQ'])
            deception_rate = total_deceptions / hq_advertised if hq_advertised > 0 else 0
            
            stats['total_deceptions_by_run'][run_id] = total_deceptions
            stats['deception_rate_by_run'][run_id] = deception_rate
            
            # Count deceptions by seller
            if not deceptions.empty:
                seller_deceptions = deceptions.groupby('user_id').size().to_dict()
                for seller_id, count in seller_deceptions.items():
                    if seller_id not in stats['seller_deception_count']:
                        stats['seller_deception_count'][seller_id] = []
                    stats['seller_deception_count'][seller_id].append(count)
        
        # Compute cross-run aggregated statistics
        if stats['total_deceptions_by_run']:
            total_deceptions_list = list(stats['total_deceptions_by_run'].values())
            deception_rates_list = list(stats['deception_rate_by_run'].values())
            
            stats['overall_deception_stats'] = {
                'avg_deceptions_per_run': np.mean(total_deceptions_list),
                'std_deceptions_per_run': np.std(total_deceptions_list),
                'max_deceptions_per_run': max(total_deceptions_list),
                'min_deceptions_per_run': min(total_deceptions_list),
                'avg_deception_rate': np.mean(deception_rates_list),
                'std_deception_rate': np.std(deception_rates_list),
                'total_deceptions_all_runs': sum(total_deceptions_list)
            }
            
            # Deception pattern distribution statistics
            for seller_id, counts in stats['seller_deception_count'].items():
                avg_deceptions = np.mean(counts)
                if avg_deceptions not in stats['deception_pattern_distribution']:
                    stats['deception_pattern_distribution'][avg_deceptions] = 0
                stats['deception_pattern_distribution'][avg_deceptions] += 1
        
        return stats
    
    def plot_seller_deception_analysis(self, out_dir: str):
        """Plot seller deception behavior analysis chart
        
        Includes:
        1. Cross-run deception count comparison
        2. Deception rate distribution
        3. Deception frequency by round
        4. Seller deception frequency distribution
        """
        if not self.run_data:
            print("No data available to plot deception analysis chart")
            return
        
        # Calculate deception statistics
        deception_stats = self._calculate_seller_deception_stats()
        
        if not deception_stats['total_deceptions_by_run']:
            print("No deception data found")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Seller Deception Behavior Analysis - Experiment {self.experiment_id}', 
                    fontsize=14, fontweight='bold')
        
        run_ids = sorted(deception_stats['total_deceptions_by_run'].keys())
        total_deceptions = [deception_stats['total_deceptions_by_run'][r] for r in run_ids]
        deception_rates = [deception_stats['deception_rate_by_run'][r] for r in run_ids]
        
        # 1. Deception count comparison bar chart
        colors = ['red' if d > 0 else 'green' for d in total_deceptions]
        axes[0, 0].bar(run_ids, total_deceptions, alpha=0.7, color=colors, edgecolor='black')
        axes[0, 0].axhline(y=np.mean(total_deceptions), color='blue', linestyle='--', 
                          label=f'Mean: {np.mean(total_deceptions):.1f}')
        axes[0, 0].set_title('Total Deception Cases per Run', fontweight='bold')
        axes[0, 0].set_xlabel('Run ID')
        axes[0, 0].set_ylabel('Number of Deceptions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Deception rate histogram
        axes[0, 1].hist(deception_rates, bins=15, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 1].axvline(np.mean(deception_rates), color='red', linestyle='--',
                          label=f'Mean: {np.mean(deception_rates):.2%}')
        axes[0, 1].set_title('Deception Rate Distribution (HQ advertised but LQ delivered)', 
                            fontweight='bold')
        axes[0, 1].set_xlabel('Deception Rate')
        axes[0, 1].set_ylabel('Frequency (Number of Runs)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Deception counts boxplot and scatter
        axes[1, 0].boxplot(total_deceptions, labels=['Deceptions per Run'], patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1, 0].scatter(range(len(run_ids)), total_deceptions, alpha=0.6, s=50, color='darkblue')
        axes[1, 0].set_title('Deception Cases Box Plot', fontweight='bold')
        axes[1, 0].set_ylabel('Number of Deceptions')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Statistics panel
        overall_stats = deception_stats['overall_deception_stats']
        axes[1, 1].axis('off')
        
        stats_text = f"""
        Overall Deception Statistics:
        ────────────────────────────
        
        ✓ Total Runs: {len(run_ids)}
        ✓ Total Deceptions: {overall_stats['total_deceptions_all_runs']:.0f}
        
        ✓ Average Deceptions/Run: {overall_stats['avg_deceptions_per_run']:.2f} ± {overall_stats['std_deceptions_per_run']:.2f}
        ✓ Max Deceptions: {overall_stats['max_deceptions_per_run']:.0f}
        ✓ Min Deceptions: {overall_stats['min_deceptions_per_run']:.0f}
        
        ✓ Average Deception Rate: {overall_stats['avg_deception_rate']:.2%}
        ✓ Std Dev of Rate: {overall_stats['std_deception_rate']:.2%}
        
        ✓ Deception Rate Range:
          - Min: {min(deception_rates):.2%}
          - Max: {max(deception_rates):.2%}
        """
        
        axes[1, 1].text(0.1, 0.95, stats_text, transform=axes[1, 1].transAxes,
                       fontfamily='monospace', fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plot_save(fig, out_dir, 'seller_deception_analysis')
        
        # Save detailed statistics
        self._save_deception_statistics(deception_stats, out_dir)
    
    def _save_deception_statistics(self, deception_stats: Dict[str, Any], out_dir: str):
        """Save deception statistics to JSON file"""
        stats_file = os.path.join(out_dir, 'deception_statistics.json')
        
        # Convert to serializable format
        output_stats = {
            'experiment_id': self.experiment_id,
            'total_runs': len(self.run_data),
            'deceptions_per_run': deception_stats['total_deceptions_by_run'],
            'deception_rates_per_run': deception_stats['deception_rate_by_run'],
            'overall_statistics': deception_stats['overall_deception_stats']
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(output_stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Deception statistics details saved to: {stats_file}")
    
    def generate_individual_run_analysis(self):
        """Generate individual analysis for each run"""
        individual_dir = self.paths['individual_analysis_dir']
        
        for run_id, data in self.run_data.items():
            run_output_dir = os.path.join(individual_dir, f'run_{run_id}')
            os.makedirs(run_output_dir, exist_ok=True)
            
            try:
                # Use existing analysis module to generate analysis for each run
                db_path = SimulationConfig.get_run_db_path(self.experiment_id, run_id)
                
                # Use analyze_market.py functionality
                import subprocess
                result = subprocess.run([
                    'python', 'analysis/analyze_market.py', 
                    db_path, '--out', run_output_dir
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Run {run_id} analysis complete")
                else:
                    print(f"Run {run_id} analysis failed: {result.stderr}")
                    
            except Exception as e:
                print(f"Error during analysis of Run {run_id}: {e}")
    
    def save_aggregated_results(self):
        """Save aggregated results"""
        if not self.aggregated_data:
            return
        
        results_file = os.path.join(self.paths['aggregated_analysis_dir'], 'aggregated_statistics.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.aggregated_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Aggregated statistics saved to: {results_file}")


async def analyze_experiment(experiment_id: str):
    """Analyze the entire experiment results
    
    Args:
        experiment_id: Experiment ID
    """
    print(f"Start analyzing experiment: {experiment_id}")
    
    analyzer = MultiRunAnalyzer(experiment_id)
    
    # Load data
    analyzer.load_experiment_data()
    
    if not analyzer.run_data:
        print("No data found for analysis")
        return
    
    # Generate aggregated statistics
    print("Generating aggregated statistics...")
    stats = analyzer.generate_aggregated_statistics()
    
    # Save results
    analyzer.save_aggregated_results()
    
    # Generate aggregated visualizations
    aggregated_dir = analyzer.paths['aggregated_analysis_dir']
    print(f"Generating aggregate visualizations to: {aggregated_dir}")
    
    sns.set_theme(style="whitegrid")
    analyzer.plot_cross_run_comparison(aggregated_dir)
    analyzer.plot_round_progression(aggregated_dir)
    analyzer.plot_distribution_analysis(aggregated_dir)
    analyzer.plot_seller_deception_analysis(aggregated_dir)
    
    # Generate individual run analysis
    print("Generating individual run analysis...")
    analyzer.generate_individual_run_analysis()
    
    print(f"Experiment analysis complete! Results saved in: {analyzer.paths['analysis_dir']}")


def main():
    """Command-line entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze multi-run experiment results')
    parser.add_argument('--experiment_id', help='Experiment ID', default='experiment_20251027_154200')
    args = parser.parse_args()
    
    asyncio.run(analyze_experiment(args.experiment_id))


if __name__ == "__main__":
    main()