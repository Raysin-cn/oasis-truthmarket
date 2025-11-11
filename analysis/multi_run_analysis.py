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
    
    def _prepare_cross_run_data(self) -> Dict[str, List]:
        """准备 cross-run 比较分析所需的数据
        
        Returns:
            包含所有分析数据的字典，如果没有数据则返回空字典
        """
        if not self.run_data:
            return {}
        
        # Prepare data
        run_ids = []
        seller_profits = []
        honest_profits = []  # 诚实收益（绿色）
        dishonest_profits = []  # 不诚实收益（红色）
        buyer_utilities = []
        honest_buyer_utilities = []  # 诚实交易的 buyer utility（绿色）
        dishonest_buyer_utilities = []  # 不诚实交易的 buyer utility（红色）
        transaction_counts = []
        honest_transaction_counts = []  # 诚实交易数量（绿色）
        dishonest_transaction_counts = []  # 不诚实交易数量（红色）
        
        for run_id, data in self.run_data.items():
            transactions = data['transactions']
            posts = data.get('post', pd.DataFrame())
            
            if not transactions.empty:
                run_ids.append(run_id)
                
                # 关联 transactions 和 post 表来区分 honest 和 dishonest 收益
                if not posts.empty and 'post_id' in transactions.columns:
                    # 合并数据以获取 advertised_quality 和 true_quality
                    merged = transactions.merge(
                        posts[['post_id', 'advertised_quality', 'true_quality']],
                        on='post_id',
                        how='left'
                    )
                    
                    # 识别 dishonest 交易：advertised_quality == 'HQ' 且 true_quality == 'LQ'
                    # 处理可能的 NaN 值
                    dishonest_mask = (
                        (merged['advertised_quality'] == 'HQ') & 
                        (merged['true_quality'] == 'LQ')
                    )
                    
                    # 计算 dishonest 和 honest seller 收益（使用 fillna(0) 处理缺失值）
                    dishonest_profit = merged[dishonest_mask]['seller_profit'].fillna(0).sum()
                    honest_profit = merged[~dishonest_mask]['seller_profit'].fillna(0).sum()
                    
                    # 计算 dishonest 和 honest buyer utility（使用 fillna(0) 处理缺失值）
                    dishonest_buyer_utility = merged[dishonest_mask]['buyer_utility'].fillna(0).sum()
                    honest_buyer_utility = merged[~dishonest_mask]['buyer_utility'].fillna(0).sum()
                    
                    # 计算 dishonest 和 honest 交易数量
                    dishonest_transaction_count = len(merged[dishonest_mask])
                    honest_transaction_count = len(merged[~dishonest_mask])
                else:
                    # 如果无法关联，使用默认值
                    honest_profit = transactions['seller_profit'].fillna(0).sum()
                    dishonest_profit = 0
                    honest_buyer_utility = transactions['buyer_utility'].fillna(0).sum()
                    dishonest_buyer_utility = 0
                    honest_transaction_count = len(transactions)
                    dishonest_transaction_count = 0
                
                honest_profits.append(honest_profit)
                dishonest_profits.append(dishonest_profit)
                seller_profits.append(honest_profit + dishonest_profit)
                honest_buyer_utilities.append(honest_buyer_utility)
                dishonest_buyer_utilities.append(dishonest_buyer_utility)
                buyer_utilities.append(honest_buyer_utility + dishonest_buyer_utility)
                honest_transaction_counts.append(honest_transaction_count)
                dishonest_transaction_counts.append(dishonest_transaction_count)
                transaction_counts.append(honest_transaction_count + dishonest_transaction_count)
        
        if not run_ids:
            return {}
        
        return {
            'run_ids': run_ids,
            'seller_profits': seller_profits,
            'honest_profits': honest_profits,
            'dishonest_profits': dishonest_profits,
            'buyer_utilities': buyer_utilities,
            'honest_buyer_utilities': honest_buyer_utilities,
            'dishonest_buyer_utilities': dishonest_buyer_utilities,
            'transaction_counts': transaction_counts,
            'honest_transaction_counts': honest_transaction_counts,
            'dishonest_transaction_counts': dishonest_transaction_counts
        }
    
    def plot_seller_profits_comparison(self, out_dir: str):
        """绘制 Seller Profits 堆叠柱状图"""
        data = self._prepare_cross_run_data()
        if not data:
            return
        
        run_ids = data['run_ids']
        honest_profits = data['honest_profits']
        dishonest_profits = data['dishonest_profits']
        seller_profits = data['seller_profits']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f'Seller Profits Comparison - Experiment {self.experiment_id}', fontsize=14)
        
        # 先绘制 honest 收益（绿色，底层）
        ax.bar(run_ids, honest_profits, alpha=0.7, color='green', label='Honest Profit')
        # 再绘制 dishonest 收益（红色，堆叠在顶层）
        ax.bar(run_ids, dishonest_profits, bottom=honest_profits, alpha=0.7, color='red', label='Dishonest Profit')
        
        # 添加三条平均值线：Honest Mean, Dishonest Mean, Total Mean
        honest_mean = np.mean(honest_profits)
        dishonest_mean = np.mean(dishonest_profits)
        total_mean = np.mean(seller_profits)
        # Honest Mean 显示在底部（honest 部分的平均值）
        ax.axhline(y=honest_mean, color='darkgreen', linestyle='--', linewidth=1.5,
                   label=f'Honest Mean: {honest_mean:.2f}')
        # Dishonest Mean 显示在 dishonest_mean 的绝对值位置（从0开始的坐标）
        ax.axhline(y=dishonest_mean, color='darkred', linestyle='--', linewidth=1.5,
                   label=f'Dishonest Mean: {dishonest_mean:.2f}')
        # Total Mean 显示在顶部（总和位置）
        ax.axhline(y=total_mean, color='blue', linestyle='--', linewidth=1.5,
                   label=f'Total Mean: {total_mean:.2f}')
        
        ax.set_title('Total Seller Profits (Honest vs Dishonest)')
        ax.set_xlabel('Run ID')
        ax.set_ylabel('Total Profit')
        ax.set_ylim(0, 200)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_save(fig, out_dir, 'seller_profits_comparison')
    
    def plot_buyer_utilities_comparison(self, out_dir: str):
        """绘制 Buyer Utilities 柱状图"""
        data = self._prepare_cross_run_data()
        if not data:
            return
        
        run_ids = data['run_ids']
        buyer_utilities = data['buyer_utilities']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f'Buyer Utilities Comparison - Experiment {self.experiment_id}', fontsize=14)
        
        ax.bar(run_ids, buyer_utilities, alpha=0.7, color='lightgreen')
        ax.axhline(y=np.mean(buyer_utilities), color='red', linestyle='--',
                   label=f'Mean: {np.mean(buyer_utilities):.2f}')
        ax.set_title('Total Buyer Utilities')
        ax.set_xlabel('Run ID')
        ax.set_ylabel('Total Utility')
        ax.set_ylim(0, 220)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_save(fig, out_dir, 'buyer_utilities_comparison')
    
    def plot_transaction_counts_comparison(self, out_dir: str):
        """绘制 Transaction Counts 堆叠柱状图"""
        data = self._prepare_cross_run_data()
        if not data:
            return
        
        run_ids = data['run_ids']
        honest_transaction_counts = data['honest_transaction_counts']
        dishonest_transaction_counts = data['dishonest_transaction_counts']
        transaction_counts = data['transaction_counts']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f'Transaction Counts Comparison - Experiment {self.experiment_id}', fontsize=14)
        
        # 先绘制 honest 交易数量（绿色，底层）
        ax.bar(run_ids, honest_transaction_counts, alpha=0.7, color='green', label='Honest Transactions')
        # 再绘制 dishonest 交易数量（红色，堆叠在顶层）
        ax.bar(run_ids, dishonest_transaction_counts, bottom=honest_transaction_counts, alpha=0.7, color='red', label='Dishonest Transactions')
        
        # 添加三条平均值线：Honest Mean, Dishonest Mean, Total Mean
        honest_mean_transactions = np.mean(honest_transaction_counts)
        dishonest_mean_transactions = np.mean(dishonest_transaction_counts)
        total_mean_transactions = np.mean(transaction_counts)
        # Honest Mean 显示在底部（honest 部分的平均值）
        ax.axhline(y=honest_mean_transactions, color='darkgreen', linestyle='--', linewidth=1.5,
                   label=f'Honest Mean: {honest_mean_transactions:.1f}')
        # Dishonest Mean 显示在 dishonest_mean_transactions 的绝对值位置（从0开始的坐标）
        ax.axhline(y=dishonest_mean_transactions, color='darkred', linestyle='--', linewidth=1.5,
                   label=f'Dishonest Mean: {dishonest_mean_transactions:.1f}')
        # Total Mean 显示在顶部（总和位置）
        ax.axhline(y=total_mean_transactions, color='blue', linestyle='--', linewidth=1.5,
                   label=f'Total Mean: {total_mean_transactions:.1f}')
        
        ax.set_title('Transaction Counts (Honest vs Dishonest)')
        ax.set_xlabel('Run ID')
        ax.set_ylabel('Number of Transactions')
        ax.set_ylim(0, max(transaction_counts) * 1.1 if transaction_counts else 200)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_save(fig, out_dir, 'transaction_counts_comparison')
    
    def plot_profit_vs_utility_scatter(self, out_dir: str):
        """绘制 Seller Profits vs Buyer Utilities 散点图"""
        data = self._prepare_cross_run_data()
        if not data:
            return
        
        run_ids = data['run_ids']
        seller_profits = data['seller_profits']
        buyer_utilities = data['buyer_utilities']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f'Profit vs Utility Scatter - Experiment {self.experiment_id}', fontsize=14)
        
        ax.scatter(seller_profits, buyer_utilities, alpha=0.7, s=60)
        ax.set_title('Seller Profits vs Buyer Utilities')
        ax.set_xlabel('Total Seller Profit')
        ax.set_ylabel('Total Buyer Utility')
        ax.set_xlim(0, max(seller_profits) * 1.1 if seller_profits else 200)
        ax.set_ylim(0, max(buyer_utilities) * 1.1 if buyer_utilities else 200)
        ax.grid(True, alpha=0.3)
        
        # Add run ID labels
        for i, run_id in enumerate(run_ids):
            ax.annotate(f'R{run_id}', (seller_profits[i], buyer_utilities[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plot_save(fig, out_dir, 'profit_vs_utility_scatter')
    
    def plot_cross_run_comparison(self, out_dir: str):
        """Plot cross-run comparison chart (组合图)"""
        data = self._prepare_cross_run_data()
        if not data:
            return
        
        run_ids = data['run_ids']
        seller_profits = data['seller_profits']
        honest_profits = data['honest_profits']
        dishonest_profits = data['dishonest_profits']
        buyer_utilities = data['buyer_utilities']
        transaction_counts = data['transaction_counts']
        honest_transaction_counts = data['honest_transaction_counts']
        dishonest_transaction_counts = data['dishonest_transaction_counts']
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Cross-Run Comparison Analysis - Experiment {self.experiment_id}', fontsize=14)
        
        # 1. Seller total profit comparison (堆叠图)
        # 先绘制 honest 收益（绿色，底层）
        axes[0, 0].bar(run_ids, honest_profits, alpha=0.7, color='green', label='Honest Profit')
        # 再绘制 dishonest 收益（红色，堆叠在顶层）
        axes[0, 0].bar(run_ids, dishonest_profits, bottom=honest_profits, alpha=0.7, color='red', label='Dishonest Profit')
        # 添加三条平均值线：Honest Mean, Dishonest Mean, Total Mean
        honest_mean = np.mean(honest_profits)
        dishonest_mean = np.mean(dishonest_profits)
        total_mean = np.mean(seller_profits)
        # Honest Mean 显示在底部（honest 部分的平均值）
        axes[0, 0].axhline(y=honest_mean, color='darkgreen', linestyle='--', linewidth=1.5,
                          label=f'Honest Mean: {honest_mean:.2f}')
        # Dishonest Mean 显示在 dishonest_mean 的绝对值位置（从0开始的坐标）
        axes[0, 0].axhline(y=dishonest_mean, color='darkred', linestyle='--', linewidth=1.5,
                          label=f'Dishonest Mean: {dishonest_mean:.2f}')
        # Total Mean 显示在顶部（总和位置）
        axes[0, 0].axhline(y=total_mean, color='blue', linestyle='--', linewidth=1.5,
                          label=f'Total Mean: {total_mean:.2f}')
        axes[0, 0].set_title('Total Seller Profits (Honest vs Dishonest)')
        axes[0, 0].set_xlabel('Run ID')
        axes[0, 0].set_ylabel('Total Profit')
        axes[0, 0].set_ylim(0, 200)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Buyer total utility comparison
        axes[0, 1].bar(run_ids, buyer_utilities, alpha=0.7, color='lightgreen')
        axes[0, 1].axhline(y=np.mean(buyer_utilities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(buyer_utilities):.2f}')
        axes[0, 1].set_title('Total Buyer Utilities')
        axes[0, 1].set_xlabel('Run ID')
        axes[0, 1].set_ylabel('Total Utility')
        axes[0, 1].set_ylim(0, 220)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Transaction count comparison (堆叠图)
        # 先绘制 honest 交易数量（绿色，底层）
        axes[1, 0].bar(run_ids, honest_transaction_counts, alpha=0.7, color='green', label='Honest Transactions')
        # 再绘制 dishonest 交易数量（红色，堆叠在顶层）
        axes[1, 0].bar(run_ids, dishonest_transaction_counts, bottom=honest_transaction_counts, alpha=0.7, color='red', label='Dishonest Transactions')
        # 添加三条平均值线：Honest Mean, Dishonest Mean, Total Mean
        honest_mean_transactions = np.mean(honest_transaction_counts)
        dishonest_mean_transactions = np.mean(dishonest_transaction_counts)
        total_mean_transactions = np.mean(transaction_counts)
        # Honest Mean 显示在底部（honest 部分的平均值）
        axes[1, 0].axhline(y=honest_mean_transactions, color='darkgreen', linestyle='--', linewidth=1.5,
                          label=f'Honest Mean: {honest_mean_transactions:.1f}')
        # Dishonest Mean 显示在 dishonest_mean_transactions 的绝对值位置（从0开始的坐标）
        axes[1, 0].axhline(y=dishonest_mean_transactions, color='darkred', linestyle='--', linewidth=1.5,
                          label=f'Dishonest Mean: {dishonest_mean_transactions:.1f}')
        # Total Mean 显示在顶部（总和位置）
        axes[1, 0].axhline(y=total_mean_transactions, color='blue', linestyle='--', linewidth=1.5,
                          label=f'Total Mean: {total_mean_transactions:.1f}')
        axes[1, 0].set_title('Transaction Counts (Honest vs Dishonest)')
        axes[1, 0].set_xlabel('Run ID')
        axes[1, 0].set_ylabel('Number of Transactions')
        axes[1, 0].set_ylim(0, max(transaction_counts) * 1.1 if transaction_counts else 200)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Profit vs Utility scatter plot
        axes[1, 1].scatter(seller_profits, buyer_utilities, alpha=0.7, s=60)
        axes[1, 1].set_title('Seller Profits vs Buyer Utilities')
        axes[1, 1].set_xlabel('Total Seller Profit')
        axes[1, 1].set_ylabel('Total Buyer Utility')
        axes[1, 1].set_xlim(0, max(seller_profits) * 1.1 if seller_profits else 200)
        axes[1, 1].set_ylim(0, max(buyer_utilities) * 1.1 if buyer_utilities else 200)
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
        axes[0, 0].set_ylim(0, 10)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Buyer utility distribution
        axes[0, 1].hist(all_buyer_utilities, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(np.mean(all_buyer_utilities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_buyer_utilities):.2f}')
        axes[0, 1].set_title('Total Buyer Utilities Distribution')
        axes[0, 1].set_xlabel('Total Utility')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_ylim(0, 10)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Transaction count distribution
        axes[0, 2].hist(all_transaction_counts, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].axvline(np.mean(all_transaction_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_transaction_counts):.1f}')
        axes[0, 2].set_title('Transaction Counts Distribution')
        axes[0, 2].set_xlabel('Number of Transactions')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_ylim(0, 10)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Second row: Box plots
        axes[1, 0].boxplot(all_seller_profits, labels=['Seller Profits'])
        axes[1, 0].set_title('Seller Profits Box Plot')
        axes[1, 0].set_ylabel('Total Profit')
        axes[1, 0].set_ylim(0, 10)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].boxplot(all_buyer_utilities, labels=['Buyer Utilities'])
        axes[1, 1].set_title('Buyer Utilities Box Plot')
        axes[1, 1].set_ylabel('Total Utility')
        axes[1, 1].set_ylim(0, 10)
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].boxplot(all_transaction_counts, labels=['Transaction Counts'])
        axes[1, 2].set_title('Transaction Counts Box Plot')
        axes[1, 2].set_ylabel('Number of Transactions')
        axes[1, 2].set_ylim(0, 10)
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
        axes[0, 0].set_ylim(0, 24)
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
        axes[0, 1].set_ylim(0, 12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Deception counts boxplot and scatter
        axes[1, 0].boxplot(total_deceptions, labels=['Deceptions per Run'], patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1, 0].scatter(range(len(run_ids)), total_deceptions, alpha=0.6, s=50, color='darkblue')
        axes[1, 0].set_title('Deception Cases Box Plot', fontweight='bold')
        axes[1, 0].set_ylabel('Number of Deceptions')
        axes[1, 0].set_ylim(0, 12)
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
    # 组合图
    analyzer.plot_cross_run_comparison(aggregated_dir)
    # 单独图表
    analyzer.plot_seller_profits_comparison(aggregated_dir)
    analyzer.plot_buyer_utilities_comparison(aggregated_dir)
    analyzer.plot_transaction_counts_comparison(aggregated_dir)
    analyzer.plot_profit_vs_utility_scatter(aggregated_dir)
    # 其他分析图表
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
    parser.add_argument('--experiment_id', help='Experiment ID', default='experiment_20251103_102825')
    args = parser.parse_args()
    
    asyncio.run(analyze_experiment(args.experiment_id))


if __name__ == "__main__":
    main()