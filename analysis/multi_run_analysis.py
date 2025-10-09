#!/usr/bin/env python3
"""
多次运行聚合分析模块
用于分析多次重复实验的结果，生成聚合统计和可视化
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
from analyze_market import read_table, ensure_output_dir, plot_save


class MultiRunAnalyzer:
    """多次运行分析器"""
    
    def __init__(self, experiment_id: str):
        """初始化分析器
        
        Args:
            experiment_id: 实验ID
        """
        self.experiment_id = experiment_id
        self.paths = SimulationConfig.get_experiment_paths(experiment_id)
        self.run_data = {}
        self.aggregated_data = {}
        
    def load_experiment_data(self):
        """加载实验数据"""
        print(f"加载实验数据: {self.experiment_id}")
        
        # 查找所有运行的数据库文件
        experiment_dir = self.paths['experiment_dir']
        if not os.path.exists(experiment_dir):
            print(f"实验目录不存在: {experiment_dir}")
            return
        
        db_files = [f for f in os.listdir(experiment_dir) if f.startswith('run_') and f.endswith('.db')]
        print(f"找到 {len(db_files)} 个运行数据库")
        
        for db_file in db_files:
            run_id = int(db_file.replace('run_', '').replace('.db', ''))
            db_path = os.path.join(experiment_dir, db_file)
            
            try:
                run_data = self._load_single_run_data(db_path, run_id)
                self.run_data[run_id] = run_data
                print(f"已加载 Run {run_id}")
            except Exception as e:
                print(f"加载 Run {run_id} 失败: {e}")
        
        print(f"成功加载 {len(self.run_data)} 次运行的数据")
    
    def _load_single_run_data(self, db_path: str, run_id: int) -> Dict[str, pd.DataFrame]:
        """加载单次运行的数据"""
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
        """生成聚合统计信息"""
        stats = {
            'experiment_id': self.experiment_id,
            'total_runs': len(self.run_data),
            'summary_stats': {},
            'round_stats': {},
            'cross_run_variance': {}
        }
        
        if not self.run_data:
            return stats
        
        # 聚合基本统计
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
        
        # 计算跨运行统计
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
        
        # 按轮次聚合统计
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
        """绘制跨运行对比图"""
        if not self.run_data:
            return
        
        # 准备数据
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
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Cross-Run Comparison Analysis - Experiment {self.experiment_id}', fontsize=14)
        
        # 1. 卖家总利润对比
        axes[0, 0].bar(run_ids, seller_profits, alpha=0.7, color='skyblue')
        axes[0, 0].axhline(y=np.mean(seller_profits), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(seller_profits):.2f}')
        axes[0, 0].set_title('Total Seller Profits')
        axes[0, 0].set_xlabel('Run ID')
        axes[0, 0].set_ylabel('Total Profit')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 买家总效用对比
        axes[0, 1].bar(run_ids, buyer_utilities, alpha=0.7, color='lightgreen')
        axes[0, 1].axhline(y=np.mean(buyer_utilities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(buyer_utilities):.2f}')
        axes[0, 1].set_title('Total Buyer Utilities')
        axes[0, 1].set_xlabel('Run ID')
        axes[0, 1].set_ylabel('Total Utility')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 交易次数对比
        axes[1, 0].bar(run_ids, transaction_counts, alpha=0.7, color='orange')
        axes[1, 0].axhline(y=np.mean(transaction_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(transaction_counts):.1f}')
        axes[1, 0].set_title('Transaction Counts')
        axes[1, 0].set_xlabel('Run ID')
        axes[1, 0].set_ylabel('Number of Transactions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 利润vs效用散点图
        axes[1, 1].scatter(seller_profits, buyer_utilities, alpha=0.7, s=60)
        axes[1, 1].set_title('Seller Profits vs Buyer Utilities')
        axes[1, 1].set_xlabel('Total Seller Profit')
        axes[1, 1].set_ylabel('Total Buyer Utility')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加运行ID标签
        for i, run_id in enumerate(run_ids):
            axes[1, 1].annotate(f'R{run_id}', (seller_profits[i], buyer_utilities[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plot_save(fig, out_dir, 'cross_run_comparison')
    
    def plot_round_progression(self, out_dir: str):
        """绘制轮次进展图"""
        if not self.run_data or not self.aggregated_data.get('round_stats'):
            return
        
        round_stats = self.aggregated_data['round_stats']
        rounds = sorted(round_stats.keys())
        
        # 准备数据
        avg_seller_profits = [round_stats[r]['avg_seller_profit'] for r in rounds]
        std_seller_profits = [round_stats[r]['std_seller_profit'] for r in rounds]
        avg_buyer_utilities = [round_stats[r]['avg_buyer_utility'] for r in rounds]
        std_buyer_utilities = [round_stats[r]['std_buyer_utility'] for r in rounds]
        avg_transactions = [round_stats[r]['avg_transactions'] for r in rounds]
        std_transactions = [round_stats[r]['std_transactions'] for r in rounds]
        
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Round Progression Analysis - Experiment {self.experiment_id}', fontsize=14)
        
        # 1. 卖家利润进展
        axes[0].errorbar(rounds, avg_seller_profits, yerr=std_seller_profits, 
                        fmt='-o', capsize=5, linewidth=2, markersize=6)
        axes[0].set_title('Seller Profit Progression')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Average Profit ± Std Dev')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 买家效用进展
        axes[1].errorbar(rounds, avg_buyer_utilities, yerr=std_buyer_utilities,
                        fmt='-o', capsize=5, linewidth=2, markersize=6, color='green')
        axes[1].set_title('Buyer Utility Progression')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Average Utility ± Std Dev')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 交易活跃度进展
        axes[2].errorbar(rounds, avg_transactions, yerr=std_transactions,
                        fmt='-o', capsize=5, linewidth=2, markersize=6, color='orange')
        axes[2].set_title('Transaction Activity Progression')
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('Average Transactions ± Std Dev')
        axes[2].grid(True, alpha=0.3)
        
        plot_save(fig, out_dir, 'round_progression')
    
    def plot_distribution_analysis(self, out_dir: str):
        """绘制分布分析图"""
        if not self.run_data:
            return
        
        # 收集所有运行的数据
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
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Distribution Analysis - Experiment {self.experiment_id}', fontsize=14)
        
        # 第一行：直方图
        # 卖家利润分布
        axes[0, 0].hist(all_seller_profits, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(all_seller_profits), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_seller_profits):.2f}')
        axes[0, 0].set_title('Total Seller Profits Distribution')
        axes[0, 0].set_xlabel('Total Profit')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 买家效用分布
        axes[0, 1].hist(all_buyer_utilities, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(np.mean(all_buyer_utilities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_buyer_utilities):.2f}')
        axes[0, 1].set_title('Total Buyer Utilities Distribution')
        axes[0, 1].set_xlabel('Total Utility')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 交易次数分布
        axes[0, 2].hist(all_transaction_counts, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].axvline(np.mean(all_transaction_counts), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_transaction_counts):.1f}')
        axes[0, 2].set_title('Transaction Counts Distribution')
        axes[0, 2].set_xlabel('Number of Transactions')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 第二行：箱线图
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
    
    def generate_individual_run_analysis(self):
        """为每次运行生成单独的分析"""
        individual_dir = self.paths['individual_analysis_dir']
        
        for run_id, data in self.run_data.items():
            run_output_dir = os.path.join(individual_dir, f'run_{run_id}')
            os.makedirs(run_output_dir, exist_ok=True)
            
            try:
                # 使用现有的分析模块为每次运行生成分析
                db_path = SimulationConfig.get_run_db_path(self.experiment_id, run_id)
                
                # 使用analyze_market.py的功能
                import subprocess
                result = subprocess.run([
                    'python', 'analysis/analyze_market.py', 
                    db_path, '--out', run_output_dir
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Run {run_id} 分析完成")
                else:
                    print(f"Run {run_id} 分析失败: {result.stderr}")
                    
            except Exception as e:
                print(f"Run {run_id} 分析过程中出错: {e}")
    
    def save_aggregated_results(self):
        """保存聚合结果"""
        if not self.aggregated_data:
            return
        
        results_file = os.path.join(self.paths['aggregated_analysis_dir'], 'aggregated_statistics.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.aggregated_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"聚合统计结果已保存到: {results_file}")


async def analyze_experiment(experiment_id: str):
    """分析整个实验的结果
    
    Args:
        experiment_id: 实验ID
    """
    print(f"开始分析实验: {experiment_id}")
    
    analyzer = MultiRunAnalyzer(experiment_id)
    
    # 加载数据
    analyzer.load_experiment_data()
    
    if not analyzer.run_data:
        print("没有找到可分析的数据")
        return
    
    # 生成聚合统计
    print("生成聚合统计...")
    stats = analyzer.generate_aggregated_statistics()
    
    # 保存结果
    analyzer.save_aggregated_results()
    
    # 生成聚合可视化
    aggregated_dir = analyzer.paths['aggregated_analysis_dir']
    print(f"生成聚合可视化到: {aggregated_dir}")
    
    sns.set_theme(style="whitegrid")
    analyzer.plot_cross_run_comparison(aggregated_dir)
    analyzer.plot_round_progression(aggregated_dir)
    analyzer.plot_distribution_analysis(aggregated_dir)
    
    # 生成单次运行分析
    print("生成单次运行分析...")
    analyzer.generate_individual_run_analysis()
    
    print(f"实验分析完成！结果保存在: {analyzer.paths['analysis_dir']}")


def main():
    """命令行入口点"""
    import argparse
    parser = argparse.ArgumentParser(description='分析多次运行实验结果')
    parser.add_argument('--experiment_id', help='实验ID', default='experiment_20251008_171502')
    args = parser.parse_args()
    
    asyncio.run(analyze_experiment(args.experiment_id))


if __name__ == "__main__":
    main()