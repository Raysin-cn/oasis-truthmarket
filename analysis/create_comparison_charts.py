#!/usr/bin/env python3
"""
创建市场机制对比可视化图表
比较 Reputation-Only vs Reputation+Warrant 市场
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

def load_experiment_data(exp_id):
    """加载实验统计数据"""
    stats_file = f"analysis/{exp_id}/aggregated/aggregated_statistics.json"
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_comparison_summary():
    """创建核心指标对比图"""
    # 加载数据
    rep_only = load_experiment_data("experiment_20251008_201013")
    rep_warrant = load_experiment_data("experiment_20251016_011004")
    
    # 提取核心指标
    metrics = {
        'Average Buyer Utility\nper Run': [
            rep_only['summary_stats']['avg_buyer_utility_per_run'],
            rep_warrant['summary_stats']['avg_buyer_utility_per_run']
        ],
        'Average Seller Profit\nper Run': [
            rep_only['summary_stats']['avg_seller_profit_per_run'],
            rep_warrant['summary_stats']['avg_seller_profit_per_run']
        ],
        'Average Transactions\nper Run': [
            rep_only['summary_stats']['avg_transactions_per_run'],
            rep_warrant['summary_stats']['avg_transactions_per_run']
        ]
    }
    
    # 标准差数据
    stds = {
        'Average Buyer Utility\nper Run': [
            rep_only['summary_stats']['std_buyer_utility_per_run'],
            rep_warrant['summary_stats']['std_buyer_utility_per_run']
        ],
        'Average Seller Profit\nper Run': [
            rep_only['summary_stats']['std_seller_profit_per_run'],
            rep_warrant['summary_stats']['std_seller_profit_per_run']
        ],
        'Average Transactions\nper Run': [
            rep_only['summary_stats']['std_transactions_per_run'],
            rep_warrant['summary_stats']['std_transactions_per_run']
        ]
    }
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Market Mechanism Comparison: Reputation-Only vs Reputation+Warrant', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#ff9999', '#66b3ff']
    labels = ['Reputation-Only', 'Reputation+Warrant']
    
    for i, (metric, values) in enumerate(metrics.items()):
        ax = axes[i]
        bars = ax.bar(labels, values, color=colors, alpha=0.8, 
                     yerr=stds[metric], capsize=10, error_kw={'linewidth': 2})
        
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for j, (bar, val, std) in enumerate(zip(bars, values, stds[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                   f'{val:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 添加改善/恶化指示
        if i == 0:  # 买家效用
            improvement = ((values[1] - values[0]) / abs(values[0])) * 100
            ax.text(0.5, 0.95, f'Improvement: +{improvement:.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                   fontweight='bold')
        elif i == 1:  # 卖家利润
            decrease = ((values[0] - values[1]) / values[0]) * 100
            ax.text(0.5, 0.95, f'Decrease: -{decrease:.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
                   fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('analysis/market_mechanism_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_round_progression_comparison():
    """创建轮次进展对比图"""
    rep_only = load_experiment_data("experiment_20251008_201013")
    rep_warrant = load_experiment_data("experiment_20251016_011004")
    
    # 提取轮次数据 - 只使用两个实验都有的轮次
    rep_only_rounds = set(rep_only['round_stats'].keys())
    rep_warrant_rounds = set(rep_warrant['round_stats'].keys())
    common_rounds = sorted([int(r) for r in rep_only_rounds.intersection(rep_warrant_rounds)])
    
    rep_only_buyer = [rep_only['round_stats'][str(r)]['avg_buyer_utility'] for r in common_rounds]
    rep_only_seller = [rep_only['round_stats'][str(r)]['avg_seller_profit'] for r in common_rounds]
    
    rep_warrant_buyer = [rep_warrant['round_stats'][str(r)]['avg_buyer_utility'] for r in common_rounds]
    rep_warrant_seller = [rep_warrant['round_stats'][str(r)]['avg_seller_profit'] for r in common_rounds]
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Round-by-Round Progression Comparison', fontsize=16, fontweight='bold')
    
    # 买家效用进展
    axes[0].plot(common_rounds, rep_only_buyer, 'o-', label='Reputation-Only', 
                linewidth=3, markersize=8, color='#ff6b6b')
    axes[0].plot(common_rounds, rep_warrant_buyer, 's-', label='Reputation+Warrant', 
                linewidth=3, markersize=8, color='#4ecdc4')
    axes[0].set_title('Average Buyer Utility by Round', fontweight='bold')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Average Buyer Utility')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 卖家利润进展
    axes[1].plot(common_rounds, rep_only_seller, 'o-', label='Reputation-Only', 
                linewidth=3, markersize=8, color='#ff6b6b')
    axes[1].plot(common_rounds, rep_warrant_seller, 's-', label='Reputation+Warrant', 
                linewidth=3, markersize=8, color='#4ecdc4')
    axes[1].set_title('Average Seller Profit by Round', fontweight='bold')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Average Seller Profit')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/market_mechanism_round_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_comparison():
    """创建分布对比图"""
    rep_only = load_experiment_data("experiment_20251008_201013")
    rep_warrant = load_experiment_data("experiment_20251016_011004")
    
    # 提取所有运行的数据
    rep_only_buyer = list(rep_only['buyer_utility_by_run'].values())
    rep_only_seller = list(rep_only['seller_profit_by_run'].values())
    
    rep_warrant_buyer = list(rep_warrant['buyer_utility_by_run'].values())
    rep_warrant_seller = list(rep_warrant['seller_profit_by_run'].values())
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution Comparison: 50 Runs Each', fontsize=16, fontweight='bold')
    
    # 买家效用分布对比
    axes[0, 0].hist(rep_only_buyer, bins=15, alpha=0.7, label='Reputation-Only', 
                   color='#ff9999', density=True)
    axes[0, 0].hist(rep_warrant_buyer, bins=15, alpha=0.7, label='Reputation+Warrant', 
                   color='#66b3ff', density=True)
    axes[0, 0].set_title('Buyer Utility Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Total Buyer Utility per Run')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(rep_only_buyer), color='red', linestyle='--', alpha=0.8)
    axes[0, 0].axvline(np.mean(rep_warrant_buyer), color='blue', linestyle='--', alpha=0.8)
    
    # 卖家利润分布对比
    axes[0, 1].hist(rep_only_seller, bins=15, alpha=0.7, label='Reputation-Only', 
                   color='#ff9999', density=True)
    axes[0, 1].hist(rep_warrant_seller, bins=15, alpha=0.7, label='Reputation+Warrant', 
                   color='#66b3ff', density=True)
    axes[0, 1].set_title('Seller Profit Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Total Seller Profit per Run')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(np.mean(rep_only_seller), color='red', linestyle='--', alpha=0.8)
    axes[0, 1].axvline(np.mean(rep_warrant_seller), color='blue', linestyle='--', alpha=0.8)
    
    # 买家效用箱线图
    axes[1, 0].boxplot([rep_only_buyer, rep_warrant_buyer], 
                      tick_labels=['Reputation-Only', 'Reputation+Warrant'],
                      patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 0].set_title('Buyer Utility Box Plot', fontweight='bold')
    axes[1, 0].set_ylabel('Total Buyer Utility per Run')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 卖家利润箱线图
    axes[1, 1].boxplot([rep_only_seller, rep_warrant_seller], 
                      tick_labels=['Reputation-Only', 'Reputation+Warrant'],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[1, 1].set_title('Seller Profit Box Plot', fontweight='bold')
    axes[1, 1].set_ylabel('Total Seller Profit per Run')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/market_mechanism_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """生成所有对比图表"""
    print("生成市场机制对比图表...")
    
    # 确保分析目录存在
    Path("analysis").mkdir(exist_ok=True)
    
    try:
        create_comparison_summary()
        print("✅ 核心指标对比图已生成: analysis/market_mechanism_comparison_summary.png")
        
        create_round_progression_comparison()
        print("✅ 轮次进展对比图已生成: analysis/market_mechanism_round_progression.png")
        
        create_distribution_comparison()
        print("✅ 分布对比图已生成: analysis/market_mechanism_distribution_comparison.png")
        
        print("\n🎉 所有对比图表生成完成！")
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")

if __name__ == "__main__":
    main()