#!/usr/bin/env python3
"""
Create market mechanism comparison visualization charts
Compare Reputation-Only vs Reputation+Warrant markets
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

# Set font and style for better visualization
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# Configuration - Unified experiment IDs
EXPERIMENT_CONFIG = {
    'reputation_only': "experiment_20251019_153954",
    'reputation_warrant': "experiment_20251019_171638"
}

def create_output_directory():
    """Create timestamped output directory and return its path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"analysis/comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_config(output_dir, config):
    """Save configuration to JSON file in output directory"""
    config_file = output_dir / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_experiment_data(exp_id):
    """Load experiment statistics data"""
    stats_file = f"analysis/{exp_id}/aggregated/aggregated_statistics.json"
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_comparison_summary(output_dir):
    """Create core metrics comparison chart"""
    # Load data
    rep_only = load_experiment_data(EXPERIMENT_CONFIG['reputation_only'])
    rep_warrant = load_experiment_data(EXPERIMENT_CONFIG['reputation_warrant'])
    
    # Extract core metrics
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
    
    # Standard deviation data
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
    
    # Create chart
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
        
        # Add value labels
        for j, (bar, val, std) in enumerate(zip(bars, values, stds[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                   f'{val:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement/deterioration indicators
        if i == 0:  # Buyer utility
            improvement = ((values[1] - values[0]) / abs(values[0])) * 100
            ax.text(0.5, 0.95, f'Improvement: +{improvement:.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                   fontweight='bold')
        elif i == 1:  # Seller profit
            decrease = ((values[0] - values[1]) / values[0]) * 100
            ax.text(0.5, 0.95, f'Decrease: -{decrease:.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
                   fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'market_mechanism_comparison_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def create_round_progression_comparison(output_dir):
    """Create round-by-round progression comparison chart"""
    rep_only = load_experiment_data(EXPERIMENT_CONFIG['reputation_only'])
    rep_warrant = load_experiment_data(EXPERIMENT_CONFIG['reputation_warrant'])
    
    # Extract round data - only use rounds that both experiments have
    rep_only_rounds = set(rep_only['round_stats'].keys())
    rep_warrant_rounds = set(rep_warrant['round_stats'].keys())
    common_rounds = sorted([int(r) for r in rep_only_rounds.intersection(rep_warrant_rounds)])
    
    rep_only_buyer = [rep_only['round_stats'][str(r)]['avg_buyer_utility'] for r in common_rounds]
    rep_only_seller = [rep_only['round_stats'][str(r)]['avg_seller_profit'] for r in common_rounds]
    
    rep_warrant_buyer = [rep_warrant['round_stats'][str(r)]['avg_buyer_utility'] for r in common_rounds]
    rep_warrant_seller = [rep_warrant['round_stats'][str(r)]['avg_seller_profit'] for r in common_rounds]
    
    # Create chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Round-by-Round Progression Comparison', fontsize=16, fontweight='bold')
    
    # Buyer utility progression
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
    
    # Seller profit progression
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
    output_path = output_dir / 'market_mechanism_round_progression.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def create_distribution_comparison(output_dir):
    """Create distribution comparison chart"""
    rep_only = load_experiment_data(EXPERIMENT_CONFIG['reputation_only'])
    rep_warrant = load_experiment_data(EXPERIMENT_CONFIG['reputation_warrant'])
    
    # Extract all run data
    rep_only_buyer = list(rep_only['buyer_utility_by_run'].values())
    rep_only_seller = list(rep_only['seller_profit_by_run'].values())
    
    rep_warrant_buyer = list(rep_warrant['buyer_utility_by_run'].values())
    rep_warrant_seller = list(rep_warrant['seller_profit_by_run'].values())
    
    # Create chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution Comparison: 50 Runs Each', fontsize=16, fontweight='bold')
    
    # Buyer utility distribution comparison
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
    
    # Seller profit distribution comparison
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
    
    # Buyer utility box plot
    axes[1, 0].boxplot([rep_only_buyer, rep_warrant_buyer], 
                      tick_labels=['Reputation-Only', 'Reputation+Warrant'],
                      patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 0].set_title('Buyer Utility Box Plot', fontweight='bold')
    axes[1, 0].set_ylabel('Total Buyer Utility per Run')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Seller profit box plot
    axes[1, 1].boxplot([rep_only_seller, rep_warrant_seller], 
                      tick_labels=['Reputation-Only', 'Reputation+Warrant'],
                      patch_artist=True,
                      boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[1, 1].set_title('Seller Profit Box Plot', fontweight='bold')
    axes[1, 1].set_ylabel('Total Seller Profit per Run')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'market_mechanism_distribution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def main():
    """Generate all comparison charts"""
    print("Generating market mechanism comparison charts...")

    
    # Create timestamped output directory
    output_dir = create_output_directory()
    print(f"Output directory created: {output_dir}")
    
    # Prepare configuration for saving
    config = {
        'generation_time': datetime.now().isoformat(),
        'experiment_ids': EXPERIMENT_CONFIG,
        'charts_generated': [],
        'description': 'Market mechanism comparison between Reputation-Only and Reputation+Warrant systems'
    }
    
    try:
        # Generate charts and collect output paths
        summary_path = create_comparison_summary(output_dir)
        config['charts_generated'].append(str(summary_path.name))
        print(f"✅ Core metrics comparison chart generated: {summary_path}")
        
        progression_path = create_round_progression_comparison(output_dir)
        config['charts_generated'].append(str(progression_path.name))
        print(f"✅ Round progression comparison chart generated: {progression_path}")
        
        distribution_path = create_distribution_comparison(output_dir)
        config['charts_generated'].append(str(distribution_path.name))
        print(f"✅ Distribution comparison chart generated: {distribution_path}")
        
        # Save configuration file
        save_config(output_dir, config)
        print(f"✅ Configuration saved: {output_dir}/config.json")
        
        print(f"\n🎉 All comparison charts generated successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Charts generated: {len(config['charts_generated'])}")
        
    except Exception as e:
        print(f"❌ Error generating charts: {e}")
        raise

if __name__ == "__main__":
    main()