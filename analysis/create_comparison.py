#!/usr/bin/env python3
"""
Create market mechanism comparison visualization charts
Compare Reputation-Only vs Reputation+Warrant markets

This script generates comprehensive comparison visualizations including:
1. Core metrics comparison (buyer utility, seller profit, transaction volume)
2. Round-by-round progression analysis
3. Distribution analysis (histograms and box plots)
4. Deception behavior analysis (TQ=Low & AQ=High instances)

The deception behavior chart quantifies seller deception by counting instances where
the true product quality (TQ) is Low but the advertised quality (AQ) is High, indicating
deliberate misrepresentation of product quality by sellers.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    'reputation_only': "experiment_20251103_104146",
    'reputation_warrant': "experiment_20251103_102825"
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


def load_experiment_config(exp_id):
    """Load experiment configuration"""
    config_file = f"experiments/{exp_id}/config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config for {exp_id}: {e}")
    return {}


def format_experiment_label(config):
    """Format experiment label from configuration"""
    parts = []
    market_type = config.get('MARKET_TYPE', 'unknown')
    comm_type = config.get('COMMUNICATION_TYPE', 'none')
    
    # Format market type
    if market_type == 'reputation_only':
        parts.append('Rep-Only')
    elif market_type == 'reputation_warrant':
        parts.append('Rep+Warrant')
    else:
        parts.append(market_type.replace('_', ' ').title())
    
    # Format communication type
    if comm_type and comm_type != 'none':
        parts.append(f'{comm_type.title()}Comm')
    
    return ' | '.join(parts) if parts else 'Unknown'

def create_deception_behavior_comparison(output_dir):
    """Create deception behavior comparison chart - TQ=Low and AQ=High instances
    
    This function quantifies seller deception across two market mechanisms by counting
    instances where sellers post Low Quality (LQ) products but advertise them as High Quality (HQ).
    
    The analysis includes:
    - Mean and standard deviation of deception instances per run
    - Bar chart showing average deception rates for each mechanism
    - Box plot showing distribution across all 50 runs
    - Percentage change indicator showing relative increase/decrease
    
    Returns:
        tuple: (output_path, statistics_dict)
            - output_path: Path to the generated PNG chart
            - statistics_dict: Dictionary containing:
                * reputation_only_mean: Average deception instances in reputation-only system
                * reputation_only_std: Standard deviation for reputation-only
                * reputation_warrant_mean: Average deception instances in reputation+warrant system
                * reputation_warrant_std: Standard deviation for reputation+warrant
                * reputation_only_data: List of deception counts per run
                * reputation_warrant_data: List of deception counts per run
    
    This metric is crucial for understanding how different market mechanisms affect
    seller incentives and behavior regarding quality misrepresentation.
    """
    import sqlite3
    
    # Collect deception data for each run
    rep_only_deceptions = []
    rep_warrant_deceptions = []
    
    # Reputation-Only experiment
    exp_dir = f"experiments/{EXPERIMENT_CONFIG['reputation_only']}"
    for run_id in range(1, 51):  # 50 runs
        db_path = f"{exp_dir}/run_{run_id}.db"
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                # Count instances where true_quality='LQ' and advertised_quality='HQ'
                cursor.execute("""
                    SELECT COUNT(*) FROM product 
                    WHERE true_quality = 'LQ' AND advertised_quality = 'HQ'
                """)
                deception_count = cursor.fetchone()[0]
                rep_only_deceptions.append(deception_count)
                conn.close()
            except Exception as e:
                print(f"Warning: Could not read deception data from {db_path}: {e}")
                rep_only_deceptions.append(0)
        else:
            rep_only_deceptions.append(0)
    
    # Reputation+Warrant experiment
    exp_dir = f"experiments/{EXPERIMENT_CONFIG['reputation_warrant']}"
    for run_id in range(1, 51):  # 50 runs
        db_path = f"{exp_dir}/run_{run_id}.db"
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                # Count instances where true_quality='LQ' and advertised_quality='HQ'
                cursor.execute("""
                    SELECT COUNT(*) FROM product 
                    WHERE true_quality = 'LQ' AND advertised_quality = 'HQ'
                """)
                deception_count = cursor.fetchone()[0]
                rep_warrant_deceptions.append(deception_count)
                conn.close()
            except Exception as e:
                print(f"Warning: Could not read deception data from {db_path}: {e}")
                rep_warrant_deceptions.append(0)
        else:
            rep_warrant_deceptions.append(0)
    
    # Calculate statistics
    rep_only_mean = np.mean(rep_only_deceptions)
    rep_only_std = np.std(rep_only_deceptions)
    rep_warrant_mean = np.mean(rep_warrant_deceptions)
    rep_warrant_std = np.std(rep_warrant_deceptions)
    
    # Create chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Deception Behavior Analysis: TQ=Low & AQ=High Instances', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#ff9999', '#66b3ff']
    labels = ['Reputation-Only', 'Reputation+Warrant']
    values = [rep_only_mean, rep_warrant_mean]
    stds = [rep_only_std, rep_warrant_std]
    
    # Bar chart comparison
    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, alpha=0.8, 
                 yerr=stds, capsize=10, error_kw={'linewidth': 2})
    ax.set_title('Average Deception Instances per Run', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count of TQ=Low & AQ=High Posts')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val, std in zip(bars, values, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
               f'{val:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Calculate percentage change
    if rep_only_mean > 0:
        change_pct = ((rep_warrant_mean - rep_only_mean) / rep_only_mean) * 100
        if change_pct < 0:
            label_text = f'Reduction: {abs(change_pct):.1f}%'
            color_box = 'lightgreen'
        else:
            label_text = f'Increase: +{change_pct:.1f}%'
            color_box = 'lightcoral'
        ax.text(0.5, 0.95, label_text, 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color_box, alpha=0.8),
               fontweight='bold')
    
    # Box plot comparison
    ax = axes[1]
    bp = ax.boxplot([rep_only_deceptions, rep_warrant_deceptions], 
                    tick_labels=labels,
                    patch_artist=True,
                    widths=0.6)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Distribution of Deception Instances', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count of TQ=Low & AQ=High Posts per Run')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'market_mechanism_deception_behavior.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path, {
        'reputation_only_mean': rep_only_mean,
        'reputation_only_std': rep_only_std,
        'reputation_warrant_mean': rep_warrant_mean,
        'reputation_warrant_std': rep_warrant_std,
        'reputation_only_data': rep_only_deceptions,
        'reputation_warrant_data': rep_warrant_deceptions
    }

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
    """Create round-by-round progression comparison chart
    
    Creates a single chart showing Consumer Utility and Producer Profit 
    for both market types across rounds, matching the reference style.
    """
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
    
    # Create single chart with all four lines
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Consumer Utility and Producer Profit by Round for each Market Type', 
                 fontsize=16, fontweight='bold')
    
    # Set background color (light blue/grey)
    ax.set_facecolor('#F0F8FF')  # Alice blue / very light blue-grey
    fig.patch.set_facecolor('white')
    
    # Plot Utility (Reputation Market) - orange solid line with circular markers
    ax.plot(common_rounds, rep_only_buyer, 'o-', 
            label='Utility (Reputation Market)', 
            linewidth=2.5, markersize=7, color='#FF8C42', markerfacecolor='#FF8C42', 
            markeredgewidth=1.5, markeredgecolor='#FF8C42')
    
    # Plot Utility (Warrant Market) - light orange/yellow dashed line with circular markers
    ax.plot(common_rounds, rep_warrant_buyer, 'o--', 
            label='Utility (Warrant Market)', 
            linewidth=2.5, markersize=7, color='#FFD93D', markerfacecolor='#FFD93D',
            markeredgewidth=1.5, markeredgecolor='#FFD93D', dashes=(8, 4))
    
    # Plot Profit (Reputation Market) - dark red solid line with circular markers
    ax.plot(common_rounds, rep_only_seller, 'o-', 
            label='Profit (Reputation Market)', 
            linewidth=2.5, markersize=7, color='#8B0000', markerfacecolor='#8B0000',
            markeredgewidth=1.5, markeredgecolor='#8B0000')
    
    # Plot Profit (Warrant Market) - red dashed line with circular markers
    ax.plot(common_rounds, rep_warrant_seller, 'o--', 
            label='Profit (Warrant Market)', 
            linewidth=2.5, markersize=7, color='#DC143C', markerfacecolor='#DC143C',
            markeredgewidth=1.5, markeredgecolor='#DC143C', dashes=(8, 4))
    
    # Format x-axis labels as "Round 1", "Round 2", etc.
    ax.set_xticks(common_rounds)
    ax.set_xticklabels([f'Round {r}' for r in common_rounds])
    
    # Set y-axis range (0 to max value, with some padding)
    all_values = rep_only_buyer + rep_only_seller + rep_warrant_buyer + rep_warrant_seller
    max_val = max(all_values) if all_values else 60
    min_val = min(all_values) if all_values else 0
    y_max = max(60, max_val * 1.1)  # At least 60 or 10% above max
    y_min = min(0, min_val * 1.1) if min_val < 0 else 0
    ax.set_ylim(y_min, y_max)
    
    # Add horizontal grid lines
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.2, axis='x', linestyle='--', linewidth=0.3)
    
    # Set y-axis major ticks
    if y_max <= 60:
        ax.set_yticks([0, 20, 40, 60])
    else:
        # Auto-generate reasonable ticks
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    
    # Add legend outside the plot (to the right of the plot)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, 
              framealpha=0.9, edgecolor='gray', fancybox=True)
    
    plt.tight_layout()
    output_path = output_dir / 'market_mechanism_round_progression.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
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
    
    # Load configurations for both experiments
    config_rep_only = load_experiment_config(EXPERIMENT_CONFIG['reputation_only'])
    config_rep_warrant = load_experiment_config(EXPERIMENT_CONFIG['reputation_warrant'])
    
    label_rep_only = format_experiment_label(config_rep_only)
    label_rep_warrant = format_experiment_label(config_rep_warrant)
    
    print(f"Experiment 1: {label_rep_only}")
    print(f"Experiment 2: {label_rep_warrant}")
    
    # Create timestamped output directory
    output_dir = create_output_directory()
    print(f"Output directory created: {output_dir}")
    
    # Prepare configuration for saving
    config = {
        'generation_time': datetime.now().isoformat(),
        'experiment_ids': EXPERIMENT_CONFIG,
        'experiment_labels': {
            'reputation_only': label_rep_only,
            'reputation_warrant': label_rep_warrant
        },
        'charts_generated': [],
        'description': f'Market mechanism comparison: {label_rep_only} vs {label_rep_warrant}'
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
        
        deception_path, deception_data = create_deception_behavior_comparison(output_dir)
        config['charts_generated'].append(str(deception_path.name))
        print(f"✅ Deception behavior comparison chart generated: {deception_path}")
        
        # Add deception behavior statistics to config
        config['deception_behavior_stats'] = {
            'reputation_only': {
                'mean': deception_data['reputation_only_mean'],
                'std': deception_data['reputation_only_std']
            },
            'reputation_warrant': {
                'mean': deception_data['reputation_warrant_mean'],
                'std': deception_data['reputation_warrant_std']
            },
            'description': 'Count of instances where TQ=Low and AQ=High (seller deception behavior)'
        }
        
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