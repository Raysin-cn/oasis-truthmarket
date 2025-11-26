#!/usr/bin/env python3
"""
Multiple repeated experiment runner script
Supports running multiple independent market simulation experiments, each experiment contains 7 rounds of interaction
Each run uses independent database files to avoid data conflicts
"""

import asyncio
import os
import sys
import argparse
from typing import Optional, Callable
import json
from datetime import datetime

from config import SimulationConfig
from utils import ExperimentManager, print_run_header, print_run_footer, setup_single_run_environment

# Import different experiment types
from run_single_simulation import run_single_simulation as run_simulation_no_comm
from run_single_simulation_seller_communication import run_single_simulation as run_simulation_seller_comm
from run_single_simulation_buyer_communication import run_single_simulation as run_simulation_buyer_comm

# Experiment type mapping
EXPERIMENT_TYPES = {
    'no_communication': run_simulation_no_comm,
    'seller_communication': run_simulation_seller_comm,
    'buyer_communication': run_simulation_buyer_comm,
}


async def run_experiments(runs: Optional[int] = None, 
                         experiment_id: Optional[str] = None,
                         experiment_type: str = 'no_communication',
                         market_type: Optional[str] = None):
    """Run multiple repeated experiments
    
    Args:
        runs: Number of runs, defaults to value in config file
        experiment_id: Experiment ID, auto-generated if None
        experiment_type: Type of experiment to run ('no_communication', 'seller_communication', or 'buyer_communication')
        market_type: Market type (optional, defaults to config value)
    """
    # Validate experiment type
    if experiment_type not in EXPERIMENT_TYPES:
        raise ValueError(f"Invalid experiment type: {experiment_type}. "
                        f"Must be one of: {', '.join(EXPERIMENT_TYPES.keys())}")
    
    # Get the corresponding simulation function
    run_simulation_func = EXPERIMENT_TYPES[experiment_type]
    
    # Use provided parameters or default configuration
    total_runs = runs or SimulationConfig.RUNS
    market_type_to_use = market_type if market_type is not None else SimulationConfig.MARKET_TYPE
    
    # Initialize experiment manager
    manager = ExperimentManager(experiment_id)
    # experiment_id = manager.prepare_experiment()
    
    config_data = SimulationConfig.to_dict()
    # Override with actual runtime values
    config_data['RUNS'] = total_runs
    config_data['MARKET_TYPE'] = market_type_to_use
    config_data['COMMUNICATION_TYPE'] = experiment_type
    config_data['experiment_id'] = experiment_id
    config_data['created_at'] = datetime.now().isoformat()
    
    with open(manager.paths['config_file'], 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"Runtime configuration saved to: {manager.paths['config_file']}")
    
    print(f"Starting multiple repeated experiments")
    print(f"Experiment ID: {experiment_id}")
    print(f"Experiment type: {experiment_type}")
    print(f"Planned number of runs: {total_runs}")
    print(f"Rounds per run: {SimulationConfig.SIMULATION_ROUNDS}")
    print(f"Number of sellers: {SimulationConfig.NUM_SELLERS}")
    print(f"Number of buyers: {SimulationConfig.NUM_BUYERS}")
    print(f"Market type: {market_type_to_use}")
    
    successful_runs = 0
    failed_runs = 0
    
    # Run all experiments
    for run_id in range(1, total_runs + 1):
        try:
            print_run_header(experiment_id, run_id, total_runs)
            
            # Set up environment for current run
            db_path = setup_single_run_environment(experiment_id, run_id)
            
            # Clean up any existing old database
            if os.path.exists(db_path):
                manager.cleanup_run_database(run_id)
            
            print(f"Starting run {run_id}/{total_runs}")
            print(f"Database path: {db_path}")
            
            # Run single simulation with selected experiment type and market type
            await run_simulation_func(db_path, market_type=market_type_to_use)
            
            print(f"Run {run_id} completed successfully")
            successful_runs += 1
            
        except Exception as e:
            print(f"Run {run_id} failed: {str(e)}")
            failed_runs += 1
            # Continue to next run, don't stop entire experiment due to one failure
            continue
        finally:
            print_run_footer(run_id, total_runs)
    
    # Collect and save results
    print(f"\nAll runs completed!")
    print(f"Successful: {successful_runs}, Failed: {failed_runs}")
    
    print(f"\nCollecting and analyzing results...")
    results = manager.collect_run_results()
    manager.save_experiment_results(results)
    manager.print_experiment_summary(results)
    
    # Run aggregated analysis
    print(f"\nGenerating aggregated analysis...")
    try:
        from analysis.multi_run_analysis import analyze_experiment
        await analyze_experiment(experiment_id)
        print(f"Aggregated analysis completed")
    except ImportError:
        print(f"Aggregated analysis module not found, skipping aggregated analysis")
    except Exception as e:
        print(f"Aggregated analysis failed: {e}")
    
    print(f"\nExperiment completed! Results saved in: {manager.paths['analysis_dir']}")
    return results


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='Run multiple repeated market simulation experiments')
    parser.add_argument('--runs', type=int, default=None, 
                      help=f'Number of runs (default: {SimulationConfig.RUNS})')
    parser.add_argument('--experiment-id', type=str, default=None,
                      help='Experiment ID (default: auto-generated timestamp-based ID)')
    parser.add_argument('--type', type=str, default='no_communication',
                      choices=list(EXPERIMENT_TYPES.keys()),
                      help='Experiment type to run (default: no_communication)')
    parser.add_argument('--market-type', type=str, default=None,
                      help='Market type to run (default: None)')
    parser.add_argument('--config', action='store_true',
                      help='Show current configuration and exit')
    
    args = parser.parse_args()
    
    if args.config:
        print("Current configuration:")
        config_dict = SimulationConfig.to_dict()
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
        return
    
    try:
        # Run experiments
        results = asyncio.run(run_experiments(args.runs, args.experiment_id, args.type, args.market_type))
        print(f"\nExperiment completed successfully!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print(f"\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()