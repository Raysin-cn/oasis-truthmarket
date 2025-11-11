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
from typing import Optional

from config import SimulationConfig
from utils import ExperimentManager, print_run_header, print_run_footer, setup_single_run_environment

# Import single run logic (extracted from original file)
from run_single_simulation import run_single_simulation


async def run_experiments(runs: Optional[int] = None, experiment_id: Optional[str] = None):
    """Run multiple repeated experiments
    
    Args:
        runs: Number of runs, defaults to value in config file
        experiment_id: Experiment ID, auto-generated if None
    """
    # Use provided parameters or default configuration
    total_runs = runs or SimulationConfig.RUNS
    
    # Initialize experiment manager
    manager = ExperimentManager(experiment_id)
    experiment_id = manager.prepare_experiment()
    
    print(f"Starting multiple repeated experiments")
    print(f"Experiment ID: {experiment_id}")
    print(f"Planned number of runs: {total_runs}")
    print(f"Rounds per run: {SimulationConfig.SIMULATION_ROUNDS}")
    print(f"Number of sellers: {SimulationConfig.NUM_SELLERS}")
    print(f"Number of buyers: {SimulationConfig.NUM_BUYERS}")
    print(f"Market type: {SimulationConfig.MARKET_TYPE}")
    
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
            
            # Run single simulation
            await run_single_simulation(db_path)
            
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
        results = asyncio.run(run_experiments(args.runs, args.experiment_id))
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