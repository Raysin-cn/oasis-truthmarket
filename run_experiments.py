#!/usr/bin/env python3
"""
多次重复实验运行脚本
支持运行多次独立的市场仿真实验，每次实验包含7轮交互
每次运行使用独立的数据库文件，避免数据冲突
"""

import asyncio
import os
import sys
import argparse
from typing import Optional

from config import SimulationConfig
from utils import ExperimentManager, print_run_header, print_run_footer, setup_single_run_environment

# 导入单次运行的逻辑（从原始文件中提取）
from run_single_simulation import run_single_simulation


async def run_experiments(runs: Optional[int] = None, experiment_id: Optional[str] = None):
    """运行多次重复实验
    
    Args:
        runs: 运行次数，默认使用配置文件中的值
        experiment_id: 实验ID，如果为None则自动生成
    """
    # 使用提供的参数或默认配置
    total_runs = runs or SimulationConfig.RUNS
    
    # 初始化实验管理器
    manager = ExperimentManager(experiment_id)
    experiment_id = manager.prepare_experiment()
    
    print(f"开始多次重复实验")
    print(f"实验ID: {experiment_id}")
    print(f"计划运行次数: {total_runs}")
    print(f"每次运行轮数: {SimulationConfig.SIMULATION_ROUNDS}")
    print(f"卖家数量: {SimulationConfig.NUM_SELLERS}")
    print(f"买家数量: {SimulationConfig.NUM_BUYERS}")
    print(f"市场类型: {SimulationConfig.MARKET_TYPE}")
    
    successful_runs = 0
    failed_runs = 0
    
    # 运行所有实验
    for run_id in range(1, total_runs + 1):
        try:
            print_run_header(experiment_id, run_id, total_runs)
            
            # 设置当前运行的环境
            db_path = setup_single_run_environment(experiment_id, run_id)
            
            # 清理可能存在的旧数据库
            if os.path.exists(db_path):
                manager.cleanup_run_database(run_id)
            
            print(f"开始运行 {run_id}/{total_runs}")
            print(f"数据库路径: {db_path}")
            
            # 运行单次仿真
            await run_single_simulation(db_path)
            
            print(f"运行 {run_id} 成功完成")
            successful_runs += 1
            
        except Exception as e:
            print(f"运行 {run_id} 失败: {str(e)}")
            failed_runs += 1
            # 继续下一次运行，不要因为一次失败就停止整个实验
            continue
        finally:
            print_run_footer(run_id, total_runs)
    
    # 收集和保存结果
    print(f"\n所有运行完成！")
    print(f"成功: {successful_runs}, 失败: {failed_runs}")
    
    print(f"\n正在收集和分析结果...")
    results = manager.collect_run_results()
    manager.save_experiment_results(results)
    manager.print_experiment_summary(results)
    
    # 运行聚合分析
    print(f"\n正在生成聚合分析...")
    try:
        from analysis.multi_run_analysis import analyze_experiment
        await analyze_experiment(experiment_id)
        print(f"聚合分析完成")
    except ImportError:
        print(f"聚合分析模块未找到，跳过聚合分析")
    except Exception as e:
        print(f"聚合分析失败: {e}")
    
    print(f"\n实验完成！结果保存在: {manager.paths['analysis_dir']}")
    return results


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(description='运行多次重复市场仿真实验')
    parser.add_argument('--runs', type=int, default=None, 
                      help=f'运行次数 (默认: {SimulationConfig.RUNS})')
    parser.add_argument('--experiment-id', type=str, default=None,
                      help='实验ID (默认: 自动生成基于时间戳的ID)')
    parser.add_argument('--config', action='store_true',
                      help='显示当前配置并退出')
    
    args = parser.parse_args()
    
    if args.config:
        print("当前配置:")
        config_dict = SimulationConfig.to_dict()
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
        return
    
    try:
        # 运行实验
        results = asyncio.run(run_experiments(args.runs, args.experiment_id))
        print(f"\n实验成功完成!")
        sys.exit(0)
        
    except KeyboardInterrupt:
        print(f"\n实验被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n实验失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()