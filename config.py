"""
Market Simulation Configuration
配置文件：包含所有仿真参数和设置
"""

import os
from datetime import datetime
from typing import Dict, Any


class SimulationConfig:
    """市场仿真配置类"""
    
    # 实验配置
    RUNS = 50
    NUM_SELLERS = 10
    NUM_BUYERS = 10
    SIMULATION_ROUNDS = 7
    
    # 市场规则参数
    REPUTATION_LAG = 1  # 声誉滞后显示轮数
    REENTRY_ALLOWED_ROUND = 5  # 重新进入市场允许轮数
    INITIAL_WINDOW_ROUNDS = [1, 2]  # 隐藏完整历史的初始轮数
    EXIT_ROUND = 7  # 退出市场允许轮数
    MARKET_TYPE = 'reputation_and_warrant'
    
    # 模型配置
    MODEL_PLATFORM = "OPENAI"
    # MODEL_TYPE = "gpt-4.1-mini-2025-04-14"
    MODEL_TYPE = "gpt-4o-mini"
    
    # 路径配置
    BASE_DATA_PATH = 'experiments'
    BASE_ANALYSIS_PATH = 'analysis'
    
    @classmethod
    def get_experiment_id(cls) -> str:
        """生成实验ID（基于时间戳）"""
        return datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    
    @classmethod
    def get_experiment_paths(cls, experiment_id: str) -> Dict[str, str]:
        """获取实验相关路径"""
        experiment_dir = os.path.join(cls.BASE_DATA_PATH, experiment_id)
        analysis_dir = os.path.join(cls.BASE_ANALYSIS_PATH, experiment_id)
        
        return {
            'experiment_dir': experiment_dir,
            'analysis_dir': analysis_dir,
            'individual_analysis_dir': os.path.join(analysis_dir, 'individual_runs'),
            'aggregated_analysis_dir': os.path.join(analysis_dir, 'aggregated'),
            'config_file': os.path.join(experiment_dir, 'config.json')
        }
    
    @classmethod
    def get_run_db_path(cls, experiment_id: str, run_id: int) -> str:
        """获取指定运行的数据库路径"""
        experiment_dir = os.path.join(cls.BASE_DATA_PATH, experiment_id)
        return os.path.join(experiment_dir, f'run_{run_id}.db')
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        return {
            'RUNS': cls.RUNS,
            'NUM_SELLERS': cls.NUM_SELLERS,
            'NUM_BUYERS': cls.NUM_BUYERS,
            'SIMULATION_ROUNDS': cls.SIMULATION_ROUNDS,
            'REPUTATION_LAG': cls.REPUTATION_LAG,
            'REENTRY_ALLOWED_ROUND': cls.REENTRY_ALLOWED_ROUND,
            'INITIAL_WINDOW_ROUNDS': cls.INITIAL_WINDOW_ROUNDS,
            'EXIT_ROUND': cls.EXIT_ROUND,
            'MARKET_TYPE': cls.MARKET_TYPE,
            'MODEL_PLATFORM': cls.MODEL_PLATFORM,
            'MODEL_TYPE': cls.MODEL_TYPE
        }
    
    @classmethod
    def save_config(cls, experiment_id: str):
        """保存配置到实验目录"""
        import json
        paths = cls.get_experiment_paths(experiment_id)
        os.makedirs(os.path.dirname(paths['config_file']), exist_ok=True)
        
        config_data = cls.to_dict()
        config_data['experiment_id'] = experiment_id
        config_data['created_at'] = datetime.now().isoformat()
        
        with open(paths['config_file'], 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

