#!/usr/bin/env python3
"""
使用 Embedding Atlas 进行 embedding 空间可视化分析

Embedding Atlas 是 Apple 开发的交互式可视化工具，用于展示大型 embedding 数据。
主要功能：
1. 交互式可视化：在浏览器中可视化、交叉过滤和搜索 embedding 和元数据
2. 自动聚类与标注：自动对数据进行聚类并生成标签
3. 密度可视化：通过核密度估计识别高密度区域和离群点
4. 实时搜索：搜索特定数据点并查找最近邻
5. 多视图协调：在不同元数据列之间进行交互式链接和筛选

使用方法：
1. 命令行工具：embedding-atlas path_to_dataset.parquet
2. Python Notebook Widget：在 Jupyter Notebook 中使用交互式小部件
3. Streamlit 组件：在 Streamlit 应用中集成可视化

参考文档：https://apple.github.io/embedding-atlas/overview.html
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse
import re
import json

# torch 是可选的，只在需要时导入
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Embedding Atlas 导入（需要先安装：pip install embedding-atlas）
try:
    from embedding_atlas.widget import EmbeddingAtlasWidget
    from embedding_atlas.streamlit import embedding_atlas
    EMBEDDING_ATLAS_AVAILABLE = True
except ImportError:
    EMBEDDING_ATLAS_AVAILABLE = False
    print("警告: embedding-atlas 未安装。请运行: pip install embedding-atlas")


def prepare_embedding_dataframe(
    embeddings: np.ndarray,
    texts: List[str],
    metadata: Optional[Dict[str, List]] = None,
    projection_method: str = "umap"
) -> pd.DataFrame:
    """
    准备用于 Embedding Atlas 可视化的 DataFrame
    
    Args:
        embeddings: embedding 向量数组，形状为 (n_samples, n_dimensions)
        texts: 文本列表，长度应与 embeddings 的第一维相同
        metadata: 可选的元数据字典，键为列名，值为列表
        projection_method: 投影方法，'umap' 或 'tsne'
        
    Returns:
        包含 embedding、文本和投影坐标的 DataFrame
    """
    n_samples = len(texts)
    if embeddings.shape[0] != n_samples:
        raise ValueError(f"文本数量 ({n_samples}) 与 embedding 数量 ({embeddings.shape[0]}) 不匹配")
    
    # 创建基础 DataFrame
    df = pd.DataFrame({
        'text': texts,
        'id': range(n_samples)
    })
    
    # 添加元数据列
    if metadata:
        for key, values in metadata.items():
            if len(values) != n_samples:
                raise ValueError(f"元数据 '{key}' 的长度 ({len(values)}) 与样本数不匹配")
            df[key] = values
    
    # Embedding Atlas 会自动计算投影，但我们也可以手动添加
    # 注意：Embedding Atlas 会自动使用 UMAP 进行投影，所以这里主要是保存原始 embedding
    # 将 embedding 向量转换为列表格式存储
    df['embedding'] = embeddings.tolist()
    
    return df


def visualize_with_widget(
    df: pd.DataFrame,
    text_column: str = "text",
    x_column: Optional[str] = None,
    y_column: Optional[str] = None,
    neighbors_column: Optional[str] = None
):
    """
    在 Jupyter Notebook 中使用 EmbeddingAtlasWidget 进行可视化
    
    注意：此函数只能在 Jupyter Notebook 环境中使用
    
    Args:
        df: 包含 embedding 数据和元数据的 DataFrame
        text_column: 文本列的列名
        x_column: 投影 x 坐标列（可选，如果 None 则自动计算）
        y_column: 投影 y 坐标列（可选，如果 None 则自动计算）
        neighbors_column: 最近邻数据列（可选）
    """
    if not EMBEDDING_ATLAS_AVAILABLE:
        raise ImportError("embedding-atlas 未安装。请运行: pip install embedding-atlas")
    
    widget = EmbeddingAtlasWidget(
        df,
        text=text_column,
        x=x_column,
        y=y_column,
        neighbors=neighbors_column
    )
    return widget


def save_for_command_line_tool(df: pd.DataFrame, output_path: str):
    """
    将 DataFrame 保存为 Parquet 格式，以便使用命令行工具可视化
    
    命令行用法：
    embedding-atlas path_to_file.parquet
    
    Args:
        df: 包含 embedding 数据的 DataFrame
        output_path: 输出 Parquet 文件路径
    """
    # 确保输出目录存在
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为 Parquet
    df.to_parquet(output_path, index=False)
    print(f"数据已保存到: {output_path}")
    print(f"使用以下命令进行可视化：")
    print(f"  embedding-atlas {output_path}")


def example_visualize_social_platform_embeddings(
    user_bios: List[str],
    user_embeddings: np.ndarray,
    user_ids: Optional[List[str]] = None,
    output_dir: str = "analysis/embedding_visualizations"
):
    """
    示例：可视化社交平台用户的 embedding
    
    Args:
        user_bios: 用户简介文本列表
        user_embeddings: 用户 embedding 向量数组
        user_ids: 可选的用户 ID 列表
        output_dir: 输出目录
    """
    if not EMBEDDING_ATLAS_AVAILABLE:
        print("错误: embedding-atlas 未安装")
        return
    
    # 准备元数据
    metadata = {}
    if user_ids:
        metadata['user_id'] = user_ids
    
    # 创建 DataFrame
    df = prepare_embedding_dataframe(
        embeddings=user_embeddings,
        texts=user_bios,
        metadata=metadata
    )
    
    # 保存为 Parquet 以便使用命令行工具
    output_path = Path(output_dir) / "user_embeddings.parquet"
    save_for_command_line_tool(df, str(output_path))
    
    print("\n在 Jupyter Notebook 中使用以下代码进行交互式可视化：")
    print(f"""
from analysis.embedding_analysis import visualize_with_widget
import pandas as pd

df = pd.read_parquet('{output_path}')
widget = visualize_with_widget(df)
widget  # 在 Notebook 中显示
    """)
    
    return df


def example_visualize_post_embeddings(
    post_contents: List[str],
    post_embeddings: np.ndarray,
    post_ids: Optional[List[str]] = None,
    author_ids: Optional[List[str]] = None,
    output_dir: str = "analysis/embedding_visualizations"
):
    """
    示例：可视化帖子/内容的 embedding
    
    Args:
        post_contents: 帖子内容文本列表
        post_embeddings: 帖子 embedding 向量数组
        post_ids: 可选的帖子 ID 列表
        author_ids: 可选的作者 ID 列表
        output_dir: 输出目录
    """
    if not EMBEDDING_ATLAS_AVAILABLE:
        print("错误: embedding-atlas 未安装")
        return
    
    # 准备元数据
    metadata = {}
    if post_ids:
        metadata['post_id'] = post_ids
    if author_ids:
        metadata['author_id'] = author_ids
    
    # 创建 DataFrame
    df = prepare_embedding_dataframe(
        embeddings=post_embeddings,
        texts=post_contents,
        metadata=metadata
    )
    
    # 保存为 Parquet
    output_path = Path(output_dir) / "post_embeddings.parquet"
    save_for_command_line_tool(df, str(output_path))
    
    print("\n在 Jupyter Notebook 中使用以下代码进行交互式可视化：")
    print(f"""
from analysis.embedding_analysis import visualize_with_widget
import pandas as pd

df = pd.read_parquet('{output_path}')
widget = visualize_with_widget(df)
widget  # 在 Notebook 中显示
    """)
    
    return df


def extract_agent_actions_from_log(log_file_path: str) -> List[Dict]:
    """
    从 log 文件中提取所有 Agent 动作的参数和 reasoning
    
    Args:
        log_file_path: log 文件路径
        
    Returns:
        包含所有动作信息的字典列表，每个字典包含：
        - agent_id: Agent ID
        - timestamp: 时间戳
        - action: 动作名称
        - args: 动作参数
        - reasoning: 推理过程
    """
    actions_data = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则表达式匹配模式
    # 匹配格式: INFO - TIMESTAMP - social.agent - Agent ID performed action: ACTION_NAME with args: {...}and reasoning: <think>...</think>
    pattern = r'INFO - ([\d\-:\s,]+) - social\.agent - Agent (\d+) performed action: (\w+) with args: ({.*?})and reasoning: <think>(.*?)</think>'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        timestamp = match.group(1).strip()
        agent_id = int(match.group(2))
        action_name = match.group(3)
        args_str = match.group(4)
        reasoning = match.group(5).strip()
        
        # 解析参数字典
        try:
            # 替换单引号为双引号以便 JSON 解析
            args_str_fixed = args_str.replace("'", '"').replace('None', 'null')
            args_dict = json.loads(args_str_fixed)
        except json.JSONDecodeError:
            # 如果解析失败，使用 eval（不太安全但对于这种情况可以接受）
            try:
                args_dict = eval(args_str)
            except:
                args_dict = {"raw": args_str}
        
        actions_data.append({
            'agent_id': agent_id,
            'timestamp': timestamp,
            'action': action_name,
            'args': args_dict,
            'reasoning': reasoning
        })
    
    return actions_data


def extract_agent_actions_to_dataframe(log_file_path: str) -> pd.DataFrame:
    """
    从 log 文件中提取 Agent 动作并转换为 DataFrame
    
    Args:
        log_file_path: log 文件路径
        
    Returns:
        包含所有动作信息的 DataFrame
    """
    actions_data = extract_agent_actions_from_log(log_file_path)
    
    if not actions_data:
        print("警告: 未找到任何 Agent 动作记录")
        return pd.DataFrame()
    
    # 将嵌套的 args 字典展开为单独的列
    df = pd.DataFrame(actions_data)
    
    # 将 args 字典展开为单独的列
    args_df = pd.json_normalize(df['args'])
    args_df.columns = ['args_' + col for col in args_df.columns]
    
    # 合并到原始 DataFrame
    df = pd.concat([df.drop('args', axis=1), args_df], axis=1)
    
    return df


def analyze_agent_reasoning_patterns(log_file_path: str, output_dir: str = "analysis/agent_analysis"):
    """
    分析 Agent 的 reasoning 模式并生成报告
    
    Args:
        log_file_path: log 文件路径
        output_dir: 输出目录
    """
    print(f"正在从 {log_file_path} 提取 Agent 动作...")
    
    # 提取数据
    actions_data = extract_agent_actions_from_log(log_file_path)
    
    if not actions_data:
        print("错误: 未找到任何 Agent 动作记录")
        return
    
    print(f"成功提取 {len(actions_data)} 条动作记录")
    
    # 创建 DataFrame
    df = extract_agent_actions_to_dataframe(log_file_path)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存原始数据
    csv_path = output_path / "agent_actions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n动作数据已保存到: {csv_path}")
    
    # 保存详细的 reasoning 数据
    reasoning_path = output_path / "agent_reasoning.json"
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        json.dump(actions_data, f, indent=2, ensure_ascii=False)
    print(f"推理详情已保存到: {reasoning_path}")
    
    # 生成统计报告
    print("\n" + "="*60)
    print("Agent 动作统计分析")
    print("="*60)
    
    print(f"\n总动作数: {len(df)}")
    print(f"涉及 Agent 数: {df['agent_id'].nunique()}")
    print(f"\n动作类型分布:")
    print(df['action'].value_counts())
    
    print(f"\n每个 Agent 的动作次数:")
    print(df['agent_id'].value_counts().sort_index())
    
    # 如果有特定的动作参数，显示分布
    if 'args_advertised_quality' in df.columns:
        print(f"\n广告质量分布:")
        print(df['args_advertised_quality'].value_counts())
    
    if 'args_product_quality' in df.columns:
        print(f"\n实际产品质量分布:")
        print(df['args_product_quality'].value_counts())
    
    # 分析 reasoning 长度
    df['reasoning_length'] = df['reasoning'].str.len()
    print(f"\nReasoning 长度统计:")
    print(f"平均长度: {df['reasoning_length'].mean():.0f} 字符")
    print(f"最短: {df['reasoning_length'].min()} 字符")
    print(f"最长: {df['reasoning_length'].max()} 字符")
    
    print("\n" + "="*60)
    
    return df, actions_data


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="使用 Embedding Atlas 进行 embedding 可视化 或 分析 Agent log 文件"
    )
    parser.add_argument(
        "--prepare-example",
        action="store_true",
        help="准备示例数据（需要从实际项目中加载 embedding 数据）"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/embedding_visualizations",
        help="输出目录（默认: analysis/embedding_visualizations）"
    )
    parser.add_argument(
        "--analyze-log",
        type=str,
        help="分析指定的 log 文件，提取 Agent 动作和 reasoning"
    )
    parser.add_argument(
        "--log-output-dir",
        default="analysis/agent_analysis",
        help="log 分析结果输出目录（默认: analysis/agent_analysis）"
    )
    
    args = parser.parse_args()
    
    if args.analyze_log:
        # 分析 log 文件
        analyze_agent_reasoning_patterns(args.analyze_log, args.log_output_dir)
    elif args.prepare_example:
        print("准备示例数据...")
        # 这里可以添加实际的示例代码，从项目中加载 embedding 数据
        print("请参考示例函数：example_visualize_social_platform_embeddings 和 example_visualize_post_embeddings")
    else:
        print("Embedding Atlas 可视化工具 & Agent Log 分析工具")
        print("\n使用方法：")
        print("1. 分析 log 文件:")
        print("   python embedding_analysis.py --analyze-log log/social.agent-2025-11-04_10-19-13.log")
        print("\n2. Embedding 可视化:")
        print("   - 命令行工具：embedding-atlas <parquet_file>")
        print("   - Python Notebook Widget：使用 visualize_with_widget() 函数")
        print("   - Streamlit 组件：使用 embedding_atlas() 函数")
        print("\n详细文档：https://apple.github.io/embedding-atlas/overview.html")


if __name__ == "__main__":
    main()

