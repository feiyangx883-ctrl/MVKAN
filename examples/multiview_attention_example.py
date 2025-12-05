#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MVKAN 多视图注意力可视化示例 (Multi-view Attention Visualization Example)

该脚本展示如何从训练好的模型中提取多视图注意力权重，
并生成可视化图像保存到结果文件夹中。

功能说明:
1. 从训练好的 Tester 对象提取注意力权重
2. 可视化多视图（原子图 + 子结构图）的注意力分布
3. 对比 MVKAN 模型与基准模型的注意力差异
4. 将结果保存到 results/attention_visualization/ 目录

使用方法:
    # 方法1: 使用模拟数据演示
    python examples/multiview_attention_example.py
    
    # 方法2: 集成到训练流程中（见下方代码示例）

作者: MVKAN Team
"""

import os
import sys
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 尝试导入可视化模块
try:
    from molgraph.visualize_attention import (
        visualize_molecule_attention,
        visualize_multiview_attention,
        visualize_attention_comparison,
        save_attention_visualization,
        normalize_attention_weights
    )
except ImportError:
    import importlib.util
    viz_spec = importlib.util.spec_from_file_location(
        'visualize_attention', 
        os.path.join(project_root, 'molgraph', 'visualize_attention.py')
    )
    viz_module = importlib.util.module_from_spec(viz_spec)
    viz_spec.loader.exec_module(viz_module)
    
    visualize_molecule_attention = viz_module.visualize_molecule_attention
    visualize_multiview_attention = viz_module.visualize_multiview_attention
    visualize_attention_comparison = viz_module.visualize_attention_comparison
    save_attention_visualization = viz_module.save_attention_visualization
    normalize_attention_weights = viz_module.normalize_attention_weights

from rdkit import Chem

# 设置 matplotlib 后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_simulated_attention(smiles: str, model_type: str = 'kan', seed: int = None) -> np.ndarray:
    """
    生成模拟的注意力权重用于演示。
    
    不同的 model_type 会生成不同的注意力分布：
    - 'kan': MVKAN 模型 - 更关注功能团和杂原子
    - 'baseline': 基准模型 - 更均匀的注意力分布
    
    Args:
        smiles: SMILES 字符串
        model_type: 模型类型 ('kan' 或 'baseline')
        seed: 随机种子
    
    Returns:
        注意力权重数组
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    num_atoms = mol.GetNumAtoms()
    
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(hash(smiles) & 0xFFFFFFFF)
    
    if model_type == 'kan':
        # MVKAN 模型: 更关注功能团和杂原子
        weights = np.random.uniform(0.1, 0.3, num_atoms)
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            
            # 杂原子权重显著增加
            if symbol in ['O', 'N']:
                weights[idx] += 0.5
            elif symbol in ['S', 'P', 'F', 'Cl', 'Br', 'I']:
                weights[idx] += 0.4
            
            # 芳香原子
            if atom.GetIsAromatic():
                weights[idx] += 0.2
            
            # 分支点
            heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() != 'H')
            if heavy_neighbors >= 3:
                weights[idx] += 0.3
    else:
        # 基准模型: 更均匀的注意力分布
        weights = np.random.uniform(0.3, 0.5, num_atoms)
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            
            # 仅轻微关注杂原子
            if symbol in ['O', 'N']:
                weights[idx] += 0.15
    
    return np.clip(weights, 0, 1)


def example_multiview_visualization():
    """
    示例: 多视图注意力可视化
    """
    print("\n" + "="*60)
    print("多视图注意力可视化示例")
    print("="*60)
    
    # 示例分子: 阿司匹林
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    
    print(f"\n分子: 阿司匹林 (Aspirin)")
    print(f"SMILES: {smiles}")
    print(f"原子数: {num_atoms}")
    
    # 生成多视图注意力权重
    np.random.seed(42)
    attention_dict = {
        'atom': generate_simulated_attention(smiles, 'kan', seed=42),
        'substructure': generate_simulated_attention(smiles, 'kan', seed=123),
    }
    
    print(f"\n原子视图注意力: {attention_dict['atom'][:5]}...")
    print(f"子结构视图注意力: {attention_dict['substructure'][:5]}...")
    
    # 可视化并保存
    output_path = "results/attention_visualization/multiview_aspirin.png"
    fig = visualize_multiview_attention(
        smiles,
        attention_dict,
        title="Multi-view Attention: Aspirin",
        output_path=output_path
    )
    plt.close(fig)
    
    print(f"\n✓ 多视图可视化已保存至: {output_path}")
    
    return output_path


def example_model_comparison():
    """
    示例: 对比 MVKAN 与基准模型的注意力差异
    """
    print("\n" + "="*60)
    print("MVKAN vs 基准模型 注意力对比示例")
    print("="*60)
    
    # 示例分子
    molecules = [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
    ]
    
    saved_files = []
    
    for smiles, name in molecules:
        mol = Chem.MolFromSmiles(smiles)
        print(f"\n分子: {name}")
        print(f"SMILES: {smiles}")
        
        # 生成两种模型的注意力
        kan_attention = generate_simulated_attention(smiles, 'kan', seed=42)
        baseline_attention = generate_simulated_attention(smiles, 'baseline', seed=42)
        
        print(f"MVKAN 注意力范围: [{kan_attention.min():.3f}, {kan_attention.max():.3f}]")
        print(f"基准模型注意力范围: [{baseline_attention.min():.3f}, {baseline_attention.max():.3f}]")
        
        # 可视化对比
        output_path = f"results/attention_visualization/comparison_{name}.png"
        fig = visualize_attention_comparison(
            smiles,
            kan_attention,
            baseline_attention,
            title=f"Attention Comparison: {name.title()}",
            output_path=output_path
        )
        plt.close(fig)
        
        saved_files.append(output_path)
        print(f"✓ 对比可视化已保存至: {output_path}")
    
    return saved_files


def example_with_tester():
    """
    示例: 从训练好的模型提取注意力并可视化
    
    注意: 此函数需要已训练的模型，仅作为代码参考
    """
    print("\n" + "="*60)
    print("从训练模型提取注意力的代码示例 (参考)")
    print("="*60)
    
    example_code = '''
# ========================================
# 完整的从训练模型提取注意力的流程
# ========================================

from molgraph.testing import Tester
from molgraph.dataset import generateDataLoaderTesting
from molgraph.visualize_attention import (
    visualize_multiview_attention,
    visualize_attention_comparison
)
import numpy as np
import os

# 1. 加载训练好的模型
args_test = {
    'log_folder_name': trainer.log_folder_name,
    'exp_name': args.experiment_number,
    'fold_number': 0,
    'seed': args.seed
}
tester = Tester(args, args_test)

# 2. 运行测试并提取注意力权重
test_loader, datasets_test = generateDataLoaderTesting(args.file, args.batch_size)
tester.test(test_loader, return_attention_weights=True)

# 3. 获取注意力权重
att_mol = tester.getAttentionMol()
# att_mol = {'atom': (edge_index, weights), 'substructure': (edge_index, weights), ...}

# 4. 处理注意力权重
atom_weights = att_mol['atom'][1].cpu().numpy()  # 原子视图

# 5. 多视图注意力可视化
for i, sample in enumerate(datasets_test[:3]):  # 可视化前3个样本
    smiles = sample.smiles
    
    # 提取当前分子的注意力
    attention_dict = {}
    for view_name in att_mol:
        _, weights = att_mol[view_name]
        # 根据分子索引提取对应的权重
        attention_dict[view_name] = weights[i].cpu().numpy()
    
    # 生成可视化
    output_path = f"results/attention_visualization/sample_{i}_{view_name}.png"
    fig = visualize_multiview_attention(
        smiles,
        attention_dict,
        title=f"Multi-view Attention: Sample {i}",
        output_path=output_path
    )
    plt.close(fig)
    print(f"✓ 已保存: {output_path}")

# 6. 模型对比可视化（如果有基准模型）
# baseline_tester = Tester(baseline_args, baseline_args_test)
# baseline_tester.test(test_loader, return_attention_weights=True)
# baseline_att = baseline_tester.getAttentionMol()
# 
# fig = visualize_attention_comparison(
#     smiles,
#     kan_attention=att_mol['atom'][1][0].cpu().numpy(),
#     baseline_attention=baseline_att['atom'][1][0].cpu().numpy(),
#     output_path="results/attention_visualization/kan_vs_baseline.png"
# )
'''
    
    print(example_code)
    
    return None


def main():
    """主函数"""
    print("="*60)
    print("MVKAN 多视图注意力可视化")
    print("="*60)
    
    # 创建输出目录
    output_dir = "results/attention_visualization"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 示例1: 多视图注意力可视化
    example_multiview_visualization()
    
    # 示例2: 模型对比可视化
    example_model_comparison()
    
    # 示例3: 从训练模型提取注意力的代码示例
    example_with_tester()
    
    print("\n" + "="*60)
    print("所有可视化已完成!")
    print(f"结果保存在: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
