#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MVKAN 注意力可视化示例脚本 (Attention Visualization Example Script)

该脚本展示如何使用 visualize_attention 模块在分子结构上可视化模型的注意力权重。

功能说明:
1. 从训练好的模型中提取注意力权重
2. 在分子结构上可视化注意力分布
3. 生成高质量的可视化图像

使用方法:
    # 基本用法 - 使用示例分子生成可视化
    python examples/visualize_attention_example.py
    
    # 指定输出目录
    python examples/visualize_attention_example.py --output_dir results/my_visualization
    
    # 使用自定义 SMILES
    python examples/visualize_attention_example.py --smiles "CCO" "c1ccccc1"

作者: MVKAN Team
"""

import os
import sys
import argparse
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 尝试从 molgraph 包导入，如果失败则使用直接加载
try:
    from molgraph.visualize_attention import (
        visualize_molecule_attention,
        visualize_multiple_molecules,
        save_attention_visualization,
        quick_visualize,
        generate_example_visualizations,
        normalize_attention_weights,
        create_atom_attention_mask
    )
except ImportError:
    # 如果 molgraph 包有未安装的依赖，直接加载 visualize_attention 模块
    import importlib.util
    viz_spec = importlib.util.spec_from_file_location(
        'visualize_attention', 
        os.path.join(project_root, 'molgraph', 'visualize_attention.py')
    )
    viz_module = importlib.util.module_from_spec(viz_spec)
    viz_spec.loader.exec_module(viz_module)
    
    # 从模块导入需要的函数
    visualize_molecule_attention = viz_module.visualize_molecule_attention
    visualize_multiple_molecules = viz_module.visualize_multiple_molecules
    save_attention_visualization = viz_module.save_attention_visualization
    quick_visualize = viz_module.quick_visualize
    generate_example_visualizations = viz_module.generate_example_visualizations
    normalize_attention_weights = viz_module.normalize_attention_weights
    create_atom_attention_mask = viz_module.create_atom_attention_mask

# RDKit
from rdkit import Chem


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='MVKAN 注意力可视化示例脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成默认示例可视化
  python visualize_attention_example.py
  
  # 自定义分子
  python visualize_attention_example.py --smiles "CCO" "c1ccccc1O"
  
  # 指定输出目录和颜色映射
  python visualize_attention_example.py --output_dir results/custom --cmap YlOrRd
        """
    )
    
    parser.add_argument(
        '--smiles', 
        nargs='+', 
        type=str, 
        default=None,
        help='要可视化的 SMILES 字符串列表 (默认使用示例分子)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='results/attention_visualization',
        help='输出目录 (默认: results/attention_visualization)'
    )
    
    parser.add_argument(
        '--cmap', 
        type=str, 
        default='RdBu_r',
        choices=['RdBu_r', 'coolwarm', 'YlOrRd', 'Blues', 'Greens', 'viridis'],
        help='颜色映射 (默认: RdBu_r，蓝色->红色)'
    )
    
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300,
        help='输出图像分辨率 (默认: 300)'
    )
    
    parser.add_argument(
        '--format', 
        type=str, 
        default='png',
        choices=['png', 'svg', 'pdf'],
        help='输出格式 (默认: png)'
    )
    
    return parser.parse_args()


def generate_simulated_attention(smiles: str, seed: int = None) -> np.ndarray:
    """
    为分子生成模拟的注意力权重。
    
    这个函数模拟模型可能关注的分子区域:
    - 杂原子(O, N, S)通常有较高的注意力
    - 芳香环原子有中等注意力
    - 功能团连接点有较高注意力
    
    Args:
        smiles: SMILES 字符串
        seed: 随机种子 (用于可重复性)
    
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
        # Use bitwise AND to ensure non-negative 32-bit value for reproducibility
        np.random.seed(hash(smiles) & 0xFFFFFFFF)
    
    # 基础随机权重
    weights = np.random.uniform(0.1, 0.4, num_atoms)
    
    # 根据原子特性调整权重
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        
        # 杂原子权重增加
        if symbol in ['O', 'N']:
            weights[idx] += 0.4
        elif symbol in ['S', 'P', 'F', 'Cl', 'Br', 'I']:
            weights[idx] += 0.3
        
        # 芳香原子
        if atom.GetIsAromatic():
            weights[idx] += 0.15
        
        # 连接到多个非氢原子的原子（分子骨架关键点）
        heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() != 'H')
        if heavy_neighbors >= 3:
            weights[idx] += 0.2
        
        # 双键或三键的原子
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                weights[idx] += 0.15
            elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                weights[idx] += 0.2
    
    # 确保权重在 [0, 1] 范围内
    weights = np.clip(weights, 0, 1)
    
    return weights


def example_single_molecule():
    """
    示例1: 单个分子的注意力可视化
    """
    print("\n" + "="*60)
    print("示例 1: 单个分子注意力可视化")
    print("="*60)
    
    # 阿司匹林
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    
    print(f"\n分子: 阿司匹林 (Aspirin)")
    print(f"SMILES: {smiles}")
    print(f"原子数: {mol.GetNumAtoms()}")
    
    # 生成模拟注意力权重
    weights = generate_simulated_attention(smiles, seed=42)
    print(f"\n模拟注意力权重:")
    for i, w in enumerate(weights):
        atom = mol.GetAtomWithIdx(i)
        print(f"  原子 {i} ({atom.GetSymbol()}): {w:.3f}")
    
    # 可视化
    fig = visualize_molecule_attention(
        smiles,
        weights,
        title="Aspirin - Attention Visualization",
        cmap_name='RdBu_r',
        show_colorbar=True,
        show_smiles=True,
        figsize=(10, 8)
    )
    
    # 保存
    output_path = "results/attention_visualization/example_aspirin.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_attention_visualization(fig, output_path, dpi=300)
    
    print(f"\n✓ 可视化已保存至: {output_path}")
    
    return fig


def example_multiple_molecules():
    """
    示例2: 多个分子的批量可视化
    """
    print("\n" + "="*60)
    print("示例 2: 多个分子批量可视化")
    print("="*60)
    
    # 定义多个分子
    molecules_info = [
        {
            'name': '咖啡因 (Caffeine)',
            'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
        },
        {
            'name': '布洛芬 (Ibuprofen)',
            'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
        },
        {
            'name': '对乙酰氨基酚 (Paracetamol)',
            'smiles': 'CC(=O)NC1=CC=C(O)C=C1'
        },
        {
            'name': '苯酚 (Phenol)',
            'smiles': 'Oc1ccccc1'
        }
    ]
    
    # 准备可视化数据
    molecules = []
    for mol_info in molecules_info:
        smiles = mol_info['smiles']
        weights = generate_simulated_attention(smiles)
        
        if weights is not None:
            molecules.append({
                'smiles': smiles,
                'weights': weights,
                'title': mol_info['name']
            })
            print(f"✓ 准备: {mol_info['name']}")
    
    # 批量可视化
    output_path = "results/attention_visualization/example_multiple_molecules.png"
    fig = visualize_multiple_molecules(
        molecules,
        ncols=2,
        figsize=(12, 10),
        cmap_name='RdBu_r',
        save_path=output_path,
        dpi=300
    )
    
    print(f"\n✓ 批量可视化已保存至: {output_path}")
    
    return fig


def example_different_colormaps():
    """
    示例3: 不同颜色映射的对比
    """
    print("\n" + "="*60)
    print("示例 3: 不同颜色映射对比")
    print("="*60)
    
    smiles = "c1ccc(c(c1)C(=O)O)O"  # 水杨酸
    weights = generate_simulated_attention(smiles, seed=123)
    
    colormaps = ['RdBu_r', 'coolwarm', 'YlOrRd', 'Blues']
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    
    for ax, cmap in zip(axes.flat, colormaps):
        # 创建单独的可视化
        sub_fig = visualize_molecule_attention(
            smiles,
            weights,
            title=f"Colormap: {cmap}",
            cmap_name=cmap,
            show_colorbar=True,
            show_smiles=False,
            figsize=(5, 4)
        )
        
        # 保存临时图像并加载 (use tempfile for cross-platform compatibility)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
        sub_fig.savefig(temp_path, dpi=100, bbox_inches='tight')
        plt.close(sub_fig)
        
        img = plt.imread(temp_path)
        ax.imshow(img)
        ax.set_title(f"Colormap: {cmap}", fontsize=12)
        ax.axis('off')
        
        os.remove(temp_path)
    
    plt.tight_layout()
    
    output_path = "results/attention_visualization/example_colormaps.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✓ 颜色映射对比已保存至: {output_path}")
    
    return fig


def example_custom_smiles(smiles_list: list, output_dir: str, cmap: str = 'RdBu_r'):
    """
    示例4: 自定义 SMILES 可视化
    
    Args:
        smiles_list: SMILES 字符串列表
        output_dir: 输出目录
        cmap: 颜色映射
    """
    print("\n" + "="*60)
    print("示例 4: 自定义分子可视化")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    for i, smiles in enumerate(smiles_list):
        print(f"\n处理分子 {i+1}: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  ✗ 无效的 SMILES: {smiles}")
            continue
        
        # 生成模拟注意力权重
        weights = generate_simulated_attention(smiles)
        
        # 可视化
        fig = visualize_molecule_attention(
            smiles,
            weights,
            title=f"Molecule {i+1}",
            cmap_name=cmap,
            show_colorbar=True,
            show_smiles=True
        )
        
        # 保存
        output_path = os.path.join(output_dir, f"custom_molecule_{i+1}.png")
        save_attention_visualization(fig, output_path, dpi=300)
        saved_files.append(output_path)
        
        plt.close(fig)
        print(f"  ✓ 已保存: {output_path}")
    
    return saved_files


def main():
    """主函数"""
    args = parse_arguments()
    
    print("="*60)
    print("MVKAN 注意力可视化示例")
    print("="*60)
    print(f"\n输出目录: {args.output_dir}")
    print(f"颜色映射: {args.cmap}")
    print(f"输出格式: {args.format}")
    print(f"分辨率: {args.dpi} DPI")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    
    if args.smiles:
        # 使用自定义 SMILES
        print(f"\n使用自定义分子 ({len(args.smiles)} 个)...")
        example_custom_smiles(args.smiles, args.output_dir, args.cmap)
    else:
        # 运行所有示例
        print("\n运行默认示例...")
        
        # 示例 1: 单个分子
        try:
            example_single_molecule()
        except Exception as e:
            print(f"示例1 出错: {e}")
        
        plt.close('all')
        
        # 示例 2: 多个分子
        try:
            example_multiple_molecules()
        except Exception as e:
            print(f"示例2 出错: {e}")
        
        plt.close('all')
        
        # 示例 3: 颜色映射对比
        try:
            example_different_colormaps()
        except Exception as e:
            print(f"示例3 出错: {e}")
        
        plt.close('all')
        
        # 生成更多示例
        print("\n" + "="*60)
        print("生成额外示例可视化")
        print("="*60)
        try:
            generate_example_visualizations(args.output_dir)
        except Exception as e:
            print(f"生成示例出错: {e}")
    
    print("\n" + "="*60)
    print("所有可视化已完成!")
    print(f"结果保存在: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
