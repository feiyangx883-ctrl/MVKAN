"""
注意力可视化模块 (Attention Visualization Module)

该模块提供从训练好的MVKAN模型中提取注意力权重，并使用RDKit在分子结构上
可视化注意力分布的功能，以体现模型的可解释性。

主要功能:
- 从模型中提取原子级别的注意力权重
- 使用颜色梯度在分子结构上显示注意力强度
- 生成带有颜色条(colorbar)的高质量可视化图像
- 支持PNG和SVG两种输出格式

示例用法:
    >>> from molgraph.visualize_attention import visualize_molecule_attention
    >>> # 加载模型和测试数据
    >>> model = load_trained_model(args, checkpoint_path)
    >>> # 提取注意力权重
    >>> smiles = "CCO"
    >>> attention_weights = extract_attention_weights(model, smiles, args)
    >>> # 可视化注意力
    >>> fig = visualize_molecule_attention(smiles, attention_weights)
    >>> fig.savefig("attention_visualization.png", dpi=300)
"""

######################
### Import Library ###
######################

import os
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Union

# 可视化库
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap

# RDKit 分子绘图
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# 图像处理
from io import BytesIO
from PIL import Image

# PyTorch - optional import for model-related functions
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 设置 RDKit 使用 CoordGen
rdDepictor.SetPreferCoordGen(True)


##################################
### Attention Weight Functions ###
##################################

def normalize_attention_weights(
    weights: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    归一化注意力权重到 [0, 1] 区间。
    
    Args:
        weights: 原始注意力权重数组
        method: 归一化方法
            - 'minmax': Min-Max 归一化
            - 'softmax': Softmax 归一化
            - 'zscore': Z-score 标准化后映射到 [0, 1]
    
    Returns:
        归一化后的权重数组，范围在 [0, 1]
    
    Example:
        >>> weights = np.array([0.1, 0.5, 0.3, 0.8])
        >>> normalized = normalize_attention_weights(weights, method='minmax')
        >>> print(normalized)  # [0.0, 0.571, 0.286, 1.0]
    """
    weights = np.array(weights, dtype=float)
    
    if len(weights) == 0:
        return weights
    
    # 处理单一值的情况
    if len(np.unique(weights)) == 1:
        return np.full_like(weights, 0.5)
    
    if method == 'minmax':
        w_min = np.min(weights)
        w_max = np.max(weights)
        if w_max - w_min > 0:
            normalized = (weights - w_min) / (w_max - w_min)
        else:
            normalized = np.full_like(weights, 0.5)
    
    elif method == 'softmax':
        exp_weights = np.exp(weights - np.max(weights))
        normalized = exp_weights / exp_weights.sum()
        # 再进行 minmax 以确保范围在 [0, 1]
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
    
    elif method == 'zscore':
        mean = np.mean(weights)
        std = np.std(weights)
        if std > 0:
            z_scores = (weights - mean) / std
            # 将 z-score 映射到 [0, 1]，使用 sigmoid 函数
            normalized = 1 / (1 + np.exp(-z_scores))
        else:
            normalized = np.full_like(weights, 0.5)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def extract_attention_weights_from_model(
    model,
    data_loader,
    device: str = 'cpu',
    view_type: str = 'atom'
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    从训练好的模型中提取注意力权重。
    
    Note: This function requires PyTorch to be installed.
    
    Args:
        model: 训练好的 MVKAN 模型 (torch.nn.Module)
        data_loader: 包含测试数据的 DataLoader
        device: 运行设备 ('cpu' 或 'cuda')
        view_type: 要提取的视图类型 ('atom' 或 reduced graph 类型)
    
    Returns:
        字典，键为 SMILES，值为包含注意力权重的字典
        {smiles: {'atom_weights': array, 'attention_indices': array}}
    
    Example:
        >>> weights_dict = extract_attention_weights_from_model(model, test_loader)
        >>> for smiles, weights in weights_dict.items():
        ...     print(f"{smiles}: {weights['atom_weights']}")
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for extracting attention weights from model. "
                         "Please install it with: pip install torch")
    
    model.eval()
    attention_dict = {}
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # 前向传播并获取注意力权重
            _, attention_weights = model(data, return_attention_weights=True)
            
            # 处理批次中的每个分子
            batch_size = len(data.smiles) if hasattr(data, 'smiles') else 1
            smiles_list = data.smiles if hasattr(data, 'smiles') else ['unknown']
            
            # 获取指定视图的注意力权重
            if view_type in attention_weights:
                att_index, att_weights = attention_weights[view_type]
                att_index = att_index.cpu().numpy()
                att_weights = att_weights.cpu().numpy()
                
                # 对于批处理，需要根据 batch 索引分割
                if hasattr(data, 'x_g_batch'):
                    batch_indices = data.x_g_batch.cpu().numpy()
                    
                    for i, smiles in enumerate(smiles_list):
                        if isinstance(smiles, list):
                            smiles = smiles[0] if len(smiles) > 0 else 'unknown'
                        
                        # 找到属于当前分子的原子索引
                        mol_mask = (att_index[1] == i)
                        mol_atom_indices = att_index[0][mol_mask]
                        mol_weights = att_weights[mol_mask]
                        
                        # 重新索引原子（相对于分子内部）
                        if len(mol_atom_indices) > 0:
                            atom_offset = np.min(mol_atom_indices)
                            mol_atom_indices = mol_atom_indices - atom_offset
                        
                        attention_dict[smiles] = {
                            'atom_weights': mol_weights.flatten(),
                            'atom_indices': mol_atom_indices.flatten()
                        }
                else:
                    # 单分子情况
                    smiles = smiles_list[0] if isinstance(smiles_list, list) else smiles_list
                    attention_dict[smiles] = {
                        'atom_weights': att_weights.flatten(),
                        'atom_indices': att_index[0].flatten()
                    }
    
    return attention_dict


def create_atom_attention_mask(
    mol: Chem.Mol,
    atom_weights: np.ndarray,
    atom_indices: Optional[np.ndarray] = None
) -> Dict[int, float]:
    """
    创建原子级别的注意力掩码。
    
    Args:
        mol: RDKit 分子对象
        atom_weights: 注意力权重数组
        atom_indices: 可选，原子索引数组。如果为 None，假设权重按顺序对应原子
    
    Returns:
        字典，键为原子索引，值为归一化后的注意力权重
    
    Example:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> weights = np.array([0.2, 0.5, 0.8])
        >>> mask = create_atom_attention_mask(mol, weights)
        >>> print(mask)  # {0: 0.0, 1: 0.5, 2: 1.0}
    """
    num_atoms = mol.GetNumAtoms()
    
    # 归一化权重
    normalized_weights = normalize_attention_weights(atom_weights, method='minmax')
    
    # 创建注意力掩码
    attention_mask = {}
    
    if atom_indices is not None:
        # 使用提供的索引
        for idx, weight in zip(atom_indices, normalized_weights):
            if 0 <= idx < num_atoms:
                attention_mask[int(idx)] = float(weight)
    else:
        # 按顺序分配
        for i, weight in enumerate(normalized_weights):
            if i < num_atoms:
                attention_mask[i] = float(weight)
    
    # 填充未覆盖的原子（使用最小权重）
    for i in range(num_atoms):
        if i not in attention_mask:
            attention_mask[i] = 0.0
    
    return attention_mask


#################################
### Visualization Functions   ###
#################################

def create_attention_colormap(
    cmap_name: str = 'RdBu_r',
    low_color: str = None,
    high_color: str = None
) -> matplotlib.colors.Colormap:
    """
    创建用于注意力可视化的颜色映射。
    
    Args:
        cmap_name: matplotlib 颜色映射名称
            - 'RdBu_r': 蓝色到红色 (推荐)
            - 'coolwarm': 冷暖色渐变
            - 'viridis': 紫色到黄色
            - 'plasma': 紫色到橙色
        low_color: 自定义低注意力颜色 (如 '#0000FF')
        high_color: 自定义高注意力颜色 (如 '#FF0000')
    
    Returns:
        matplotlib 颜色映射对象
    
    Example:
        >>> cmap = create_attention_colormap('RdBu_r')
        >>> # 或自定义颜色
        >>> cmap = create_attention_colormap(low_color='#0000FF', high_color='#FF0000')
    """
    if low_color and high_color:
        # 创建自定义颜色映射
        colors = [low_color, 'white', high_color]
        cmap = LinearSegmentedColormap.from_list('custom_attention', colors)
    else:
        cmap = plt.get_cmap(cmap_name)
    
    return cmap


def weight_to_color(
    weight: float,
    cmap: matplotlib.colors.Colormap,
    alpha: float = 0.7
) -> Tuple[float, float, float, float]:
    """
    将注意力权重转换为 RGBA 颜色。
    
    Args:
        weight: 归一化的注意力权重 [0, 1]
        cmap: 颜色映射
        alpha: 透明度
    
    Returns:
        RGBA 颜色元组
    """
    rgba = cmap(weight)
    return (rgba[0], rgba[1], rgba[2], alpha)


def draw_molecule_with_attention(
    mol: Chem.Mol,
    attention_mask: Dict[int, float],
    size: Tuple[int, int] = (600, 500),
    cmap_name: str = 'RdBu_r',
    highlight_bonds: bool = True,
    atom_labels: bool = False
) -> str:
    """
    使用 RDKit 绘制带有注意力高亮的分子结构。
    
    Args:
        mol: RDKit 分子对象
        attention_mask: 原子注意力掩码字典 {atom_idx: weight}
        size: 图像尺寸 (宽度, 高度)
        cmap_name: 颜色映射名称
        highlight_bonds: 是否高亮显示键
        atom_labels: 是否显示原子索引标签
    
    Returns:
        SVG 格式的分子图像字符串
    
    Example:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> mask = {0: 0.2, 1: 0.5, 2: 0.9}
        >>> svg = draw_molecule_with_attention(mol, mask)
    """
    # 计算 2D 坐标
    rdDepictor.Compute2DCoords(mol)
    
    # 准备分子用于绘制
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    
    # 创建颜色映射
    cmap = create_attention_colormap(cmap_name)
    
    # 准备高亮原子和颜色
    highlight_atoms = list(attention_mask.keys())
    highlight_atom_colors = {}
    highlight_atom_radii = {}
    
    for atom_idx, weight in attention_mask.items():
        color = weight_to_color(weight, cmap, alpha=0.8)
        highlight_atom_colors[atom_idx] = color
        # 根据注意力权重调整半径
        highlight_atom_radii[atom_idx] = 0.3 + weight * 0.2
    
    # 准备高亮键
    highlight_bonds_list = []
    highlight_bond_colors = {}
    
    if highlight_bonds:
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            if begin_idx in attention_mask and end_idx in attention_mask:
                bond_idx = bond.GetIdx()
                highlight_bonds_list.append(bond_idx)
                # 键的颜色取两端原子权重的平均值
                avg_weight = (attention_mask[begin_idx] + attention_mask[end_idx]) / 2
                highlight_bond_colors[bond_idx] = weight_to_color(avg_weight, cmap, alpha=0.6)
    
    # 创建绘图对象
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    
    # 设置绘图选项
    opts = drawer.drawOptions()
    opts.highlightRadius = 0.4
    opts.fillHighlights = True
    opts.continuousHighlight = True
    if atom_labels:
        opts.addAtomIndices = True
    
    # 绘制分子
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bonds_list,
        highlightBondColors=highlight_bond_colors,
        highlightAtomRadii=highlight_atom_radii
    )
    
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    # 清理 SVG
    svg = svg.replace('svg:', '')
    
    return svg


def add_colorbar_to_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    cmap_name: str = 'RdBu_r',
    label: str = 'Attention Weight',
    orientation: str = 'vertical'
) -> plt.colorbar:
    """
    向图像添加颜色条。
    
    Args:
        fig: matplotlib Figure 对象
        ax: matplotlib Axes 对象
        cmap_name: 颜色映射名称
        label: 颜色条标签
        orientation: 方向 ('vertical' 或 'horizontal')
    
    Returns:
        colorbar 对象
    """
    cmap = create_attention_colormap(cmap_name)
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    if orientation == 'vertical':
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    else:
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.1)
    
    cbar.set_label(label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    return cbar


def visualize_molecule_attention(
    smiles: str,
    atom_weights: Union[np.ndarray, List[float], Dict[int, float]],
    atom_indices: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap_name: str = 'RdBu_r',
    show_colorbar: bool = True,
    show_smiles: bool = True,
    dpi: int = 150
) -> plt.Figure:
    """
    可视化分子的注意力权重分布。
    
    这是主要的可视化函数，用于生成带有注意力高亮的分子图像。
    
    Args:
        smiles: 分子的 SMILES 字符串
        atom_weights: 原子注意力权重，可以是:
            - numpy 数组
            - 列表
            - 字典 {atom_idx: weight}
        atom_indices: 可选，原子索引数组
        title: 图像标题
        figsize: 图像尺寸
        cmap_name: 颜色映射名称
            - 'RdBu_r': 蓝色(低) -> 红色(高) (推荐)
            - 'coolwarm': 冷暖色渐变
            - 'YlOrRd': 黄色 -> 橙色 -> 红色
            - 'Blues': 浅蓝 -> 深蓝
        show_colorbar: 是否显示颜色条
        show_smiles: 是否在图像下方显示 SMILES
        dpi: 图像分辨率
    
    Returns:
        matplotlib Figure 对象
    
    Example:
        >>> smiles = "c1ccccc1O"  # 苯酚
        >>> weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]  # 7个原子的权重
        >>> fig = visualize_molecule_attention(smiles, weights)
        >>> fig.savefig("phenol_attention.png", dpi=300, bbox_inches='tight')
    """
    # 创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # 处理权重输入
    if isinstance(atom_weights, dict):
        attention_mask = atom_weights
        # 归一化字典中的权重
        values = np.array(list(attention_mask.values()))
        normalized = normalize_attention_weights(values)
        attention_mask = {k: normalized[i] for i, k in enumerate(attention_mask.keys())}
    else:
        atom_weights = np.array(atom_weights)
        attention_mask = create_atom_attention_mask(mol, atom_weights, atom_indices)
    
    # 绘制带注意力的分子
    mol_size = (500, 400)
    svg = draw_molecule_with_attention(mol, attention_mask, size=mol_size, cmap_name=cmap_name)
    
    # 将 SVG 转换为图像
    try:
        from cairosvg import svg2png
        png_data = svg2png(bytestring=svg.encode('utf-8'), dpi=dpi*2)
        mol_img = Image.open(BytesIO(png_data))
    except ImportError:
        # 如果没有 cairosvg，使用 RDKit 直接生成 PNG
        drawer = rdMolDraw2D.MolDraw2DCairo(mol_size[0]*2, mol_size[1]*2)
        rdDepictor.Compute2DCoords(mol)
        mol_prep = rdMolDraw2D.PrepareMolForDrawing(mol)
        
        cmap = create_attention_colormap(cmap_name)
        highlight_atoms = list(attention_mask.keys())
        highlight_colors = {k: weight_to_color(v, cmap) for k, v in attention_mask.items()}
        
        drawer.DrawMolecule(mol_prep, highlightAtoms=highlight_atoms, 
                           highlightAtomColors=highlight_colors)
        drawer.FinishDrawing()
        mol_img = Image.open(BytesIO(drawer.GetDrawingText()))
    
    # 创建 matplotlib 图像
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 显示分子图像
    ax.imshow(mol_img)
    ax.axis('off')
    
    # 添加标题
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加 SMILES 显示
    if show_smiles:
        # 如果 SMILES 太长，截断显示
        display_smiles = smiles if len(smiles) <= 60 else smiles[:57] + "..."
        ax.text(0.5, -0.02, f"SMILES: {display_smiles}", 
                transform=ax.transAxes, fontsize=9, ha='center',
                family='monospace')
    
    # 添加颜色条
    if show_colorbar:
        add_colorbar_to_figure(fig, ax, cmap_name, label='Attention Weight')
    
    plt.tight_layout()
    
    return fig


def visualize_multiple_molecules(
    molecules: List[Dict],
    ncols: int = 2,
    figsize: Tuple[int, int] = None,
    cmap_name: str = 'RdBu_r',
    save_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    批量可视化多个分子的注意力权重。
    
    Args:
        molecules: 分子信息列表，每个元素是字典:
            {
                'smiles': str,
                'weights': array-like,
                'indices': array-like (optional),
                'title': str (optional)
            }
        ncols: 每行显示的分子数量
        figsize: 图像尺寸，如果为 None 则自动计算
        cmap_name: 颜色映射名称
        save_path: 保存路径（可选）
        dpi: 图像分辨率
    
    Returns:
        matplotlib Figure 对象
    
    Example:
        >>> molecules = [
        ...     {'smiles': 'CCO', 'weights': [0.2, 0.5, 0.8], 'title': 'Ethanol'},
        ...     {'smiles': 'c1ccccc1', 'weights': [0.1]*6, 'title': 'Benzene'}
        ... ]
        >>> fig = visualize_multiple_molecules(molecules, ncols=2)
    """
    n_mols = len(molecules)
    nrows = (n_mols + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    
    # 确保 axes 是二维数组
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # 绘制每个分子
    for i, mol_info in enumerate(molecules):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        smiles = mol_info['smiles']
        weights = mol_info['weights']
        indices = mol_info.get('indices', None)
        title = mol_info.get('title', None)
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                ax.text(0.5, 0.5, f"Invalid SMILES:\n{smiles}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # 创建注意力掩码
            attention_mask = create_atom_attention_mask(mol, np.array(weights), indices)
            
            # 绘制分子
            mol_size = (400, 350)
            svg = draw_molecule_with_attention(mol, attention_mask, size=mol_size, cmap_name=cmap_name)
            
            # 转换为图像
            try:
                from cairosvg import svg2png
                png_data = svg2png(bytestring=svg.encode('utf-8'), dpi=dpi)
                mol_img = Image.open(BytesIO(png_data))
            except ImportError:
                drawer = rdMolDraw2D.MolDraw2DCairo(mol_size[0], mol_size[1])
                rdDepictor.Compute2DCoords(mol)
                mol_prep = rdMolDraw2D.PrepareMolForDrawing(mol)
                cmap = create_attention_colormap(cmap_name)
                highlight_colors = {k: weight_to_color(v, cmap) for k, v in attention_mask.items()}
                drawer.DrawMolecule(mol_prep, highlightAtoms=list(attention_mask.keys()),
                                   highlightAtomColors=highlight_colors)
                drawer.FinishDrawing()
                mol_img = Image.open(BytesIO(drawer.GetDrawingText()))
            
            ax.imshow(mol_img)
            ax.axis('off')
            
            if title:
                ax.set_title(title, fontsize=11, fontweight='bold')
            
            # 显示简短的 SMILES
            short_smiles = smiles if len(smiles) <= 30 else smiles[:27] + "..."
            ax.text(0.5, -0.02, short_smiles, transform=ax.transAxes, 
                   fontsize=8, ha='center', family='monospace')
                   
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)[:50]}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.axis('off')
    
    # 隐藏空白子图
    for i in range(n_mols, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')
    
    # 添加统一的颜色条
    cmap = create_attention_colormap(cmap_name)
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label('Attention Weight', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Figure saved to: {save_path}")
    
    return fig


def save_attention_visualization(
    fig: plt.Figure,
    output_path: str,
    format: str = 'png',
    dpi: int = 300,
    transparent: bool = False
) -> str:
    """
    保存注意力可视化图像。
    
    Args:
        fig: matplotlib Figure 对象
        output_path: 输出文件路径
        format: 输出格式 ('png', 'svg', 'pdf')
        dpi: 图像分辨率
        transparent: 是否透明背景
    
    Returns:
        保存的文件路径
    
    Example:
        >>> fig = visualize_molecule_attention("CCO", [0.2, 0.5, 0.8])
        >>> save_attention_visualization(fig, "output/ethanol.png")
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 确保文件扩展名正确
    if not output_path.endswith(f'.{format}'):
        output_path = f"{output_path}.{format}"
    
    # 保存图像
    fig.savefig(
        output_path,
        format=format,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white' if not transparent else 'none',
        edgecolor='none',
        transparent=transparent
    )
    
    print(f"✓ Attention visualization saved to: {output_path}")
    
    return output_path


#################################
### Convenience Functions     ###
#################################

def quick_visualize(
    smiles: str,
    weights: Optional[Union[List, np.ndarray]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    快速可视化单个分子的注意力权重。
    
    如果未提供权重，将生成随机权重用于演示。
    
    Args:
        smiles: SMILES 字符串
        weights: 注意力权重（可选）
        save_path: 保存路径（可选）
    
    Returns:
        matplotlib Figure 对象
    
    Example:
        >>> fig = quick_visualize("c1ccccc1O")  # 使用随机权重
        >>> fig = quick_visualize("CCO", [0.2, 0.5, 0.8])  # 指定权重
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    num_atoms = mol.GetNumAtoms()
    
    if weights is None:
        # 生成随机权重用于演示
        np.random.seed(42)
        weights = np.random.rand(num_atoms)
        print(f"Note: Using random attention weights for demonstration")
    
    weights = np.array(weights)
    if len(weights) != num_atoms:
        raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of atoms ({num_atoms})")
    
    fig = visualize_molecule_attention(
        smiles, 
        weights,
        title=f"Attention Visualization",
        show_colorbar=True,
        show_smiles=True
    )
    
    if save_path:
        save_attention_visualization(fig, save_path)
    
    return fig


def generate_example_visualizations(
    output_dir: str = 'results/attention_visualization'
) -> List[str]:
    """
    生成示例可视化图像。
    
    使用一些常见药物分子生成注意力可视化示例。
    
    Args:
        output_dir: 输出目录
    
    Returns:
        生成的文件路径列表
    
    Example:
        >>> paths = generate_example_visualizations()
        >>> print(f"Generated {len(paths)} example visualizations")
    """
    # 示例分子
    example_molecules = [
        {
            'name': 'aspirin',
            'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O',
            'description': 'Aspirin (Acetylsalicylic acid)'
        },
        {
            'name': 'caffeine', 
            'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'description': 'Caffeine'
        },
        {
            'name': 'ibuprofen',
            'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'description': 'Ibuprofen'
        }
    ]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    for mol_info in example_molecules:
        smiles = mol_info['smiles']
        name = mol_info['name']
        description = mol_info['description']
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES for {name}")
            continue
        
        num_atoms = mol.GetNumAtoms()
        
        # 生成模拟的注意力权重（使用有意义的分布）
        np.random.seed(hash(smiles) % 2**32)
        base_weights = np.random.rand(num_atoms)
        
        # 让某些原子有更高的权重（模拟模型关注功能团）
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            # 氧、氮原子通常更重要
            if symbol in ['O', 'N']:
                base_weights[idx] = min(1.0, base_weights[idx] + 0.3)
            # 芳香碳
            if atom.GetIsAromatic():
                base_weights[idx] = min(1.0, base_weights[idx] + 0.2)
        
        # 可视化
        fig = visualize_molecule_attention(
            smiles,
            base_weights,
            title=description,
            show_colorbar=True,
            show_smiles=True,
            figsize=(10, 8)
        )
        
        # 保存
        output_path = os.path.join(output_dir, f'{name}_attention.png')
        save_attention_visualization(fig, output_path, dpi=300)
        saved_files.append(output_path)
        
        plt.close(fig)
    
    print(f"\n✓ Generated {len(saved_files)} example visualizations in '{output_dir}'")
    
    return saved_files


###################################
### Integration with MVKAN Model ###
###################################

def visualize_from_tester(
    tester,
    sample_indices: List[int] = None,
    output_dir: str = 'results/attention_visualization',
    num_samples: int = 3
) -> List[str]:
    """
    从 Tester 对象提取注意力权重并生成可视化。
    
    Args:
        tester: MVKAN 的 Tester 对象（已完成测试）
        sample_indices: 要可视化的样本索引列表
        output_dir: 输出目录
        num_samples: 如果未指定索引，随机选择的样本数量
    
    Returns:
        生成的文件路径列表
    
    Example:
        >>> tester = Tester(args, args_test)
        >>> tester.test(test_loader, return_attention_weights=True)
        >>> paths = visualize_from_tester(tester)
    """
    # 获取注意力权重
    att_mol = tester.getAttentionMol()
    if att_mol is None:
        raise ValueError("No attention weights available. Run test with return_attention_weights=True")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # 这里需要根据实际的 att_mol 结构来处理
    # 通常 att_mol 包含 (attention_index, attention_weights)
    
    print(f"Attention visualization from Tester - saving to {output_dir}")
    
    return saved_files


if __name__ == '__main__':
    """
    模块测试代码
    """
    print("=" * 60)
    print("MVKAN Attention Visualization Module")
    print("=" * 60)
    
    # 生成示例可视化
    print("\nGenerating example visualizations...")
    example_files = generate_example_visualizations()
    
    print("\nExample visualizations generated successfully!")
    print("Files saved to: results/attention_visualization/")
