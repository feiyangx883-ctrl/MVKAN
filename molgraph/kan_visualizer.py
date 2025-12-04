"""
KAN (Kolmogorov-Arnold Network) Interpretability Visualization Module

This module provides visualization tools for KAN model interpretability,
including activation function visualization and Fourier coefficient heatmaps.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional


class KANVisualizer:
    """KAN 模型可解释性可视化工具"""
    
    def __init__(self, model, device='cuda:0', save_dir='./visualizations'):
        """
        初始化可视化器
        
        Args:
            model: 训练好的 GNN_Combine 模型
            device: 计算设备
            save_dir: 保存目录
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_kan_activation_functions(self, layer_idx=0, num_samples=5, 
                                          input_range=(-3, 3), save_name='kan_activations.png'):
        """
        可视化 KAN 层学习到的激活函数
        
        Args:
            layer_idx: KAN 层索引
            num_samples: 显示的输出维度数量
            input_range: 输入值范围 (min, max)
            save_name: 保存的文件名
        
        Returns:
            fig: matplotlib Figure 对象
        """
        # 获取所有 KAN 层
        kan_layers = self.model.get_kan_layers()
        
        if len(kan_layers) == 0:
            print("⚠️  No KAN layers found in the model")
            return None
        
        if layer_idx >= len(kan_layers):
            print(f"⚠️  Layer index {layer_idx} out of range (only {len(kan_layers)} KAN layers)")
            layer_idx = 0
        
        kan_layer = kan_layers[layer_idx]
        
        # 生成输入范围
        x_range = torch.linspace(input_range[0], input_range[1], 200).to(self.device)
        
        # 获取激活函数值
        with torch.no_grad():
            activation_values = kan_layer.get_activation_function_values(x_range)
        
        # 确保 activation_values 是 2D 的
        if activation_values.dim() == 1:
            activation_values = activation_values.unsqueeze(1)
        
        # 限制显示的输出维度数量
        output_dim = min(num_samples, activation_values.shape[1])
        
        # 创建 2x3 子图布局
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        x_np = x_range.cpu().numpy()
        activation_np = activation_values.cpu().numpy()
        
        for idx in range(6):
            ax = axes[idx]
            if idx < output_dim:
                # 绘制激活函数曲线
                ax.plot(x_np, activation_np[:, idx], color='#2E86AB', linewidth=2)
                ax.set_title(f'Output Neuron {idx}')
                ax.set_xlabel('Input')
                ax.set_ylabel('Activation')
                # 添加网格线
                ax.grid(True, alpha=0.3)
                # 添加零线参考
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
            else:
                # 隐藏多余的子图
                ax.axis('off')
        
        # 设置总标题
        fig.suptitle(f'KAN Layer {layer_idx}: Learned Activation Functions', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Saved KAN activation visualization to {save_path}")
        
        return fig
    
    def visualize_fourier_coefficients(self, max_layers=4, save_name='fourier_heatmap.png'):
        """
        可视化 Fourier-KAN 的频率系数分布
        
        Args:
            max_layers: 最多显示的层数
            save_name: 保存的文件名
        
        Returns:
            fig: matplotlib Figure 对象
        """
        # 获取所有 KAN 层
        kan_layers = self.model.get_kan_layers()
        
        if len(kan_layers) == 0:
            print("⚠️  No KAN layers found in the model")
            return None
        
        # 限制显示的层数
        num_layers = min(max_layers, len(kan_layers))
        
        # 创建 1 行 N 列布局
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
        
        # 如果只有一层，确保 axes 是列表
        if num_layers == 1:
            axes = [axes]
        
        for idx, kan_layer in enumerate(kan_layers[:num_layers]):
            ax = axes[idx]
            
            # 获取权重: shape [output_dim, input_dim, grid_size, 2]
            weights = kan_layer.weight.data
            
            # 计算 Fourier 幅度: sqrt(cos² + sin²)
            cos_component = weights[:, :, :, 0]
            sin_component = weights[:, :, :, 1]
            magnitude = torch.sqrt(cos_component ** 2 + sin_component ** 2)
            
            # 对输入维度求平均: shape [output_dim, grid_size]
            avg_magnitude = magnitude.mean(dim=1)
            
            # 转换为 numpy
            avg_magnitude_np = avg_magnitude.cpu().numpy()
            
            # 绘制热图
            im = ax.imshow(avg_magnitude_np, cmap='viridis', aspect='auto')
            ax.set_xlabel('Frequency Index')
            ax.set_ylabel('Output Dimension')
            ax.set_title(f'Layer {idx}')
            
            # 添加颜色条
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Magnitude')
        
        # 设置总标题
        fig.suptitle('Fourier Coefficient Magnitudes Across KAN Layers', fontsize=14)
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, save_name)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ Saved Fourier coefficient visualization to {save_path}")
        
        return fig
    
    def generate_all_visualizations(self, test_loader=None):
        """
        生成所有可解释性可视化图表
        
        Args:
            test_loader: 可选的测试数据加载器
        """
        print("\n🎨 Generating KAN interpretability visualizations...")
        
        # 1. KAN 激活函数
        self.visualize_kan_activation_functions(layer_idx=0)
        
        # 2. Fourier 系数热图
        self.visualize_fourier_coefficients()
        
        print(f"✅ All visualizations saved to {self.save_dir}\n")
