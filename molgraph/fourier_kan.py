from __future__ import annotations

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KANLinear"]


class KANLinear(nn.Module):
    """Enhanced Fourier-based Kolmogorov–Arnold Network linear layer with regularization.

    The layer expands each input dimension using sinusoidal bases with
    frequencies from 1 to ``grid_size`` and learns a linear combination of the
    resulting Fourier coefficients. Enhanced with adaptive frequencies,
    residual connections, and built-in regularization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: int = 4,
        *,
        add_bias: bool = True,
        dropout: float = 0.0,
        use_residual: bool = True,
        adaptive_freq: bool = True,
        freq_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.grid_size = int(grid_size)
        self.use_residual = use_residual and (input_dim == output_dim)
        self.adaptive_freq = adaptive_freq
        self.dropout = dropout

        # Fourier coefficients - shape: (output_dim, input_dim, grid_size, 2)
        # 2 for cos and sin terms
        weight = torch.empty(self.output_dim, self.input_dim, self.grid_size, 2)
        self.weight = nn.Parameter(weight)

        # Bias term
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter("bias", None)

        # Base frequencies - fixed or learnable
        if adaptive_freq:
            # Learnable frequency scaling factors
            freq_init = torch.arange(1, self.grid_size + 1, dtype=torch.float32) * freq_scale
            self.frequencies = nn.Parameter(freq_init.view(1, 1, self.grid_size))
        else:
            # Fixed frequencies
            frequencies = torch.arange(
                1, self.grid_size + 1, dtype=torch.float32
            ).view(1, 1, self.grid_size) * freq_scale
            self.register_buffer("frequencies", frequencies, persistent=False)

        # Residual connection (only if input_dim == output_dim)
        if self.use_residual:
            self.residual_weight = nn.Parameter(torch.ones(1))
            if input_dim != output_dim:
                self.residual_proj = nn.Linear(input_dim, output_dim, bias=False)
            else:
                self.residual_proj = None
        
        # Normalization for better training stability
        self.layer_norm = nn.LayerNorm(self.input_dim)
        
        # Dropout layer
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters with improved initialization."""
        # Xavier/Glorot initialization for Fourier coefficients
        std = math.sqrt(2.0 / (self.input_dim * self.grid_size + self.output_dim))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        if self.use_residual:
            nn.init.constant_(self.residual_weight, 0.1)  # Start with small residual
            if hasattr(self, 'residual_proj') and self.residual_proj is not None:
                nn.init.xavier_uniform_(self.residual_proj.weight)

    def get_fourier_basis(self, x: Tensor) -> Tensor:
        """Compute Fourier basis functions with enhanced numerical stability."""
        # x shape: (batch, input_dim)
        frequencies = self.frequencies.to(dtype=x.dtype, device=x.device)
        
        # Compute angles: (batch, input_dim, grid_size)
        angles = x.unsqueeze(-1) * frequencies
        
        # Compute cos and sin terms with improved numerical stability
        cos_terms = torch.cos(angles)
        sin_terms = torch.sin(angles)
        
        # Stack and reshape: (batch, input_dim * grid_size * 2)
        fourier_basis = torch.stack((cos_terms, sin_terms), dim=-1)
        fourier_basis = fourier_basis.reshape(x.size(0), -1)
        
        return fourier_basis

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)
        
        # Layer normalization for better training stability
        x_normed = self.layer_norm(x_flat)
        
        # Store input for residual connection
        residual = x_flat if self.use_residual else None
        
        # Compute Fourier basis
        fourier_basis = self.get_fourier_basis(x_normed)
        
        # Apply dropout to Fourier basis if training
        if self.dropout > 0 and self.training:
            fourier_basis = self.dropout_layer(fourier_basis)
        
        # Linear transformation: fourier_basis @ weight
        weight_flat = self.weight.view(self.output_dim, -1).t()
        y = fourier_basis @ weight_flat  # (batch, output_dim)
        
        # Add bias
        if self.bias is not None:
            y = y + self.bias
        
        # Add residual connection
        if self.use_residual and residual is not None:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            y = y + self.residual_weight * residual
        
        # Reshape back to original shape
        y = y.view(*original_shape, self.output_dim)
        return y

    def compute_regularization_loss(self) -> Tensor:
        """Compute regularization losses for KAN layer."""
        device = self.weight.device
        
        # L2 regularization on Fourier coefficients
        l2_reg = torch.sum(self.weight ** 2)
        
        # Smoothness regularization: penalize high frequency components
        freq_penalty = 0.0
        if self.adaptive_freq:
            # Penalize very high frequencies to encourage smoother functions
            freq_weights = F.softmax(self.frequencies.squeeze(), dim=-1)
            freq_indices = torch.arange(1, self.grid_size + 1, device=device, dtype=torch.float32)
            freq_penalty = torch.sum(freq_weights * freq_indices ** 2) * 0.01
        
        # Sparsity regularization: encourage some coefficients to be small
        sparsity_reg = torch.sum(torch.abs(self.weight)) * 0.001
        
        return l2_reg + freq_penalty + sparsity_reg

    def get_activation_function_values(self, x_range: Tensor) -> Tensor:
        """Get the learned activation function values for visualization."""
        with torch.no_grad():
            # x_range should be 1D tensor
            x_expanded = x_range.unsqueeze(0).unsqueeze(-1)  # (1, len, 1)
            if x_expanded.size(-1) != self.input_dim:
                # For visualization, we typically look at 1D functions
                x_expanded = x_expanded.expand(-1, -1, self.input_dim)
            
            fourier_basis = self.get_fourier_basis(x_expanded.squeeze(0))
            weight_flat = self.weight.view(self.output_dim, -1).t()
            y = fourier_basis @ weight_flat
            
            if self.bias is not None:
                y = y + self.bias
                
            return y.squeeze()