# MVKAN: Multi-View Kolmogorov-Arnold Networks for Molecular Property Prediction

## 1. Problem Formulation

### 1.1 Background

Molecular property prediction is a fundamental task in drug discovery and materials science. Given a molecule represented as a SMILES string or molecular graph, the goal is to predict its chemical or biological properties (e.g., toxicity, solubility, binding affinity).

### 1.2 Challenges

Traditional approaches face several challenges:
- **Single-view limitation**: Most methods only consider atom-level molecular graphs, missing important structural patterns at higher abstraction levels
- **Fixed activation functions**: Standard neural networks use predefined activation functions (ReLU, Sigmoid), limiting their ability to learn optimal nonlinear transformations for specific tasks
- **Multi-scale information fusion**: Effectively combining information from different molecular representations remains an open problem

### 1.3 Problem Definition

Given a molecule $M$ with its multi-view representations $\{G_1, G_2, ..., G_K\}$ (e.g., atom-level graph, fragment graph, functional group graph), we aim to learn a function:

$$f: \{G_1, G_2, ..., G_K\} \rightarrow y$$

where $y$ is the target property (continuous for regression, discrete for classification).

---

## 2. MVKAN Architecture

### 2.1 Overview and Motivation

**MVKAN (Multi-View Kolmogorov-Arnold Networks)** is a novel architecture that combines multi-view molecular representations with Kolmogorov-Arnold Networks (KAN) for enhanced molecular property prediction.

#### Core Innovations

1. **Learnable Activation Functions**: Unlike traditional MLPs with fixed activations, MVKAN uses Fourier-KAN layers that learn optimal nonlinear transformations directly from data
2. **Multi-View Integration**: Supports multiple molecular views (atom, fragment, functional group, pharmacophore) with learnable view alignment
3. **Parameter Efficiency**: Achieves comparable or better performance with fewer parameters through the Kolmogorov-Arnold representation theorem

#### Architecture Overview

```
Input: SMILES/Molecule
         ↓
┌────────────────────────────────────┐
│     Multi-View Graph Construction   │
│  ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │ Atom    │ │Fragment │ │ Func. │ │
│  │ Graph   │ │ Graph   │ │ Group │ │
│  └────┬────┘ └────┬────┘ └───┬───┘ │
└───────┼───────────┼──────────┼─────┘
        ↓           ↓          ↓
┌────────────────────────────────────┐
│      GNN Encoders (per view)        │
│   GATv2/GIN + GRU Message Passing   │
└────────────────────────────────────┘
        ↓           ↓          ↓
┌────────────────────────────────────┐
│      View Alignment Module          │
│   (Attention-based pooling)         │
└────────────────────────────────────┘
        ↓
┌────────────────────────────────────┐
│    Fourier-KAN Aggregation          │
│   (Learnable nonlinear fusion)      │
└────────────────────────────────────┘
        ↓
┌────────────────────────────────────┐
│    Classification/Regression Head   │
└────────────────────────────────────┘
        ↓
     Output: y
```

### 2.2 View Alignment Module

The View Alignment Module ensures that features from different molecular views are comparable and can be effectively combined.

#### Purpose

Different molecular views capture information at different granularities:
- **Atom-level graph**: Captures detailed bonding patterns and local chemical environments
- **Fragment graph**: Captures functional substructures and their connectivity
- **Reduced graphs**: Captures higher-level topological patterns

These representations have different dimensionalities and semantic meanings. The View Alignment Module projects them into a common latent space.

#### Mechanism

For each view $k$, we apply:

1. **Node Feature Transformation**:
   $$h_v^{(k)} = \text{LeakyReLU}(\text{BN}(W_{\text{node}}^{(k)} \cdot x_v^{(k)}))$$

2. **Edge Feature Transformation**:
   $$e_{uv}^{(k)} = \text{LeakyReLU}(\text{BN}(W_{\text{edge}}^{(k)} \cdot x_{uv}^{(k)}))$$

3. **Attention-based Graph Pooling**:
   Using GATv2Conv for attention-weighted message passing:
   $$\alpha_{ij} = \text{softmax}\left(\text{LeakyReLU}(\mathbf{a}^T [W h_i \| W h_j])\right)$$
   
   $$h_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j$$

4. **GRU-based Feature Refinement**:
   $$h_i^{(t+1)} = \text{GRU}(h_i', h_i^{(t)})$$

This ensures each view's molecule-level embedding lies in a compatible space for subsequent fusion.

### 2.3 Fourier-KAN Aggregation (Detailed)

The Fourier-KAN layer is the core innovation of MVKAN, replacing traditional linear transformations with learnable nonlinear functions based on Fourier series.

#### 2.3.1 Mathematical Foundation: Kolmogorov-Arnold Representation Theorem

The Kolmogorov-Arnold representation theorem states that any continuous multivariate function $f: [0,1]^n \rightarrow \mathbb{R}$ can be represented as:

$$f(x_1, ..., x_n) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

where $\phi_{q,p}: [0,1] \rightarrow \mathbb{R}$ are continuous univariate functions.

**Key Insight**: Instead of learning weight matrices (as in MLPs), we can learn the univariate activation functions $\phi$ themselves, potentially achieving better expressiveness with fewer parameters.

#### 2.3.2 Fourier Basis Representation

We represent each learnable function $\phi(x)$ using Fourier series:

$$\phi(x) = \sum_{k=1}^{K} \left[ a_k \cos(2\pi f_k x) + b_k \sin(2\pi f_k x) \right]$$

where:
- $K$ is the grid size (number of frequency components)
- $f_k$ are the frequencies (learnable or fixed)
- $a_k, b_k$ are learnable Fourier coefficients

#### 2.3.3 Why Fourier Basis Functions?

The choice of Fourier basis offers several advantages:

1. **Universal Approximation**: Fourier series can approximate any continuous periodic function arbitrarily well (Weierstrass approximation theorem)

2. **Smooth Gradients**: Sinusoidal functions have continuous derivatives of all orders, ensuring smooth gradient flow during training

3. **Frequency Decomposition**: Different frequency components capture patterns at different scales:
   - Low frequencies: Global trends and slow variations
   - High frequencies: Local patterns and fine details

4. **Orthogonality**: Fourier basis functions are orthogonal, reducing redundancy in the learned representation

5. **Computational Efficiency**: Fast Fourier Transform (FFT) enables efficient computation

#### 2.3.4 Implementation Details

The `KANLinear` layer computes:

```python
# Input normalization for stability
x_normed = LayerNorm(x)

# Compute Fourier basis: (batch, input_dim, grid_size, 2)
frequencies = [f_1, f_2, ..., f_K]  # learnable or fixed
angles = x * frequencies  # (batch, input_dim, grid_size)
cos_terms = cos(angles)
sin_terms = sin(angles)
fourier_basis = stack([cos_terms, sin_terms])

# Linear combination of Fourier features
# weight shape: (output_dim, input_dim, grid_size, 2)
output = fourier_basis @ weight

# Residual connection (when input_dim == output_dim)
if use_residual:
    output = output + residual_weight * x
```

#### 2.3.5 Adaptive Frequency Learning

MVKAN supports learnable frequencies that adapt to the data:

$$f_k^{(t+1)} = f_k^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial f_k}$$

This allows the model to:
- **Increase frequency resolution** where the data requires fine-grained patterns
- **Focus on dominant frequencies** for each task
- **Adapt to dataset characteristics** automatically

### 2.4 Theoretical Advantages

#### 2.4.1 Expressiveness

**Theorem**: A KAN layer with grid size $K$ can represent any function that an MLP with $O(K)$ hidden units can represent, but with better sample efficiency for smooth functions.

**Intuition**: KAN learns the "shape" of activation functions, while MLP learns only weights. For functions with regular structure (common in molecular properties), learning the activation shape is more efficient.

#### 2.4.2 Interpretability

Unlike black-box neural networks, Fourier-KAN provides:

1. **Frequency Analysis**: Examining learned frequencies reveals which scales are important for the task
2. **Activation Visualization**: Plotting $\phi(x)$ shows the learned nonlinearity
3. **Coefficient Sparsity**: Important features have larger Fourier coefficients

#### 2.4.3 Multi-View Synergy

When combining multiple views with Fourier-KAN:

$$h_{\text{fused}} = \text{KAN}\left( \text{concat}[h_{\text{atom}}, h_{\text{frag}}, h_{\text{func}}] \right)$$

The KAN layer learns:
- **Cross-view interactions**: Nonlinear relationships between different views
- **View-specific transformations**: Optimal processing for each view type
- **Adaptive weighting**: Implicitly learns view importance through frequency patterns

### 2.5 Parameter Efficiency and Regularization

#### 2.5.1 Parameter Count Comparison

For a transformation from dimension $d_{in}$ to $d_{out}$:

| Layer Type | Parameters |
|------------|------------|
| Linear (MLP) | $d_{in} \times d_{out} + d_{out}$ |
| KANLinear (grid_size=K) | $d_{out} \times d_{in} \times K \times 2 + d_{out}$ |

While KAN has more parameters per layer, it often requires fewer layers to achieve the same expressiveness, leading to comparable total parameters.

#### 2.5.2 Regularization Strategies

MVKAN employs multiple regularization techniques:

1. **L2 Regularization on Fourier Coefficients**:
   $$\mathcal{L}_{L2} = \lambda_1 \sum_{k} (a_k^2 + b_k^2)$$

2. **Frequency Penalty** (discourages overly high frequencies):
   $$\mathcal{L}_{freq} = \lambda_2 \sum_{k} k^2 \cdot |w_k|$$

3. **Sparsity Regularization** (encourages coefficient sparsity):
   $$\mathcal{L}_{sparse} = \lambda_3 \sum_{k} |a_k| + |b_k|$$

4. **Dropout on Fourier Basis**: Applied during training to prevent overfitting to specific frequency components

5. **Layer Normalization**: Applied before the Fourier transformation for training stability

#### 2.5.3 Residual Connections

When input and output dimensions match, MVKAN uses residual connections:

$$y = \text{KAN}(x) + \gamma \cdot x$$

where $\gamma$ is a learnable scalar initialized to a small value (0.1). This:
- Facilitates gradient flow in deep networks
- Provides a "fallback" to identity transformation
- Stabilizes early training

---

## 3. Training and Optimization

### 3.1 Loss Functions

- **Regression**: MSE Loss
  $$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

- **Binary Classification**: BCEWithLogitsLoss with class weighting
  $$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}[w \cdot y_i \log(\sigma(\hat{y}_i)) + (1-y_i)\log(1-\sigma(\hat{y}_i))]$$

- **Multi-task**: Task-wise BCE with NaN handling for missing labels

### 3.2 Optimization Strategy

- **Optimizer**: Adam (classification) or RMSprop (regression)
- **Learning Rate Schedule**: Cosine annealing with warm-up
- **Gradient Clipping**: Norm-based clipping to prevent exploding gradients
- **Early Stopping**: Based on validation metric with patience

### 3.3 Training Command

```bash
python main.py --file bace --schema AR --model GIN --use_kan_readout True
```

---

## 4. Experiments

### 4.1 Datasets

We evaluate MVKAN on standard molecular property prediction benchmarks from MoleculeNet, including BACE (blood-brain barrier penetration prediction) and BBBP (binary classification for brain permeability). These datasets provide challenging real-world scenarios for evaluating molecular property prediction models.

### 4.2 Results Highlights

MVKAN demonstrates consistent improvements over baseline methods:
- **Classification tasks (BACE, BBBP)**: Higher AUC-ROC through multi-view integration
- **Regression tasks**: Lower RMSE with Fourier-KAN's adaptive nonlinearities
- **Interpretability**: Learned activation functions reveal task-specific patterns

### 4.3 Key Findings

1. **Multi-view representations** consistently outperform single-view baselines
2. **Fourier-KAN** provides smoother training curves and better generalization
3. **Adaptive frequencies** automatically adjust to dataset characteristics
4. **Regularization** is crucial for preventing overfitting in small molecular datasets

---

## 5. Quick Start

### 5.1 Installation

```bash
# Clone the repository
git clone https://github.com/feiyangx883-ctrl/MVKAN.git
cd MVKAN

# Install dependencies
pip install torch torch_geometric rdkit numpy pandas scikit-learn
```

### 5.2 Training

```bash
# Basic training
python main.py

# With specific configuration
python main.py --file bbbp --schema AR --use_kan_readout True --kan_grid_size 4
```

### 5.3 Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--file` | Dataset name | `bace` |
| `--schema` | View schema (A/R/AR/AR_N) | `AR` |
| `--model` | GNN type (GIN/GAT) | `GIN` |
| `--use_kan_readout` | Enable Fourier-KAN | `True` |
| `--kan_grid_size` | Number of Fourier frequencies | `4` |

---

## 6. Code Structure

```
MVKAN/
├── main.py                 # Entry point
├── dataset/                # Data files
├── molgraph/
│   ├── fourier_kan.py     # Fourier-KAN layer implementation
│   ├── graphmodel.py      # MVKAN model architecture
│   ├── training.py        # Training loop
│   ├── dataset.py         # Data loading and preprocessing
│   ├── fragmentation.py   # Molecular fragmentation
│   └── interpret.py       # Model interpretation
├── util/                   # Utility functions
└── vocab/                  # Substructure vocabularies
```

---

## 7. Citation

If you find this work useful, please cite:

```bibtex
@article{mvkan2024,
  title={MVKAN: Multi-View Kolmogorov-Arnold Networks for Molecular Property Prediction},
  author={},
  journal={},
  year={2024}
}
```

---

## 8. References

1. Liu, Z., et al. "KAN: Kolmogorov-Arnold Networks." arXiv preprint arXiv:2404.19756 (2024).
2. Wu, Z., et al. "MoleculeNet: A Benchmark for Molecular Machine Learning." Chemical Science 9.2 (2018).
3. Kolmogorov, A. N. "On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition." Doklady Akademii Nauk (1957).
