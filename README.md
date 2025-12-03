# MVKAN 模型介绍


## 一、模型概览
- 名称：MVKAN（Multi-View Kolmogorov–Arnold Networks for molecular property prediction）
- 目标：针对分子属性预测任务，使用多视图表示与 KAN（Kolmogorov–Arnold Network）思想结合图网络，提升分子性质预测性能。
- 主要思想：
  - 使用基于傅里叶/Fourier 特征扩展的 KAN 变体。
  - 支持分子碎片/子结构视图（fragmentation.py、substructurevocab.py），并对多视图结果做融合/一致性测试（testingconsensus.py）。
  - 提供训练、验证和测试的完整流程（training.py、testing.py、main.py）。

## 二、关键文件与作用
- main.py
  - 项目入口，负责解析配置/参数并协调训练/测试流程。

## 三、模型输入与输出（推断自代码结构）
- 输入
  - 分子表示（SMILES 或 rdkit Mol），经 molgraph 工具转换为图结构（节点特征、边特征、子结构视图）。
  - 可选的多视图输入：原子/键级图、碎片/子结构图、频域扩展特征等。
- 输出
  - 分子级回归或分类预测（具体任务可在 main.py / dataset 中指定）。
  - 解释性输出（interpret.py 提供子结构/原子贡献或重要性排序）。

## 四、训练与评估流程
- 配置/超参：查看 molgraph/hyperparameter.py 获取默认配置（学习率、batch size、训练轮次、模型层数等）。
- 训练：运行 main.py（或调用 molgraph/training.py 中的训练入口），训练流程包含：
  - 数据加载与预处理（dataset.py）
  - 多视图构建（fragmentation.py、substructurevocab.py）
  - 模型构建（graphmodel.py 调用 fourier_kan.py / gineconv.py 等）
  - 损失计算与优化（training.py）
  - checkpoint 保存与日志记录
- 测试/评估：
  - 单模型测试：molgraph/testing.py
  - 多视图/共识测试：molgraph/testingconsensus.py
  - 解释性评估：molgraph/interpret.py
- 实验管理：molgraph/experiment.py 提供实验跑批与结果组织相关的逻辑。

## 五、使用示例（快速起步）
- 训练
  - 常见命令（仓库根目录）：
    - python main.py

## 六、可视化与解释
- 运行可视化工具查看分子图或子结构贡献：
  - molgraph/visualize.py, molgraph/molgraphdisplay.py：用于生成分子图片、子结构高亮与预测解释视图。
- 解释流程通常是先用 trained checkpoint 做预测，再调用 interpret.py 计算贡献度/重要性评分并可视化。

## 七、复现实验建议
- 环境与依赖
  - 仓库为 Python 项目，主要依赖可能包括 rdkit、torch（或 torch_geometric）、numpy、pandas 等。请参照仓库顶部或 README 补充依赖。
- 数据准备
  - 确保将原始数据按 dataset.py 要求放置并运行预处理（如果有单独的数据脚本）。
- 固定随机种子、记录超参（hyperparameter.py）并保存 checkpoints。
- 若使用多视图/子结构词表，请先运行相应的词表构建流程（substructurevocab.py / fragmentation.py）。

## 八、模型拓展与改进点（建议）
- 尝试不同的图消息传递层（替换或扩展 gineconv.py）。
- 对 Fourier 特征的参数和频率范围做系统搜索（fourier_kan.py）。
- 增强子结构表示：尝试更丰富的子结构编码或预训练子结构向量。
- 融合更多视图（例如 3D 构象信息）以提升对立体化学敏感的预测任务。

## 九、参考与进一步阅读
- 仓库文件：molgraph/ 下的文件是阅读与理解模型实现的关键入口（graphmodel.py、fourier_kan.py、gineconv.py、training.py）。
- 项目描述（仓库简介）：基于 Kolmogorov-Arnold Networks(KAN) 的多视图分子性质预测。
