# 🚀 通过应变曲线预测受力

## 📖 项目概述
基于深度学习的四点应变时序数据到施力点位置和力大小的预测系统。该系统通过分析四个应变传感器（顶部、底部、左侧、右侧）的时序变化，准确预测施力点的二维坐标和施加的力大小。

### 🎯 主要功能
- **多任务学习**：同时预测位置坐标（x, y）和力大小
- **时序特征提取**：处理连续时间步的应变数据
- **物理约束**：在损失函数中加入物理先验知识
- **完整流程**：从数据预处理到模型训练、评估、预测的完整流水线

---

## 🛠️ 环境配置
### 系统要求
- Python 3.8+
- CUDA 11.0+（如果使用GPU）
- 至少8GB RAM
- 推荐：NVIDIA GPU（用于加速训练）

### 安装步骤
#### 方法一：使用Conda（推荐）
```bash
# 1. 创建虚拟环境
conda create -n strain_force python=3.9 -y
conda activate strain_force

# 2. 安装PyTorch（根据您的CUDA版本选择）
# CUDA 11.8版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 或CPU版本
pip install torch torchvision torchaudio

# 3. 安装其他依赖
pip install pandas numpy scikit-learn matplotlib seaborn jupyter notebook tqdm tensorboard scipy plotly pyyaml
```

#### 方法二：使用venv
```bash
# 1. 创建虚拟环境
python -m venv strain_force_env

# 2. 激活环境
# Windows
strain_force_env\Scripts\activate
# Linux/Mac
source strain_force_env/bin/activate

# 3. 安装依赖（同上）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn matplotlib seaborn jupyter notebook tqdm tensorboard scipy plotly pyyaml
```

### 验证安装

运行验证：
```bash
python env_check.py
```

---

## 📁 项目结构
```
strain_force_system/
├── data/ # 数据管理
│   ├── raw/ # 原始数据
│   ├── processed/ # 处理后的数据
│   └── splits/ # 数据集划分
├── models/ # 模型存储
├── src/ # 源代码
│   ├── data_processor.py # 数据处理
│   ├── models.py # 模型定义
│   ├── trainer.py # 训练器
│   ├── predictor.py # 预测器
│   └── utils.py # 工具函数
├── configs/ # 配置文件
├── scripts/ # 运行脚本
├── results/ # 实验结果
├── requirements.txt # 依赖包列表
└── run.py # 一键运行脚本
```

---

## 📊 数据准备
### 数据格式要求
您的Abaqus导出数据需要整理成CSV格式：
```
event_id	time_step	strain_top	strain_bottom	strain_left	strain_right	force_x	force_y	force_magnitude
1	0	32.21215116	0.267530047	0.656140663	0.847740312	54.8910648	98.91786499	213.7190602
1	0.01	32.39504436	0.315830057	0.652612133	0.879107464	54.8910648	98.91786499	213.7190602
1	0.02	32.56166721	0.30310202	0.709935575	0.855289069	54.8910648	98.91786499	213.7190602
1	0.03	32.80692352	0.287256728	0.711183128	0.844890223	54.8910648	98.91786499	213.7190602
1	0.04	32.96415004	0.268301821	0.693339974	0.872915539	54.8910648	98.91786499	213.7190602
1	0.05	33.18016378	0.275622413	0.682984195	0.846563216	54.8910648	98.91786499	213.7190602
1	0.06	33.38967293	0.320113329	0.640831557	0.876154392	54.8910648	98.91786499	213.7190602
1	0.07	33.5153485	0.30614366	0.679474056	0.836395853	54.8910648	98.91786499	213.7190602
1	0.08	33.79172483	0.3169196	0.707447842	0.879072301	54.8910648	98.91786499	213.7190602
1	0.09	33.93429659	0.311154353	0.652259573	0.882525254	54.8910648	98.91786499	213.7190602
...
```

### 数据说明
- **输入特征**：4个应变点的时序变化
  - `strain_top`：顶部应变
  - `strain_bottom`：底部应变
  - `strain_left`：左侧应变
  - `strain_right`：右侧应变
- **输出标签**：
  - `force_x`：施力点X坐标
  - `force_y`：施力点Y坐标
  - `force_magnitude`：力的大小

### 生成示例数据（可选）
如果没有真实数据，可以使用内置函数生成合成数据：
```bash
# 生成10000个样本的合成数据
python -c "
from src.data_processor import StrainDataProcessor
StrainDataProcessor.generate_synthetic_data(n_samples=10000, save_path='data/raw/synthetic_data.csv')
"
```

---

## 🚀 快速开始
### 方式一：一键运行（推荐）
```bash
# 生成合成数据并完整训练
python run_experiment.py --mode full

# 仅使用已有数据训练
python run_experiment.py --mode train

# 仅进行预测
python run_experiment.py --mode predict

# 自定义参数训练
python run_experiment.py --mode train --epochs 100 --batch_size 32 --seq_length 20
```

### 方式二：交互式快速开始
```bash
python scripts/quick_start.py
```
按照提示选择：
1. 生成示例数据并训练
2. 加载已有数据训练
3. 使用预训练模型预测

### 方式三：分步执行
#### 1. 数据预处理
```python
import yaml
from src.data_processor import StrainDataProcessor

with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

processor = StrainDataProcessor(config)
datasets = processor.process_pipeline()
print('数据处理完成')
```

#### 2. 训练模型
```python
import yaml
import torch
from src.models import PositionForceNet
from src.trainer import StrainForceTrainer
from torch.utils.data import DataLoader, TensorDataset

# 加载配置
with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载数据
import numpy as np
train_data = np.load('data/splits/train.npz')
val_data = np.load('data/splits/val.npz')

# 创建DataLoader
train_dataset = TensorDataset(torch.tensor(train_data['X']), torch.tensor(train_data['y']))
val_dataset = TensorDataset(torch.tensor(val_data['X']), torch.tensor(val_data['y']))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 创建模型
model = PositionForceNet(config)

# 训练
trainer = StrainForceTrainer(config)
trainer.setup_model(model)
results = trainer.train(train_loader, val_loader)
```

#### 3. 进行预测
```python
from src.predictor import StrainForcePredictor
import numpy as np

# 加载模型
predictor = StrainForcePredictor('models/best_model.pth')

# 创建示例数据
strain_data = np.random.randn(1, 20, 4) # [batch, seq_length, 4]

# 预测
result = predictor.predict(strain_data)
print(f'预测结果: {result}')
```

---

## ⚙️ 详细使用方法
### 1. 数据处理
#### 加载和处理数据
```python
from src.data_processor import StrainDataProcessor
import yaml

# 加载配置
with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建处理器
processor = StrainDataProcessor(config)

# 处理数据
datasets = processor.process_pipeline('data/raw/abaqus_strain_data.csv')
```

#### 添加衍生特征
系统会自动添加以下衍生特征以提高预测精度：
- 应变梯度（垂直和水平）
- 应变比值
- 统计特征（均值、标准差、最大值、最小值）
- 复合特征

### 2. 模型训练
#### 基础训练
```python
from src.models import PositionForceNet
from src.trainer import StrainForceTrainer
import torch
from torch.utils.data import DataLoader, TensorDataset

# 创建模型
model = PositionForceNet(config)

# 创建训练器
trainer = StrainForceTrainer(config)
trainer.setup_model(model)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 开始训练
results = trainer.train(train_loader, val_loader)
```

#### 使用不同模型架构
修改 `configs/base_config.yaml`：
```yaml
model:
  model_type: "CNN_LSTM" # 可选: "CNN_LSTM", "Transformer", "SimpleMLP"
```

### 3. 模型预测
#### 单样本预测
```python
from src.predictor import StrainForcePredictor
import numpy as np

# 创建预测器
predictor = StrainForcePredictor('models/best_model.pth')

# 创建输入数据（20个时间步，4个应变点）
strain_data = np.random.randn(1, 20, 4)

# 预测
result = predictor.predict(strain_data)
print(f"位置: ({result['position_x']:.2f}, {result['position_y']:.2f})")
print(f"力大小: {result['force_magnitude']:.2f}")
```

#### 批量预测
```python
# 批量数据
batch_data = np.random.randn(10, 20, 4) # 10个样本

# 批量预测
results = predictor.predict(batch_data)
for i, res in enumerate(results['batch_results']):
    print(f"样本{i}: 位置({res['position_x']:.2f}, {res['position_y']:.2f}), 力大小: {res['force_magnitude']:.2f}")
```

### 4. 模型评估
```python
# 在测试集上评估
test_data = np.load('data/splits/test.npz')
X_test, y_test = test_data['X'], test_data['y']

eval_results = predictor.evaluate_on_test_set(X_test, y_test)

# 查看评估指标
print(f"位置R²: {eval_results['metrics']['position_r2_mean']:.4f}")
print(f"力大小R²: {eval_results['metrics']['force_r2']:.4f}")
```

---

## ⚙️ 配置说明
### 基础配置 (configs/base_config.yaml)
```yaml
# 数据配置
data:
  input_csv: "data/raw/abaqus_strain_data.csv"
  feature_columns: ["strain_top", "strain_bottom", "strain_left", "strain_right"]
  label_columns: ["force_x", "force_y", "force_magnitude"]
  sequence_length: 20 # 每个序列的时间步数

# 模型配置
model:
  model_type: "CNN_LSTM" # 模型类型
  cnn_channels: [32, 64, 128] # CNN通道数
  lstm_hidden_size: 128 # LSTM隐藏层大小

# 训练配置
training:
  epochs: 200 # 训练轮数
  batch_size: 32 # 批大小
  learning_rate: 0.001 # 学习率
  loss_weights:
    position: 0.6 # 位置损失权重
    force: 0.4 # 力大小损失权重
```

### 小数据集配置
如果数据量较少（<1000样本），建议使用 `configs/simple_config.yaml`：
```yaml
model:
  model_type: "SimpleMLP" # 使用简单模型防止过拟合
  mlp_hidden_layers: [128, 64, 32] # 较小的网络

training:
  epochs: 50 # 较少轮次
  batch_size: 16 # 较小批次
  learning_rate: 0.0005 # 较小学习率
```

### 高级配置
对于大数据集（>10000样本），使用 `configs/advanced_config.yaml`：
```yaml
model:
  model_type: "CNN_LSTM"
  cnn_channels: [64, 128, 256] # 更多通道
  lstm_hidden_size: 256 # 更大的LSTM

training:
  epochs: 300 # 更多轮次
  batch_size: 64 # 更大批次
  scheduler: "CosineAnnealing" # 更好的学习率调度器
```

---

## 📈 结果解读
### 输出文件
训练完成后，系统会生成以下文件：
1. **models/best_model_*.pth** - 最佳模型权重
2. **models/scaler_features.pkl** - 特征归一化器
3. **models/scaler_labels.pkl** - 标签归一化器
4. **results/training_history_*.json** - 训练历史
5. **results/test_results_*.json** - 测试结果
6. **results/figures/** - 可视化图表

### 评估指标
系统计算以下指标：

| 指标 | 说明 | 理想值 |
|------|------|--------|
| 位置R² | 位置预测的决定系数 | 接近1.0 |
| 位置MAE | 位置平均绝对误差 | 接近0.0 |
| 位置RMSE | 位置均方根误差 | 接近0.0 |
| 力大小R² | 力大小预测的决定系数 | 接近1.0 |
| 力大小MAE | 力大小平均绝对误差 | 接近0.0 |
| 力大小RMSE | 力大小均方根误差 | 接近0.0 |

### 可视化图表
系统自动生成以下图表：
1. **训练曲线**：训练和验证损失变化
2. **位置预测对比**：真实位置 vs 预测位置
3. **力大小预测对比**：真实力大小 vs 预测力大小
4. **误差分布**：位置误差和力大小误差的分布

查看图表：
```bash
# 在浏览器中打开TensorBoard
tensorboard --logdir logs/

# 或查看生成的图片
ls results/figures/
```

---

## 🔧 故障排除
### 常见问题
#### 1. 内存不足
```bash
# 减小批大小和序列长度
python run_experiment.py --batch_size 16 --seq_length 10

# 使用简单模型
python run_experiment.py --config configs/simple_config.yaml
```

#### 2. 训练不收敛
```bash
# 降低学习率
python run_experiment.py --learning_rate 0.0001

# 增加训练轮次
python run_experiment.py --epochs 300
```

#### 3. 过拟合
```bash
# 使用更简单的模型
python run_experiment.py --config configs/simple_config.yaml
```
（额外：在配置文件中调整 dropout 和 weight_decay 增加正则化）

#### 4. 数据格式错误
- 检查CSV文件格式
- 确保列名正确
- 检查是否有缺失值

---

## 📄 许可证
本项目采用 MIT 许可证。详见 LICENSE 文件。

---

## 📞 联系方式
如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至 [qsq@mail.dlut.edu.cn]

---

## 🙏 致谢
感谢以下开源项目：
- PyTorch
- Scikit-learn
- NumPy & Pandas
- Matplotlib & Seaborn

---

**使用愉快！** 🎉