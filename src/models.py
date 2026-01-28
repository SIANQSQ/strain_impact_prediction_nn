import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class PositionForceNet(nn.Module):
    """
    四点应变时序数据 -> 施力点坐标 + 力大小
    基于CNN+LSTM的神经网络
    """
    
    def __init__(self, config: dict):
        """
        初始化模型
        
        参数:
            config: 模型配置字典
        """
        super(PositionForceNet, self).__init__()
        
        self.config = config
        self.model_type = config.get('model_type', 'CNN_LSTM')
        
        # 获取输入维度
        seq_length = config.get('sequence_length', 20)
        feature_dim = config.get('feature_dim', 4)  # 4个应变点
        
        if self.model_type == 'CNN_LSTM':
            self._build_cnn_lstm_model(seq_length, feature_dim, config)
        elif self.model_type == 'Transformer':
            self._build_transformer_model(seq_length, feature_dim, config)
        elif self.model_type == 'SimpleMLP':
            self._build_mlp_model(seq_length, feature_dim, config)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 初始化权重
        self._init_weights()
        
        print(f"模型 '{self.model_type}' 初始化完成")
        print(f"参数量: {self._count_parameters():,}")
    
    def _build_cnn_lstm_model(self, seq_length: int, feature_dim: int, config: dict):
        """构建CNN+LSTM模型"""
        model_config = config.get('model', {})
        
        # CNN部分 - 提取空间特征
        cnn_channels = model_config.get('cnn_channels', [32, 64, 128])
        cnn_kernels = model_config.get('cnn_kernel_sizes', [3, 3, 3])
        cnn_dropout = model_config.get('cnn_dropout', 0.2)
        use_batch_norm = model_config.get('batch_norm', True)
        
        cnn_layers = []
        in_channels = feature_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, cnn_kernels)):
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                        padding=kernel_size//2))
            
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm1d(out_channels))
            
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(cnn_dropout))
            
            # 添加池化层
            if i < len(cnn_channels) - 1:
                cnn_layers.append(nn.MaxPool1d(kernel_size=2))
            
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # 计算经过CNN后的序列长度
        cnn_seq_length = seq_length
        for _ in range(len(cnn_channels) - 1):
            cnn_seq_length = cnn_seq_length // 2
        
        # LSTM部分 - 提取时序特征
        lstm_hidden_size = model_config.get('lstm_hidden_size', 128)
        lstm_num_layers = model_config.get('lstm_num_layers', 2)
        lstm_bidirectional = model_config.get('lstm_bidirectional', True)
        lstm_dropout = model_config.get('lstm_dropout', 0.3)
        
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        # 特征融合部分
        lstm_output_dim = lstm_hidden_size * 2 if lstm_bidirectional else lstm_hidden_size
        
        mlp_hidden = model_config.get('mlp_hidden_layers', [256, 128, 64])
        mlp_dropout = model_config.get('mlp_dropout', 0.3)
        
        fusion_layers = []
        in_features = lstm_output_dim
        
        for hidden_size in mlp_hidden:
            fusion_layers.append(nn.Linear(in_features, hidden_size))
            fusion_layers.append(nn.BatchNorm1d(hidden_size))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(mlp_dropout))
            in_features = hidden_size
        
        self.fusion = nn.Sequential(*fusion_layers)
        
        # 双输出头
        # 位置预测头
        pos_hidden = model_config.get('position_head_hidden', [64, 32])
        pos_layers = []
        in_features = mlp_hidden[-1]
        
        for hidden_size in pos_hidden:
            pos_layers.append(nn.Linear(in_features, hidden_size))
            pos_layers.append(nn.ReLU())
            in_features = hidden_size
        
        pos_layers.append(nn.Linear(in_features, 2))  # 输出x, y坐标
        self.position_head = nn.Sequential(*pos_layers)
        
        # 力大小预测头
        force_hidden = model_config.get('force_head_hidden', [64, 32])
        force_layers = []
        in_features = mlp_hidden[-1]
        
        for hidden_size in force_hidden:
            force_layers.append(nn.Linear(in_features, hidden_size))
            force_layers.append(nn.ReLU())
            in_features = hidden_size
        
        force_layers.append(nn.Linear(in_features, 1))  # 输出力大小
        self.force_head = nn.Sequential(*force_layers)
        
        # 保存维度信息
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.cnn_seq_length = cnn_seq_length
        
    def _build_transformer_model(self, seq_length: int, feature_dim: int, config: dict):
        """构建Transformer模型"""
        model_config = config.get('model', {})
        
        # 输入嵌入层
        d_model = model_config.get('transformer_d_model', 128)
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # 位置编码
        self.position_encoding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        
        # Transformer编码器
        nhead = model_config.get('transformer_nhead', 8)
        num_layers = model_config.get('transformer_num_layers', 3)
        dropout = model_config.get('dropout', 0.1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 特征融合
        mlp_hidden = model_config.get('mlp_hidden_layers', [256, 128, 64])
        fusion_layers = []
        in_features = d_model
        
        for hidden_size in mlp_hidden:
            fusion_layers.append(nn.Linear(in_features, hidden_size))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(0.3))
            in_features = hidden_size
        
        self.fusion = nn.Sequential(*fusion_layers)
        
        # 双输出头
        # 位置预测头
        pos_hidden = model_config.get('position_head_hidden', [64, 32])
        pos_layers = []
        in_features = mlp_hidden[-1]
        
        for hidden_size in pos_hidden:
            pos_layers.append(nn.Linear(in_features, hidden_size))
            pos_layers.append(nn.ReLU())
            in_features = hidden_size
        
        pos_layers.append(nn.Linear(in_features, 2))
        self.position_head = nn.Sequential(*pos_layers)
        
        # 力大小预测头
        force_hidden = model_config.get('force_head_hidden', [64, 32])
        force_layers = []
        in_features = mlp_hidden[-1]
        
        for hidden_size in force_hidden:
            force_layers.append(nn.Linear(in_features, hidden_size))
            force_layers.append(nn.ReLU())
            in_features = hidden_size
        
        force_layers.append(nn.Linear(in_features, 1))
        self.force_head = nn.Sequential(*force_layers)
        
    def _build_mlp_model(self, seq_length: int, feature_dim: int, config: dict):
        """构建简单MLP模型（适用于小数据集）"""
        model_config = config.get('model', {})
        
        # 扁平化输入
        input_dim = seq_length * feature_dim
        
        # MLP层
        mlp_hidden = model_config.get('mlp_hidden_layers', [256, 128, 64])
        mlp_dropout = model_config.get('mlp_dropout', 0.3)
        
        mlp_layers = []
        in_features = input_dim
        
        for hidden_size in mlp_hidden:
            mlp_layers.append(nn.Linear(in_features, hidden_size))
            mlp_layers.append(nn.BatchNorm1d(hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(mlp_dropout))
            in_features = hidden_size
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 双输出头
        # 位置预测头
        pos_hidden = model_config.get('position_head_hidden', [64, 32])
        pos_layers = []
        in_features = mlp_hidden[-1]
        
        for hidden_size in pos_hidden:
            pos_layers.append(nn.Linear(in_features, hidden_size))
            pos_layers.append(nn.ReLU())
            in_features = hidden_size
        
        pos_layers.append(nn.Linear(in_features, 2))
        self.position_head = nn.Sequential(*pos_layers)
        
        # 力大小预测头
        force_hidden = model_config.get('force_head_hidden', [64, 32])
        force_layers = []
        in_features = mlp_hidden[-1]
        
        for hidden_size in force_hidden:
            force_layers.append(nn.Linear(in_features, hidden_size))
            force_layers.append(nn.ReLU())
            in_features = hidden_size
        
        force_layers.append(nn.Linear(in_features, 1))
        self.force_head = nn.Sequential(*force_layers)
        
    def _init_weights(self):
        """初始化模型权重"""
        init_method = self.config.get('model', {}).get('weight_init', 'xavier_normal')
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_method == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif init_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif init_method == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        print(f"权重初始化方法: {init_method}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_length, feature_dim]
            
        返回:
            position: 位置预测 [batch_size, 2]
            force: 力大小预测 [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        if self.model_type == 'CNN_LSTM':
            # CNN处理
            x = x.permute(0, 2, 1)  # [batch, feature, seq]
            x = self.cnn(x)
            x = x.permute(0, 2, 1)  # [batch, seq_cnn, channels]
            
            # LSTM处理
            lstm_out, (h_n, c_n) = self.lstm(x)
            x = lstm_out[:, -1, :]  # 取最后一个时间步
            
            # 特征融合
            x = self.fusion(x)
            
            # 双输出
            position = self.position_head(x)
            force = self.force_head(x)
            
        elif self.model_type == 'Transformer':
            # 输入投影
            x = self.input_projection(x)  # [batch, seq, d_model]
            
            # 添加位置编码
            x = x + self.position_encoding[:, :x.size(1), :]
            
            # Transformer处理
            x = self.transformer(x)
            x = x[:, -1, :]  # 取最后一个时间步
            
            # 特征融合
            x = self.fusion(x)
            
            # 双输出
            position = self.position_head(x)
            force = self.force_head(x)
            
        elif self.model_type == 'SimpleMLP':
            # 扁平化
            x = x.reshape(batch_size, -1)
            
            # MLP处理
            x = self.mlp(x)
            
            # 双输出
            position = self.position_head(x)
            force = self.force_head(x)
        
        return position, force
    
    def _count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhysicsInformedLoss(nn.Module):
    """
    物理信息损失函数
    加入物理约束的损失计算
    """
    
    def __init__(self, config: dict):
        super(PhysicsInformedLoss, self).__init__()
        
        self.config = config
        self.loss_weights = config.get('loss_weights', {'position': 0.6, 'force': 0.4})
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # 物理约束参数
        self.force_min = 0.0  # 力大小应为正数
        self.position_bounds = {'x_min': 0, 'x_max': 100, 'y_min': 0, 'y_max': 100}
        
    def forward(self, pred_position: torch.Tensor, pred_force: torch.Tensor,
                target_position: torch.Tensor, target_force: torch.Tensor) -> dict:
        """
        计算总损失
        
        参数:
            pred_position: 预测位置 [batch, 2]
            pred_force: 预测力大小 [batch, 1]
            target_position: 真实位置 [batch, 2]
            target_force: 真实力大小 [batch, 1]
            
        返回:
            损失字典
        """
        # 基础损失
        position_loss = self.mse_loss(pred_position, target_position)
        force_loss = self.mse_loss(pred_force, target_force)
        
        # 物理约束损失
        physics_loss = self._compute_physics_loss(pred_position, pred_force)
        
        # 组合损失
        total_loss = (self.loss_weights['position'] * position_loss +
                     self.loss_weights['force'] * force_loss +
                     0.1 * physics_loss)
        
        # 额外指标
        position_mae = self.mae_loss(pred_position, target_position)
        force_mae = self.mae_loss(pred_force, target_force)
        
        return {
            'total': total_loss,
            'position': position_loss,
            'force': force_loss,
            'physics': physics_loss,
            'position_mae': position_mae,
            'force_mae': force_mae
        }
    
    def _compute_physics_loss(self, position: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """计算物理约束损失"""
        physics_loss = 0.0
        
        # 1. 力大小应为正数
        force_positive_loss = torch.relu(-force).mean()
        physics_loss += force_positive_loss
        
        # 2. 位置应在合理范围内
        x, y = position[:, 0], position[:, 1]
        bounds = self.position_bounds
        
        position_bound_loss = (
            torch.relu(x - bounds['x_max']) +
            torch.relu(bounds['x_min'] - x) +
            torch.relu(y - bounds['y_max']) +
            torch.relu(bounds['y_min'] - y)
        ).mean()
        
        physics_loss += position_bound_loss * 0.1
        
        # 3. 力大小与位置的相关性（可选）
        # 这里可以添加更多物理约束
        
        return physics_loss