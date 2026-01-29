import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

class SimpleLoss(nn.Module):
    """简化损失函数"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_weights = config.get('training', {}).get('loss_weights', {'position': 0.6, 'force': 0.4})
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def forward(self, pred_position, pred_force, target_position, target_force):
        position_loss = self.mse_loss(pred_position, target_position)
        force_loss = self.mse_loss(pred_force, target_force)
        total_loss = (self.loss_weights['position'] * position_loss + 
                     self.loss_weights['force'] * force_loss)
        
        return {
            'total': total_loss,
            'position': position_loss,
            'force': force_loss,
            'position_mae': self.mae_loss(pred_position, target_position),
            'force_mae': self.mae_loss(pred_force, target_force)
        }

class StrainForceTrainer:
    """
    应变到力神经网络的训练器（简化版本）
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.training_config = config.get('training', {})
        self.experiment_config = config.get('experiment', {})
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.experiment_name = self.experiment_config.get('name', 'strain_force_experiment')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_directories()
        
        self.model = None
        self.criterion = SimpleLoss(config)  # 使用简化损失函数
        self.optimizer = None
        self.scheduler = None
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_position_loss': [], 'val_position_loss': [],
            'train_force_loss': [], 'val_force_loss': [],
            'learning_rate': []
        }
        
        if self.experiment_config.get('use_tensorboard', False):
            self.writer = SummaryWriter(f"{self.experiment_config.get('log_dir', 'logs')}/{self.timestamp}")
        else:
            self.writer = None
        
        print(f"训练器初始化完成 - 实验: {self.experiment_name}")
    
    def _create_directories(self):
        directories = ['models', 'results', 'logs', 'results/figures']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("输出目录创建完成")
    
    def setup_model(self, model: nn.Module):
        self.model = model.to(self.device)
        
        optimizer_name = self.training_config.get('optimizer', 'AdamW')
        learning_rate = self.training_config.get('learning_rate', 0.001)
        weight_decay = self.training_config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        
        # 学习率调度器
        scheduler_name = self.training_config.get('scheduler', 'ReduceLROnPlateau')
        
        if scheduler_name.lower() == 'none' or scheduler_name.lower() == 'false':
            self.scheduler = None
            print("不使用学习率调度器")
        elif scheduler_name == 'ReduceLROnPlateau':
            # 确保所有参数都是浮点数
            factor = float(self.training_config.get('scheduler_factor', 0.5))
            patience = int(self.training_config.get('scheduler_patience', 10))
            min_lr = float(self.training_config.get('min_lr', 0.000001))
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=True
            )
        elif scheduler_name == 'CosineAnnealing':
            T_max = int(self.training_config.get('epochs', 100))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max
            )
        
        print(f"模型设置完成: {optimizer_name}优化器, {scheduler_name}调度器")
    
    def create_dataloaders(self, datasets):
        batch_size = self.training_config.get('batch_size', 32)
        dataloaders = {}
        
        for name, (X, y) in datasets.items():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
            shuffle = (name == 'train')
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            dataloaders[name] = dataloader
            
            print(f"{name} DataLoader: {len(dataset)} 个样本, {len(dataloader)} 个批次")
        
        return dataloaders
    
    def train_epoch(self, dataloader):
        self.model.train()
        epoch_losses = {'total': 0.0, 'position': 0.0, 'force': 0.0}
        n_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc="训练", leave=False)
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            target_position = batch_y[:, :2]
            target_force = batch_y[:, 2:]
            
            self.optimizer.zero_grad()
            pred_position, pred_force = self.model(batch_X)
            
            losses = self.criterion(pred_position, pred_force, target_position, target_force)
            losses['total'].backward()
            
            gradient_clip = self.training_config.get('gradient_clip', 1.0)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            self.optimizer.step()
            
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            pbar.set_postfix({'loss': losses['total'].item()})
        
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证一个epoch
        
        参数:
            dataloader: 验证DataLoader
            
        返回:
            验证指标字典
        """
        self.model.eval()
        
        epoch_losses = {
            'total': 0.0, 'position': 0.0, 'force': 0.0,
            'physics': 0.0, 'position_mae': 0.0, 'force_mae': 0.0
        }
        
        all_predictions = []
        all_targets = []
        
        n_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                # 移动到设备
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # 分离位置和力标签
                target_position = batch_y[:, :2]
                target_force = batch_y[:, 2:]
                
                # 前向传播
                pred_position, pred_force = self.model(batch_X)
                
                # 计算损失
                losses = self.criterion(pred_position, pred_force,
                                       target_position, target_force)
                
                # 累计损失
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key].item()
                
                # 收集预测结果用于评估
                batch_pred = torch.cat([pred_position, pred_force], dim=1).cpu().numpy()
                batch_target = batch_y.cpu().numpy()
                
                all_predictions.append(batch_pred)
                all_targets.append(batch_target)
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # 合并所有预测结果
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 计算额外指标
        from sklearn.metrics import r2_score, mean_absolute_error
        
        # 位置指标
        pos_pred = all_predictions[:, :2]
        pos_true = all_targets[:, :2]
        
        epoch_losses['position_r2_x'] = r2_score(pos_true[:, 0], pos_pred[:, 0])
        epoch_losses['position_r2_y'] = r2_score(pos_true[:, 1], pos_pred[:, 1])
        epoch_losses['position_r2_mean'] = (epoch_losses['position_r2_x'] + epoch_losses['position_r2_y']) / 2
        
        # 力大小指标
        force_pred = all_predictions[:, 2]
        force_true = all_targets[:, 2]
        
        epoch_losses['force_r2'] = r2_score(force_true, force_pred)
        
        return epoch_losses, (all_predictions, all_targets)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              test_loader: Optional[DataLoader] = None) -> Dict:
        """
        完整训练流程
        
        参数:
            train_loader: 训练DataLoader
            val_loader: 验证DataLoader
            test_loader: 测试DataLoader
            
        返回:
            训练结果字典
        """
        print("="*50)
        print("开始训练")
        print("="*50)
        
        epochs = self.training_config.get('epochs', 100)
        patience = self.training_config.get('patience', 20)
        min_delta = self.training_config.get('min_delta', 0.001)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        best_model_state = None
        best_epoch = 0
        
        # 训练循环
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 30)
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_losses, _ = self.validate_epoch(val_loader)
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_position_loss'].append(train_losses['position'])
            self.history['val_position_loss'].append(val_losses['position'])
            self.history['train_force_loss'].append(train_losses['force'])
            self.history['val_force_loss'].append(val_losses['force'])
            self.history['learning_rate'].append(current_lr)
            
            # 记录到TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_losses['total'], epoch)
                self.writer.add_scalar('Loss/val', val_losses['total'], epoch)
                self.writer.add_scalar('Loss/train_position', train_losses['position'], epoch)
                self.writer.add_scalar('Loss/val_position', val_losses['position'], epoch)
                self.writer.add_scalar('Loss/train_force', train_losses['force'], epoch)
                self.writer.add_scalar('Loss/val_force', val_losses['force'], epoch)
                self.writer.add_scalar('Metrics/val_position_r2', val_losses['position_r2_mean'], epoch)
                self.writer.add_scalar('Metrics/val_force_r2', val_losses['force_r2'], epoch)
                self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # 打印指标
            print(f"训练损失: {train_losses['total']:.6f} (位置: {train_losses['position']:.6f}, "
                  f"力: {train_losses['force']:.6f})")
            print(f"验证损失: {val_losses['total']:.6f} (位置: {val_losses['position']:.6f}, "
                  f"力: {val_losses['force']:.6f})")
            print(f"位置R²: X={val_losses['position_r2_x']:.4f}, "
                  f"Y={val_losses['position_r2_y']:.4f}, 平均={val_losses['position_r2_mean']:.4f}")
            print(f"力大小R²: {val_losses['force_r2']:.4f}")
            print(f"学习率: {current_lr:.6f}")
            
            # 早停检查
            if val_losses['total'] < best_val_loss - min_delta:
                best_val_loss = val_losses['total']
                patience_counter = 0
                best_epoch = epoch
                best_model_state = self.model.state_dict().copy()
                
                # 保存最佳模型
                if self.experiment_config.get('save_model', True):
                    self.save_model(f"models/best_model_{self.timestamp}.pth", epoch=epoch)
                
                print(f"✓ 最佳模型更新! 验证损失: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"早停计数: {patience_counter}/{patience}")
            
            # 定期保存检查点
            if self.experiment_config.get('save_checkpoints', True):
                checkpoint_freq = self.experiment_config.get('checkpoint_frequency', 10)
                if epoch % checkpoint_freq == 0:
                    self.save_model(f"models/checkpoint_epoch_{epoch}.pth", epoch=epoch)
            
            # 检查早停
            if patience_counter >= patience:
                print(f"\n早停触发! 在epoch {epoch}停止训练")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"恢复最佳模型 (epoch {best_epoch})")
        
        # 最终测试
        test_results = None
        if test_loader is not None:
            print("\n" + "="*50)
            print("最终测试")
            print("="*50)
            
            test_losses, (test_pred, test_true) = self.validate_epoch(test_loader)
            
            test_results = {
                'losses': test_losses,
                'predictions': test_pred,
                'targets': test_true
            }
            
            print(f"测试损失: {test_losses['total']:.6f}")
            print(f"位置R²: {test_losses['position_r2_mean']:.4f}")
            print(f"力大小R²: {test_losses['force_r2']:.4f}")
        
        # 保存训练历史
        self._save_training_history()
        
        # 关闭TensorBoard
        if self.writer is not None:
            self.writer.close()
        
        print("\n" + "="*50)
        print("训练完成!")
        print("="*50)
        
        return {
            'history': self.history,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'test_results': test_results
        }
    
    def save_model(self, filepath: str, epoch: int = None):
        """
        保存模型
        
        参数:
            filepath: 保存路径
            epoch: 当前epoch
        """
        save_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        if self.scheduler is not None:
            save_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(save_data, filepath)
        print(f"模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        参数:
            filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载历史
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"模型已加载: {filepath}")
        print(f"训练epoch: {checkpoint.get('epoch', 'unknown')}")
    
    def _save_training_history(self):
        """保存训练历史"""
        history_file = f"results/training_history_{self.timestamp}.json"
        
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"训练历史已保存: {history_file}")
    
    def plot_training_history(self, save: bool = True):
        """绘制训练历史"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 总损失
        ax = axes[0, 0]
        ax.plot(self.history['train_loss'], label='训练损失', linewidth=2)
        ax.plot(self.history['val_loss'], label='验证损失', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('损失')
        ax.set_title('总损失曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 位置损失
        ax = axes[0, 1]
        ax.plot(self.history['train_position_loss'], label='训练位置损失', linewidth=2)
        ax.plot(self.history['val_position_loss'], label='验证位置损失', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('损失')
        ax.set_title('位置损失曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 力损失
        ax = axes[1, 0]
        ax.plot(self.history['train_force_loss'], label='训练力损失', linewidth=2)
        ax.plot(self.history['val_force_loss'], label='验证力损失', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('损失')
        ax.set_title('力大小损失曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 学习率
        ax = axes[1, 1]
        ax.plot(self.history['learning_rate'], label='学习率', linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('学习率')
        ax.set_title('学习率变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = f"results/figures/training_curves_{self.timestamp}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存: {save_path}")
        
        plt.show()
        return fig