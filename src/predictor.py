import torch
import numpy as np
import pickle
from typing import Dict, List, Union, Optional

class StrainForcePredictor:
    """
    应变到力的预测器
    用于加载训练好的模型并进行预测
    """
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        初始化预测器
        
        参数:
            model_path: 模型文件路径
            config_path: 配置文件路径
        """
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载配置
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 加载归一化器
        self.scaler_features = None
        self.scaler_labels = None
        self._load_scalers()
        
        print("预测器初始化完成")
    
    def _load_model(self, model_path: str):
        """加载模型"""
        if not torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_path)
        
        # 从检查点获取配置
        if 'config' in checkpoint:
            model_config = checkpoint['config'].get('model', {})
        else:
            model_config = self.config.get('model', {})
        
        # 创建模型
        from models import PositionForceNet
        model = PositionForceNet(self.config)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"模型加载成功: {model_path}")
        print(f"模型类型: {model.model_type}")
        
        return model
    
    def _load_scalers(self):
        """加载归一化器"""
        try:
            with open('models/scaler_features.pkl', 'rb') as f:
                self.scaler_features = pickle.load(f)
            
            with open('models/scaler_labels.pkl', 'rb') as f:
                self.scaler_labels = pickle.load(f)
            
            print("归一化器加载成功")
        except FileNotFoundError:
            print("警告: 归一化器文件未找到，将使用原始数据")
    
    def predict(self, strain_data: np.ndarray, 
                return_original_scale: bool = True) -> Dict[str, np.ndarray]:
        """
        预测施力点位置和力大小
        
        参数:
            strain_data: 应变数据，形状为 [batch_size, seq_length, 4] 或 [seq_length, 4]
            return_original_scale: 是否返回原始尺度（反归一化）
            
        返回:
            预测结果字典
        """
        # 检查输入形状
        if len(strain_data.shape) == 2:
            strain_data = strain_data[np.newaxis, :, :]  # 添加批次维度
        
        batch_size = strain_data.shape[0]
        
        print(f"输入数据形状: {strain_data.shape}")
        print(f"批次大小: {batch_size}")
        
        # 归一化输入
        if self.scaler_features is not None:
            original_shape = strain_data.shape
            strain_flat = strain_data.reshape(-1, strain_data.shape[-1])
            strain_normalized = self.scaler_features.transform(strain_flat)
            strain_data = strain_normalized.reshape(original_shape)
        
        # 转换为张量
        strain_tensor = torch.tensor(strain_data, dtype=torch.float32).to(self.device)
        
        # 预测
        with torch.no_grad():
            position_pred, force_pred = self.model(strain_tensor)
            
            # 移动到CPU并转换为numpy
            position_pred = position_pred.cpu().numpy()
            force_pred = force_pred.cpu().numpy()
        
        # 反归一化
        if return_original_scale and self.scaler_labels is not None:
            # 合并位置和力预测
            combined_pred = np.concatenate([position_pred, force_pred], axis=1)
            combined_original = self.scaler_labels.inverse_transform(combined_pred)
            
            position_pred = combined_original[:, :2]
            force_pred = combined_original[:, 2:]
        
        # 组织结果
        results = []
        for i in range(batch_size):
            result = {
                'position_x': float(position_pred[i, 0]),
                'position_y': float(position_pred[i, 1]),
                'force_magnitude': float(force_pred[i, 0]),
                'position_array': position_pred[i].tolist(),
                'force_array': force_pred[i].tolist()
            }
            results.append(result)
        
        # 如果是单样本，直接返回字典
        if batch_size == 1:
            return results[0]
        else:
            return {
                'batch_results': results,
                'positions': position_pred,
                'forces': force_pred
            }
    
    def predict_from_raw(self, strain_top: List[float], strain_bottom: List[float],
                        strain_left: List[float], strain_right: List[float],
                        sequence_length: int = 20) -> Dict:
        """
        从原始应变数据预测
        
        参数:
            strain_top: 顶部应变序列
            strain_bottom: 底部应变序列
            strain_left: 左侧应变序列
            strain_right: 右侧应变序列
            sequence_length: 序列长度
            
        返回:
            预测结果
        """
        # 检查输入长度
        min_length = min(len(strain_top), len(strain_bottom), 
                        len(strain_left), len(strain_right))
        
        if min_length < sequence_length:
            raise ValueError(f"输入数据长度不足，至少需要{sequence_length}个时间步")
        
        # 创建输入数组
        strain_data = np.column_stack([
            strain_top[:sequence_length],
            strain_bottom[:sequence_length],
            strain_left[:sequence_length],
            strain_right[:sequence_length]
        ])
        
        # 添加批次维度
        strain_data = strain_data[np.newaxis, :, :]
        
        return self.predict(strain_data)
    
    def evaluate_on_test_set(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict:
        """
        在测试集上评估模型
        
        参数:
            test_data: 测试数据 [n_samples, seq_length, n_features]
            test_labels: 测试标签 [n_samples, 3]
            
        返回:
            评估结果字典
        """
        print("在测试集上评估模型...")
        
        # 预测
        predictions = self.predict(test_data, return_original_scale=False)
        
        # 如果是批次结果，提取预测值
        if 'batch_results' in predictions:
            pred_positions = predictions['positions']
            pred_forces = predictions['forces']
        else:
            pred_positions = predictions['position_array']
            pred_forces = predictions['force_array']
        
        # 真实值
        true_positions = test_labels[:, :2]
        true_forces = test_labels[:, 2]
        
        # 计算指标
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {}
        
        # 位置指标
        metrics['position_mae_x'] = mean_absolute_error(true_positions[:, 0], pred_positions[:, 0])
        metrics['position_mae_y'] = mean_absolute_error(true_positions[:, 1], pred_positions[:, 1])
        metrics['position_mae_mean'] = (metrics['position_mae_x'] + metrics['position_mae_y']) / 2
        
        metrics['position_rmse_x'] = np.sqrt(mean_squared_error(true_positions[:, 0], pred_positions[:, 0]))
        metrics['position_rmse_y'] = np.sqrt(mean_squared_error(true_positions[:, 1], pred_positions[:, 1]))
        metrics['position_rmse_mean'] = np.sqrt((metrics['position_rmse_x']**2 + metrics['position_rmse_y']**2) / 2)
        
        metrics['position_r2_x'] = r2_score(true_positions[:, 0], pred_positions[:, 0])
        metrics['position_r2_y'] = r2_score(true_positions[:, 1], pred_positions[:, 1])
        metrics['position_r2_mean'] = (metrics['position_r2_x'] + metrics['position_r2_y']) / 2
        
        # 力大小指标
        metrics['force_mae'] = mean_absolute_error(true_forces, pred_forces.flatten())
        metrics['force_rmse'] = np.sqrt(mean_squared_error(true_forces, pred_forces.flatten()))
        metrics['force_r2'] = r2_score(true_forces, pred_forces.flatten())
        
        # 打印结果
        print("\n" + "="*50)
        print("模型评估结果")
        print("="*50)
        print(f"位置预测 - MAE: {metrics['position_mae_mean']:.4f}, "
              f"RMSE: {metrics['position_rmse_mean']:.4f}, "
              f"R²: {metrics['position_r2_mean']:.4f}")
        print(f"力大小预测 - MAE: {metrics['force_mae']:.4f}, "
              f"RMSE: {metrics['force_rmse']:.4f}, "
              f"R²: {metrics['force_r2']:.4f}")
        
        # 保存评估结果
        import json
        with open('results/test_evaluation.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n评估结果已保存到: results/test_evaluation.json")
        
        return {
            'metrics': metrics,
            'predictions': pred_positions,
            'true_values': true_positions,
            'force_predictions': pred_forces,
            'force_true': true_forces
        }
    
    def save_predictions(self, predictions: Dict, filepath: str = None):
        """
        保存预测结果到CSV
        
        参数:
            predictions: 预测结果字典
            filepath: 保存路径
        """
        import pandas as pd
        
        if filepath is None:
            filepath = f"results/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 创建DataFrame
        if 'batch_results' in predictions:
            results = predictions['batch_results']
            data = []
            for i, res in enumerate(results):
                data.append({
                    'sample_id': i,
                    'pred_position_x': res['position_x'],
                    'pred_position_y': res['position_y'],
                    'pred_force': res['force_magnitude']
                })
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([{
                'pred_position_x': predictions['position_x'],
                'pred_position_y': predictions['position_y'],
                'pred_force': predictions['force_magnitude']
            }])
        
        # 保存
        df.to_csv(filepath, index=False)
        print(f"预测结果已保存: {filepath}")
        
        return df