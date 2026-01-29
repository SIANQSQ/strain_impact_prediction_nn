import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Tuple, Dict, List, Optional

class StrainDataProcessor:
    """
    处理按事件分组的四点应变时序数据
    每个事件是一个独立的样本，包含固定长度的时间序列
    """
    
    def __init__(self, config: dict):
        """
        初始化数据处理类
        
        参数:
            config: 配置字典
        """
        self.config = config
        self.data_config = config.get('data', {})
        
        # 初始化归一化器
        self.scaler_features = StandardScaler()
        self.scaler_labels = StandardScaler()
        
        # 数据缓存
        self.raw_data = None
        self.processed_data = None
        self.sequences = None
        
    def load_data(self, file_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        按事件ID加载数据
        
        参数:
            file_path: 数据文件路径
            
        返回:
            X: 特征数据 [n_events, sequence_length, n_features]
            y: 标签数据 [n_events, n_labels]
        """
        if file_path is None:
            file_path = self.data_config.get('input_csv', 'data/raw/event_strain_data.csv')
        
        print(f"加载事件数据: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        try:
            # 加载CSV
            df = pd.read_csv(file_path)
            print(f"数据加载成功，原始形状: {df.shape}")
            
            # 验证必要的列是否存在
            required_columns = [self.data_config.get('event_id_column', 'event_id')] + \
                              self.data_config.get('feature_columns', []) + \
                              self.data_config.get('label_columns', [])
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            self.raw_data = df
            
            # 按事件ID处理数据
            return self._process_event_data(df)
            
        except Exception as e:
            raise Exception(f"数据加载失败: {e}")
    
    def _process_event_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        按事件ID处理数据，每个事件作为一个独立样本
        
        参数:
            df: 原始数据
            
        返回:
            X: 特征数据 [n_events, sequence_length, n_features]
            y: 标签数据 [n_events, n_labels]
        """
        event_id_column = self.data_config.get('event_id_column', 'event_id')
        feature_columns = self.data_config.get('feature_columns', [])
        label_columns = self.data_config.get('label_columns', [])
        time_column = self.data_config.get('time_column', 'time_step')
        target_seq_length = self.data_config.get('sequence_length', 100)
        pad_mode = self.data_config.get('pad_mode', 'edge')
        
        # 获取所有事件ID
        event_ids = df[event_id_column].unique()
        n_events = len(event_ids)
        
        print(f"发现 {n_events} 个独立冲击事件")
        
        X_list = []
        y_list = []
        
        # 处理每个事件
        for i, event_id in enumerate(event_ids):
            # 提取该事件的所有数据
            event_data = df[df[event_id_column] == event_id]
            
            # 按时间排序
            if time_column in event_data.columns:
                event_data = event_data.sort_values(time_column)
            
            # 提取特征和标签
            event_features = event_data[feature_columns].values  # [n_steps, n_features]
            event_labels = event_data[label_columns].iloc[0].values  # [n_labels]，取第一个时间步的标签
            
            # 验证标签在整个事件中是否一致
            for j in range(1, len(event_data)):
                if not np.allclose(event_data[label_columns].iloc[j].values, event_labels, atol=1e-6):
                    print(f"警告: 事件 {event_id} 的标签在时间序列中不一致")
                    break
            
            # 调整序列长度
            event_features = self._adjust_sequence_length(event_features, target_seq_length, pad_mode)
            
            # 添加衍生特征（如果需要）
            if self.data_config.get('add_derived_features', False):
                event_features = self._add_event_features(event_features)
            
            X_list.append(event_features)
            y_list.append(event_labels)
            
            # 显示进度
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{n_events} 个事件")
        
        # 转换为数组
        X = np.array(X_list)  # [n_events, sequence_length, n_features]
        y = np.array(y_list)  # [n_events, n_labels]
        
        print(f"数据处理完成，特征形状: {X.shape}, 标签形状: {y.shape}")
        
        return X, y
    
    def _adjust_sequence_length(self, sequence: np.ndarray, target_length: int, 
                               pad_mode: str = 'edge') -> np.ndarray:
        """
        调整序列长度到目标长度
        
        参数:
            sequence: 原始序列 [n_steps, n_features]
            target_length: 目标长度
            pad_mode: 填充模式 ('edge', 'zero', 'mean', 'constant')
            
        返回:
            调整后的序列 [target_length, n_features]
        """
        current_length = sequence.shape[0]
        n_features = sequence.shape[1]
        
        if current_length == target_length:
            return sequence
        elif current_length > target_length:
            # 截断
            truncate_mode = self.data_config.get('truncate_mode', 'end')
            if truncate_mode == 'start':
                return sequence[:target_length]
            elif truncate_mode == 'end':
                return sequence[-target_length:]
            elif truncate_mode == 'random':
                start = np.random.randint(0, current_length - target_length + 1)
                return sequence[start:start + target_length]
            else:
                return sequence[:target_length]
        else:
            # 填充
            pad_width = target_length - current_length
            
            if pad_mode == 'edge':
                # 使用边缘值填充
                last_row = sequence[-1:]
                padding = np.repeat(last_row, pad_width, axis=0)
                return np.vstack([sequence, padding])
                
            elif pad_mode == 'zero':
                # 零填充
                padding = np.zeros((pad_width, n_features))
                return np.vstack([sequence, padding])
                
            elif pad_mode == 'mean':
                # 使用平均值填充
                mean_row = np.mean(sequence, axis=0, keepdims=True)
                padding = np.repeat(mean_row, pad_width, axis=0)
                return np.vstack([sequence, padding])
                
            elif pad_mode == 'constant':
                # 使用常数值填充
                constant_value = 0.0
                padding = np.full((pad_width, n_features), constant_value)
                return np.vstack([sequence, padding])
                
            else:
                # 默认使用边缘值
                last_row = sequence[-1:]
                padding = np.repeat(last_row, pad_width, axis=0)
                return np.vstack([sequence, padding])
    
    def _add_event_features(self, event_features: np.ndarray) -> np.ndarray:
        """
        为事件添加衍生特征
        
        参数:
            event_features: 原始特征 [sequence_length, n_features]
            
        返回:
            添加衍生特征后的特征
        """
        sequence_length, n_features = event_features.shape
        
        # 如果没有特征，直接返回
        if n_features != 4:
            print(f"警告: 期望4个特征，实际有 {n_features} 个")
            return event_features
        
        # 添加衍生特征
        derived_features = []
        
        # 1. 应变梯度
        gradient_v = event_features[:, 0] - event_features[:, 1]  # 垂直梯度
        gradient_h = event_features[:, 2] - event_features[:, 3]  # 水平梯度
        
        # 2. 应变比值
        ratio_tb = event_features[:, 0] / (event_features[:, 1] + 1e-8)  # 顶部/底部
        ratio_lr = event_features[:, 2] / (event_features[:, 3] + 1e-8)  # 左侧/右侧
        
        # 3. 统计特征（随时间滑动窗口）
        window_size = min(10, sequence_length)
        for i in range(sequence_length):
            start = max(0, i - window_size // 2)
            end = min(sequence_length, i + window_size // 2)
            window = event_features[start:end, :]
            
            # 窗口内的统计特征
            mean_val = np.mean(window, axis=0)
            std_val = np.std(window, axis=0)
            
            # 添加到衍生特征
            derived_features.append(np.concatenate([
                event_features[i],
                [gradient_v[i], gradient_h[i], ratio_tb[i], ratio_lr[i]],
                mean_val, std_val
            ]))
        
        return np.array(derived_features)
    
    def normalize_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        归一化数据（事件数据不需要滑动窗口）
        
        参数:
            X: 特征数据 [n_events, sequence_length, n_features]
            y: 标签数据 [n_events, n_labels]
            
        返回:
            (归一化特征, 归一化标签)
        """
        print("数据归一化...")
        
        # 重塑特征数据以便归一化
        original_shape = X.shape
        X_reshaped = X.reshape(-1, original_shape[-1])
        
        # 归一化特征
        X_normalized = self.scaler_features.fit_transform(X_reshaped)
        X = X_normalized.reshape(original_shape)
        
        # 归一化标签
        y_normalized = self.scaler_labels.fit_transform(y)
        
        # 保存归一化器
        self._save_scalers()
        
        print(f"特征归一化完成，形状: {X.shape}")
        print(f"标签归一化完成，形状: {y_normalized.shape}")
        
        return X, y_normalized
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        划分数据集（按事件划分）
        
        参数:
            X: 特征数据 [n_events, sequence_length, n_features]
            y: 标签数据 [n_events, n_labels]
            
        返回:
            划分后的数据集字典
        """
        print("划分数据集...")
        
        train_ratio = self.data_config.get('train_ratio', 0.7)
        val_ratio = self.data_config.get('val_ratio', 0.15)
        test_ratio = self.data_config.get('test_ratio', 0.15)
        random_seed = self.data_config.get('random_seed', 42)
        
        # 直接使用train_test_split划分事件
        n_events = X.shape[0]
        indices = np.arange(n_events)
        
        if self.data_config.get('shuffle', True):
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        train_end = int(n_events * train_ratio)
        val_end = train_end + int(n_events * val_ratio)
        
        # 划分索引
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # 创建数据集
        datasets = {
            'train': (X[train_idx], y[train_idx]),
            'val': (X[val_idx], y[val_idx]),
            'test': (X[test_idx], y[test_idx])
        }
        
        print(f"数据集大小: 训练集={len(train_idx)}, 验证集={len(val_idx)}, 测试集={len(test_idx)}")
        
        # 保存划分
        self._save_splits(datasets)
        
        return datasets
    
    def process_pipeline(self, data_path: str = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        完整的数据处理流程（事件数据版本）
        
        参数:
            data_path: 数据文件路径
            
        返回:
            处理后的数据集
        """
        print("="*50)
        print("开始事件数据处理流程")
        print("="*50)
        
        # 1. 按事件加载数据
        X, y = self.load_data(data_path)
        
        # 2. 归一化数据
        X_normalized, y_normalized = self.normalize_data(X, y)
        
        # 3. 划分数据集（注意：这里没有创建序列，因为每个事件已经是一个序列）
        datasets = self.split_data(X_normalized, y_normalized)
        
        # 4. 保存处理后的数据
        self._save_processed_data(X_normalized, y_normalized)
        
        print("事件数据处理完成！")
        print("="*50)
        
        return datasets
    
    def _save_scalers(self):
        """保存归一化器"""
        os.makedirs('models', exist_ok=True)
        
        with open('models/scaler_features.pkl', 'wb') as f:
            pickle.dump(self.scaler_features, f)
        
        with open('models/scaler_labels.pkl', 'wb') as f:
            pickle.dump(self.scaler_labels, f)
        
        print("归一化器已保存到 models/ 目录")
    
    def _save_processed_data(self, X: np.ndarray, y: np.ndarray):
        """保存处理后的数据"""
        os.makedirs('data/processed', exist_ok=True)
        
        np.savez_compressed('data/processed/event_sequences.npz', X=X, y=y)
        print("处理后的数据已保存到 data/processed/event_sequences.npz")
    
    def _save_splits(self, datasets: Dict):
        """保存数据集划分"""
        os.makedirs('data/splits', exist_ok=True)
        
        for name, (X, y) in datasets.items():
            np.savez_compressed(f'data/splits/event_{name}.npz', X=X, y=y)
        
        print("数据集划分已保存到 data/splits/ 目录")
    
    def load_scalers(self):
        """加载归一化器"""
        try:
            with open('models/scaler_features.pkl', 'rb') as f:
                self.scaler_features = pickle.load(f)
            
            with open('models/scaler_labels.pkl', 'rb') as f:
                self.scaler_labels = pickle.load(f)
            
            print("归一化器加载成功")
            return True
        except:
            print("警告: 无法加载归一化器，将使用新归一化器")
            return False
    
    @staticmethod
    def generate_event_data(n_events: int = 1000, time_steps: int = 100, 
                           save_path: str = None):
        """
        生成事件数据用于测试
        
        参数:
            n_events: 事件数量
            time_steps: 每个事件的时间步数
            save_path: 保存路径
        """
        print(f"生成 {n_events} 个事件，每个事件 {time_steps} 个时间步...")
        
        data = []
        
        for event_id in range(1, n_events + 1):
            # 为每个事件生成随机的位置和力大小
            force_x = np.random.uniform(0, 100)
            force_y = np.random.uniform(0, 100)
            force_magnitude = np.random.uniform(50, 500)
            
            # 生成该事件的时间序列
            for t in range(time_steps):
                time_step = t / time_steps
                
                # 基于位置和力大小计算应变（简化的物理模型）
                distance_top = np.sqrt((force_x - 50)**2 + (force_y - 100)**2)
                distance_bottom = np.sqrt((force_x - 50)**2 + force_y**2)
                distance_left = np.sqrt(force_x**2 + (force_y - 50)**2)
                distance_right = np.sqrt((force_x - 100)**2 + (force_y - 50)**2)
                
                # 计算应变（力大小 / (距离 + 1) * 衰减函数）
                strain_top = force_magnitude / (distance_top + 1) * np.exp(-distance_top/50)
                strain_bottom = force_magnitude / (distance_bottom + 1) * np.exp(-distance_bottom/50)
                strain_left = force_magnitude / (distance_left + 1) * np.exp(-distance_left/50)
                strain_right = force_magnitude / (distance_right + 1) * np.exp(-distance_right/50)
                
                # 添加时间变化
                time_factor = 1 + 0.1 * np.sin(2 * np.pi * time_step)
                strain_top *= time_factor
                strain_bottom *= time_factor
                strain_left *= time_factor
                strain_right *= time_factor
                
                # 添加噪声
                noise_level = 0.02
                noise = np.random.normal(0, noise_level, 4)
                
                data.append([
                    event_id, time_step,
                    strain_top + noise[0], strain_bottom + noise[1],
                    strain_left + noise[2], strain_right + noise[3],
                    force_x, force_y, force_magnitude
                ])
            
            # 显示进度
            if event_id % 100 == 0:
                print(f"已生成 {event_id}/{n_events} 个事件")
        
        # 创建DataFrame
        columns = [
            'event_id', 'time_step',
            'strain_top', 'strain_bottom', 'strain_left', 'strain_right',
            'force_x', 'force_y', 'force_magnitude'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        
        # 保存数据
        if save_path is None:
            save_path = 'data/raw/event_strain_data.csv'
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        
        print(f"事件数据已保存到: {save_path}")
        print(f"数据形状: {df.shape}")
        
        return df