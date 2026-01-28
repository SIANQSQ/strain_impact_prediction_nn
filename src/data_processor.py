import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Tuple, Dict, List, Optional

class StrainDataProcessor:
    """
    处理四点应变时序数据，输出施力点坐标和力大小
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
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        加载数据
        
        参数:
            file_path: 数据文件路径
            
        返回:
            pandas DataFrame
        """
        if file_path is None:
            file_path = self.data_config.get('input_csv', 'data/raw/abaqus_strain_data.csv')
        
        print(f"正在加载数据: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 加载CSV
        try:
            df = pd.read_csv(file_path)
            print(f"数据加载成功，形状: {df.shape}")
            print(f"数据列: {list(df.columns)}")
            
            # 验证必要的列是否存在
            required_columns = self.data_config.get('feature_columns', []) + \
                             self.data_config.get('label_columns', [])
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            self.raw_data = df
            return df
            
        except Exception as e:
            raise Exception(f"数据加载失败: {e}")
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加有物理意义的衍生特征
        
        参数:
            df: 原始数据
            
        返回:
            添加衍生特征后的DataFrame
        """
        print("添加衍生特征...")
        
        # 1. 应变梯度特征（反映力的方向）
        df['strain_gradient_v'] = df['strain_top'] - df['strain_bottom']  # 垂直梯度
        df['strain_gradient_h'] = df['strain_left'] - df['strain_right']  # 水平梯度
        
        # 2. 应变比值特征
        df['strain_ratio_tb'] = df['strain_top'] / (df['strain_bottom'] + 1e-8)
        df['strain_ratio_lr'] = df['strain_left'] / (df['strain_right'] + 1e-8)
        
        # 3. 统计特征
        strain_columns = ['strain_top', 'strain_bottom', 'strain_left', 'strain_right']
        df['strain_mean'] = df[strain_columns].mean(axis=1)
        df['strain_std'] = df[strain_columns].std(axis=1)
        df['strain_max'] = df[strain_columns].max(axis=1)
        df['strain_min'] = df[strain_columns].min(axis=1)
        
        # 4. 复合特征
        df['strain_product_tl'] = df['strain_top'] * df['strain_left']
        df['strain_product_br'] = df['strain_bottom'] * df['strain_right']
        
        # 5. 归一化应变
        df['strain_normalized_top'] = df['strain_top'] / (df['strain_mean'] + 1e-8)
        df['strain_normalized_bottom'] = df['strain_bottom'] / (df['strain_mean'] + 1e-8)
        
        print(f"添加衍生特征后数据形状: {df.shape}")
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        归一化数据
        
        参数:
            df: 原始数据
            
        返回:
            (归一化特征, 归一化标签)
        """
        print("数据归一化...")
        
        # 获取特征和标签列
        feature_columns = self.data_config.get('feature_columns', [])
        label_columns = self.data_config.get('label_columns', [])
        
        # 如果需要添加衍生特征
        if self.data_config.get('add_derived_features', False):
            df = self.add_derived_features(df)
            # 更新特征列
            feature_columns = [col for col in df.columns 
                              if col.startswith('strain') and col not in label_columns]
        
        # 提取特征和标签
        X = df[feature_columns].values
        y = df[label_columns].values
        
        print(f"特征形状: {X.shape}, 标签形状: {y.shape}")
        
        # 归一化
        X_normalized = self.scaler_features.fit_transform(X)
        y_normalized = self.scaler_labels.fit_transform(y)
        
        # 保存归一化器
        self._save_scalers()
        
        return X_normalized, y_normalized
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时序序列
        
        参数:
            X: 特征数组 [n_samples, n_features]
            y: 标签数组 [n_samples, n_labels]
            
        返回:
            (序列特征, 序列标签)
        """
        print("创建时序序列...")
        
        seq_length = self.data_config.get('sequence_length', 20)
        step_size = self.data_config.get('step_size', 1)
        
        X_sequences = []
        y_sequences = []
        
        n_samples = len(X)
        
        for i in range(0, n_samples - seq_length + 1, step_size):
            X_sequences.append(X[i:i+seq_length])
            # 使用序列最后一个时间步的标签
            y_sequences.append(y[i+seq_length-1])
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences)
        
        print(f"序列数据形状: X={X_seq.shape}, y={y_seq.shape}")
        return X_seq, y_seq
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        划分数据集
        
        参数:
            X: 特征序列
            y: 标签序列
            
        返回:
            划分后的数据集字典
        """
        print("划分数据集...")
        
        train_ratio = self.data_config.get('train_ratio', 0.7)
        val_ratio = self.data_config.get('val_ratio', 0.15)
        test_ratio = self.data_config.get('test_ratio', 0.15)
        random_seed = self.data_config.get('random_seed', 42)
        
        # 计算划分索引
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.data_config.get('shuffle', True):
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
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
        完整的数据处理流程
        
        参数:
            data_path: 数据文件路径
            
        返回:
            处理后的数据集
        """
        print("="*50)
        print("开始数据处理流程")
        print("="*50)
        
        # 1. 加载数据
        df = self.load_data(data_path)
        
        # 2. 归一化数据
        X, y = self.normalize_data(df)
        
        # 3. 创建序列
        X_seq, y_seq = self.create_sequences(X, y)
        
        # 4. 划分数据集
        datasets = self.split_data(X_seq, y_seq)
        
        # 5. 保存处理后的数据
        self._save_processed_data(X_seq, y_seq)
        
        print("数据处理完成！")
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
        
        np.savez_compressed('data/processed/sequences.npz', X=X, y=y)
        print("处理后的数据已保存到 data/processed/sequences.npz")
    
    def _save_splits(self, datasets: Dict):
        """保存数据集划分"""
        os.makedirs('data/splits', exist_ok=True)
        
        for name, (X, y) in datasets.items():
            np.savez_compressed(f'data/splits/{name}.npz', X=X, y=y)
        
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
    def generate_synthetic_data(n_samples: int = 10000, save_path: str = None):
        """
        生成合成数据用于测试
        
        参数:
            n_samples: 样本数量
            save_path: 保存路径
        """
        print(f"生成 {n_samples} 个合成样本...")
        
        data = []
        
        for i in range(n_samples):
            # 随机生成施力点位置和大小
            force_x = np.random.uniform(0, 100)
            force_y = np.random.uniform(0, 100)
            force_magnitude = np.random.uniform(50, 500)
            
            # 生成时间序列（20个时间步）
            time_steps = 20
            time = np.linspace(0, 1, time_steps)
            
            for t in time:
                # 基于位置和力大小计算应变（简化物理模型）
                # 假设应变随距离增大而减小
                distance_top = np.sqrt((force_x - 50)**2 + (force_y - 100)**2)
                distance_bottom = np.sqrt((force_x - 50)**2 + (force_y - 0)**2)
                distance_left = np.sqrt((force_x - 0)**2 + (force_y - 50)**2)
                distance_right = np.sqrt((force_x - 100)**2 + (force_y - 50)**2)
                
                # 计算应变（力大小 / (距离 + 1) * 衰减函数）
                strain_top = force_magnitude / (distance_top + 1) * np.exp(-distance_top/50)
                strain_bottom = force_magnitude / (distance_bottom + 1) * np.exp(-distance_bottom/50)
                strain_left = force_magnitude / (distance_left + 1) * np.exp(-distance_left/50)
                strain_right = force_magnitude / (distance_right + 1) * np.exp(-distance_right/50)
                
                # 添加时间变化
                time_factor = 1 + 0.1 * np.sin(2 * np.pi * t)
                strain_top *= time_factor
                strain_bottom *= time_factor
                strain_left *= time_factor
                strain_right *= time_factor
                
                # 添加噪声
                noise_level = 0.02
                strain_top += np.random.normal(0, strain_top * noise_level)
                strain_bottom += np.random.normal(0, strain_bottom * noise_level)
                strain_left += np.random.normal(0, strain_left * noise_level)
                strain_right += np.random.normal(0, strain_right * noise_level)
                
                data.append([
                    t, strain_top, strain_bottom, strain_left, strain_right,
                    force_x, force_y, force_magnitude
                ])
        
        # 创建DataFrame
        columns = ['time_step', 'strain_top', 'strain_bottom', 
                  'strain_left', 'strain_right', 'force_x', 
                  'force_y', 'force_magnitude']
        
        df = pd.DataFrame(data, columns=columns)
        
        # 保存数据
        if save_path is None:
            save_path = 'data/raw/synthetic_data.csv'
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        
        print(f"合成数据已保存到: {save_path}")
        print(f"数据形状: {df.shape}")
        
        return df