import yaml
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import pandas as pd

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: dict, save_path: str):
    """保存配置"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算评估指标"""
    metrics = {}
    
    # 位置误差（x, y分别计算）
    if y_true.shape[1] >= 2 and y_pred.shape[1] >= 2:
        metrics['mae_x'] = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        metrics['mae_y'] = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
        metrics['mae_pos'] = (metrics['mae_x'] + metrics['mae_y']) / 2
        
        metrics['rmse_x'] = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        metrics['rmse_y'] = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
        metrics['rmse_pos'] = np.sqrt((metrics['rmse_x']**2 + metrics['rmse_y']**2) / 2)
        
        metrics['r2_x'] = r2_score(y_true[:, 0], y_pred[:, 0])
        metrics['r2_y'] = r2_score(y_true[:, 1], y_pred[:, 1])
        metrics['r2_pos'] = (metrics['r2_x'] + metrics['r2_y']) / 2
    
    # 力的大小误差
    if y_true.shape[1] >= 3 and y_pred.shape[1] >= 3:
        metrics['mae_force'] = mean_absolute_error(y_true[:, 2], y_pred[:, 2])
        metrics['rmse_force'] = np.sqrt(mean_squared_error(y_true[:, 2], y_pred[:, 2]))
        metrics['r2_force'] = r2_score(y_true[:, 2], y_pred[:, 2])
    
    return metrics

def plot_results(y_true: np.ndarray, y_pred: np.ndarray, title: str = "预测结果对比"):
    """绘制预测结果对比图"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 位置散点图
    ax = axes[0, 0]
    ax.scatter(y_true[:, 0], y_true[:, 1], alpha=0.6, label='真实位置', s=30)
    ax.scatter(y_pred[:, 0], y_pred[:, 1], alpha=0.6, label='预测位置', s=30)
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title('受力点位置预测')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 力的大小对比
    ax = axes[0, 1]
    samples = range(min(50, len(y_true)))
    ax.plot(samples, y_true[:50, 2], 'b-', label='真实力大小', linewidth=2)
    ax.plot(samples, y_pred[:50, 2], 'r--', label='预测力大小', linewidth=2)
    ax.set_xlabel('事件索引')
    ax.set_ylabel('力的大小')
    ax.set_title('力大小预测对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. X坐标误差
    ax = axes[1, 0]
    error_x = y_pred[:, 0] - y_true[:, 0]
    ax.hist(error_x, bins=30, alpha=0.7, color='blue')
    ax.set_xlabel('X坐标预测误差')
    ax.set_ylabel('频数')
    ax.set_title(f'X坐标误差分布 (均值: {np.mean(error_x):.4f})')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    
    # 4. Y坐标误差
    ax = axes[1, 1]
    error_y = y_pred[:, 1] - y_true[:, 1]
    ax.hist(error_y, bins=30, alpha=0.7, color='green')
    ax.set_xlabel('Y坐标预测误差')
    ax.set_ylabel('频数')
    ax.set_title(f'Y坐标误差分布 (均值: {np.mean(error_y):.4f})')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/prediction_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def validate_event_data(df: pd.DataFrame, config: dict) -> bool:
    """
    验证事件数据的完整性
    
    参数:
        df: 数据DataFrame
        config: 配置字典
        
    返回:
        是否验证通过
    """
    data_config = config.get('data', {})
    
    event_id_column = data_config.get('event_id_column', 'event_id')
    feature_columns = data_config.get('feature_columns', [])
    label_columns = data_config.get('label_columns', [])
    
    print("="*60)
    print("事件数据验证")
    print("="*60)
    
    # 检查必要的列
    required_columns = [event_id_column] + feature_columns + label_columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"✗ 缺少必要的列: {missing_columns}")
        return False
    else:
        print("✓ 所有必要列都存在")
    
    # 获取所有事件ID
    event_ids = df[event_id_column].unique()
    n_events = len(event_ids)
    
    print(f"✓ 发现 {n_events} 个独立事件")
    
    # 检查每个事件的数据
    issues = []
    for event_id in event_ids[:5]:  # 只检查前5个作为样本
        event_data = df[df[event_id_column] == event_id]
        
        # 检查事件是否有足够的数据
        if len(event_data) < data_config.get('sequence_length', 100):
            issues.append(f"事件 {event_id}: 时间步数不足 ({len(event_data)} 个时间步)")
        
        # 检查标签一致性
        label_values = event_data[label_columns].values
        first_label = label_values[0]
        for i in range(1, len(label_values)):
            if not np.allclose(label_values[i], first_label, atol=1e-6):
                issues.append(f"事件 {event_id}: 标签在时间序列中不一致")
                break
    
    if issues:
        print("发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        print("建议: 确保每个事件的所有时间步有相同的标签")
        return False
    else:
        print("✓ 事件数据验证通过")
        return True

def save_training_report(config: dict, results: dict, timestamp: str):
    """保存训练报告"""
    report = {
        'config': config,
        'timestamp': timestamp,
        'best_epoch': results.get('best_epoch', 0),
        'best_val_loss': results.get('best_val_loss', 0),
        'test_metrics': results.get('test_results', {}).get('metrics', {}),
        'training_history': results.get('history', {})
    }
    
    with open(f'results/training_report_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"训练报告已保存: results/training_report_{timestamp}.json")