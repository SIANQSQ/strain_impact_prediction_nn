#!/usr/bin/env python3
"""
应变到力神经网络 - 一键运行脚本（事件数据版本）
四点应变时序数据（按事件分组） -> 施力点坐标 + 力大小
"""

import os
import sys
import argparse
import yaml
import numpy as np
from datetime import datetime

# 添加src目录到路径
sys.path.append('src')

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='应变到力神经网络训练和预测（事件数据版本）')
    
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'predict', 'generate', 'full', 'validate'],
                       help='运行模式: train(训练), predict(预测), generate(生成数据), full(完整流程), validate(验证数据)')
    
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='配置文件路径')
    
    parser.add_argument('--data', type=str, default=None,
                       help='数据文件路径（如果不使用配置文件中的路径）')
    
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件路径（预测模式使用）')
    
    parser.add_argument('--events', type=int, default=1000,
                       help='生成的事件数量')
    
    parser.add_argument('--seq_length', type=int, default=100,
                       help='每个事件的时间步数')
    
    args = parser.parse_args()
    
    print("="*60)
    print("应变到力神经网络系统（事件数据版本）")
    print("="*60)
    
    # 加载配置
    print(f"加载配置: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖配置参数
    if args.data is not None:
        config['data']['input_csv'] = args.data
    
    # 根据模式运行
    if args.mode == 'validate':
        validate_data(config)
        return
    
    if args.mode == 'generate' or args.mode == 'full':
        print("\n生成事件数据...")
        from src.data_processor import StrainDataProcessor
        StrainDataProcessor.generate_event_data(
            n_events=args.events,
            time_steps=args.seq_length,
            save_path=config['data']['input_csv']
        )
    
    if args.mode == 'train' or args.mode == 'full':
        print("\n训练模型...")
        train_model(config)
    
    if args.mode == 'predict' or args.mode == 'full':
        print("\n进行预测...")
        if args.model is None:
            # 查找最新的模型
            model_files = [f for f in os.listdir('models') if f.startswith('best_model')]
            if model_files:
                latest_model = max(model_files)
                model_path = f"models/{latest_model}"
            else:
                print("警告: 未找到训练好的模型，请先训练模型")
                return
        else:
            model_path = args.model
        
        predict_event_example(model_path, config)
    
    print("\n" + "="*60)
    print("运行完成!")
    print("="*60)

def validate_data(config: dict):
    """验证事件数据格式"""
    print("\n验证事件数据格式...")
    
    from src.utils import validate_event_data
    import pandas as pd
    
    data_path = config['data']['input_csv']
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return
    
    # 加载数据
    df = pd.read_csv(data_path)
    print(f"数据加载成功，形状: {df.shape}")
    
    # 验证数据格式
    is_valid = validate_event_data(df, config)
    
    if is_valid:
        print("\n✓ 事件数据格式正确，可以用于训练")
    else:
        print("\n✗ 事件数据格式有问题，请检查并修正")

def train_model(config: dict):
    """训练模型（事件数据版本）"""
    try:
        # 1. 数据处理
        print("\n1. 数据处理（按事件加载）")
        from src.data_processor import StrainDataProcessor
        processor = StrainDataProcessor(config)
        datasets = processor.process_pipeline()
        
        # 验证数据形状
        for name, (X, y) in datasets.items():
            print(f"{name}集 - 样本数: {X.shape[0]}, 时间步: {X.shape[1]}, 特征数: {X.shape[2]}")
            print(f"{name}集标签形状: {y.shape}")
        
        # 2. 创建模型
        print("\n2. 创建模型")
        from src.models import PositionForceNet
        model = PositionForceNet(config)
        
        # 3. 训练
        print("\n3. 训练模型")
        from src.trainer import StrainForceTrainer
        trainer = StrainForceTrainer(config)
        trainer.setup_model(model)
        
        dataloaders = trainer.create_dataloaders(datasets)
        
        results = trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            test_loader=dataloaders['test']
        )
        
        # 4. 可视化
        print("\n4. 可视化结果")
        trainer.plot_training_history(save=True)
        
        # 5. 评估
        if results['test_results'] is not None:
            print("\n5. 测试集评估")
            test_results = results['test_results']
            
            # 导入评估器
            from src.predictor import StrainForcePredictor
            
            # 加载最佳模型进行评估
            model_path = f"models/best_model_{trainer.timestamp}.pth"
            predictor = StrainForcePredictor(model_path, args.config)
            
            # 评估
            eval_results = predictor.evaluate_on_test_set(
                datasets['test'][0], datasets['test'][1]
            )
            
            # 可视化预测结果
            visualize_event_predictions(eval_results, trainer.timestamp)
        
        print(f"\n训练完成! 最佳模型保存在: models/best_model_{trainer.timestamp}.pth")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

def predict_event_example(model_path: str, config: dict):
    """事件数据预测示例"""
    try:
        from src.predictor import StrainForcePredictor
        
        # 创建预测器
        predictor = StrainForcePredictor(model_path)
        
        # 生成示例事件数据
        print("\n生成示例事件数据进行预测...")
        
        # 创建一个完整事件的时间序列
        seq_length = config['data']['sequence_length']
        time = np.linspace(0, 1, seq_length)
        
        # 模拟四个应变点的时序数据（一个完整事件）
        strain_data = np.zeros((1, seq_length, 4))
        
        # 模拟位置和力大小
        force_x, force_y, force_mag = 50, 50, 200
        
        for t_idx, t in enumerate(time):
            # 简化的物理模型
            strain_data[0, t_idx, 0] = 0.1 * np.sin(2 * np.pi * t) + 0.05 * np.random.randn()
            strain_data[0, t_idx, 1] = 0.08 * np.cos(2 * np.pi * t) + 0.03 * np.random.randn()
            strain_data[0, t_idx, 2] = 0.12 * np.sin(2 * np.pi * t + np.pi/4) + 0.04 * np.random.randn()
            strain_data[0, t_idx, 3] = 0.09 * np.cos(2 * np.pi * t + np.pi/4) + 0.035 * np.random.randn()
        
        # 预测
        print(f"\n预测一个完整事件 ({seq_length} 个时间步)...")
        result = predictor.predict(strain_data)
        
        # 打印结果
        print("\n" + "="*50)
        print("事件预测结果")
        print("="*50)
        print(f"施力点位置: ({result['position_x']:.2f}, {result['position_y']:.2f})")
        print(f"力大小: {result['force_magnitude']:.2f}")
        
        # 保存结果
        predictor.save_predictions(result)
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

def visualize_event_predictions(eval_results: dict, timestamp: str):
    """可视化事件预测结果"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置样式
        sns.set_style("whitegrid")
        
        predictions = eval_results['predictions']
        true_values = eval_results['true_values']
        force_predictions = eval_results['force_predictions']
        force_true = eval_results['force_true']
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 位置散点图（事件位置）
        ax = axes[0, 0]
        ax.scatter(true_values[:, 0], true_values[:, 1], alpha=0.6, 
                  label='真实位置', s=40, color='blue')
        ax.scatter(predictions[:, 0], predictions[:, 1], alpha=0.6, 
                  label='预测位置', s=40, color='red')
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        ax.set_title('施力点位置预测（事件级别）', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 2. 力大小对比图（事件级别）
        ax = axes[0, 1]
        samples = range(min(50, len(force_true)))
        ax.plot(samples, force_true[samples], 'b-', label='真实力大小', 
                linewidth=2, marker='o', markersize=4)
        ax.plot(samples, force_predictions[samples], 'r--', label='预测力大小', 
                linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('事件索引', fontsize=12)
        ax.set_ylabel('力大小', fontsize=12)
        ax.set_title('力大小预测对比（事件级别）', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 3. 位置误差分布（事件级别）
        ax = axes[1, 0]
        position_errors = np.sqrt(np.sum((predictions - true_values)**2, axis=1))
        ax.hist(position_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('位置误差（欧氏距离）', fontsize=12)
        ax.set_ylabel('事件数', fontsize=12)
        ax.set_title('位置误差分布（事件级别）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. 力大小误差分布（事件级别）
        ax = axes[1, 1]
        force_errors = np.abs(force_predictions.flatten() - force_true)
        ax.hist(force_errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('力大小绝对误差', fontsize=12)
        ax.set_ylabel('事件数', fontsize=12)
        ax.set_title('力大小误差分布（事件级别）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图形
        save_path = f"results/figures/event_predictions_visualization_{timestamp}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"事件预测可视化图已保存: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")

if __name__ == "__main__":
    main()