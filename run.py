#!/usr/bin/env python3
"""
应变到力神经网络 - 一键运行脚本
四点应变时序数据 -> 施力点坐标 + 力大小
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
    parser = argparse.ArgumentParser(description='应变到力神经网络训练和预测')
    
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'predict', 'generate', 'full'],
                       help='运行模式: train(训练), predict(预测), generate(生成数据), full(完整流程)')
    
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='配置文件路径')
    
    parser.add_argument('--data', type=str, default=None,
                       help='数据文件路径（如果不使用配置文件中的路径）')
    
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件路径（预测模式使用）')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置）')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批大小（覆盖配置）')
    
    parser.add_argument('--seq_length', type=int, default=None,
                       help='序列长度（覆盖配置）')
    
    parser.add_argument('--generate_samples', type=int, default=10000,
                       help='生成合成数据的样本数')
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("应变到力神经网络系统")
    print("="*60)
    
    # 加载配置
    print(f"加载配置: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖配置参数
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    if args.seq_length is not None:
        config['data']['sequence_length'] = args.seq_length
    
    if args.data is not None:
        config['data']['input_csv'] = args.data
    
    # 设置输出目录
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        config['experiment']['name'] = args.output_dir
    
    # 根据模式运行
    if args.mode == 'generate' or args.mode == 'full':
        print("\n生成合成数据...")
        from src.data_processor import StrainDataProcessor
        StrainDataProcessor.generate_synthetic_data(
            n_samples=args.generate_samples,
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
        
        predict_example(model_path, config)
    
    print("\n" + "="*60)
    print("运行完成!")
    print("="*60)

def train_model(config: dict):
    """训练模型"""
    try:
        # 1. 数据处理
        print("\n1. 数据处理")
        from src.data_processor import StrainDataProcessor
        processor = StrainDataProcessor(config)
        datasets = processor.process_pipeline()
        
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
            visualize_predictions(eval_results, trainer.timestamp)
        
        print(f"\n训练完成! 最佳模型保存在: models/best_model_{trainer.timestamp}.pth")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

def predict_example(model_path: str, config: dict):
    """示例预测"""
    try:
        from src.predictor import StrainForcePredictor
        
        # 创建预测器
        predictor = StrainForcePredictor(model_path)
        
        # 生成示例数据
        print("\n生成示例数据进行预测...")
        
        # 创建模拟的应变数据
        seq_length = config['data'].get('sequence_length', 20)
        time = np.linspace(0, 1, seq_length)
        
        # 模拟四个应变点的时序数据
        strain_top = 0.1 * np.sin(2 * np.pi * time) + 0.05 * np.random.randn(seq_length)
        strain_bottom = 0.08 * np.cos(2 * np.pi * time) + 0.03 * np.random.randn(seq_length)
        strain_left = 0.12 * np.sin(2 * np.pi * time + np.pi/4) + 0.04 * np.random.randn(seq_length)
        strain_right = 0.09 * np.cos(2 * np.pi * time + np.pi/4) + 0.035 * np.random.randn(seq_length)
        
        # 预测
        print("\n进行预测...")
        result = predictor.predict_from_raw(
            strain_top=strain_top.tolist(),
            strain_bottom=strain_bottom.tolist(),
            strain_left=strain_left.tolist(),
            strain_right=strain_right.tolist(),
            sequence_length=seq_length
        )
        
        # 打印结果
        print("\n" + "="*50)
        print("预测结果")
        print("="*50)
        print(f"施力点位置: ({result['position_x']:.2f}, {result['position_y']:.2f})")
        print(f"力大小: {result['force_magnitude']:.2f}")
        
        # 保存结果
        predictor.save_predictions(result)
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

def visualize_predictions(eval_results: dict, timestamp: str):
    """可视化预测结果"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
        plt.rcParams['axes.unicode_minus'] = False    # 用于显示负号
        
        predictions = eval_results['predictions']
        true_values = eval_results['true_values']
        force_predictions = eval_results['force_predictions']
        force_true = eval_results['force_true']
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 位置散点图
        ax = axes[0, 0]
        ax.scatter(true_values[:, 0], true_values[:, 1], alpha=0.6, 
                  label='真实位置', s=40, color='blue')
        ax.scatter(predictions[:, 0], predictions[:, 1], alpha=0.6, 
                  label='预测位置', s=40, color='red')
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        ax.set_title('施力点位置预测', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 添加误差线
        for i in range(min(20, len(true_values))):
            ax.plot([true_values[i, 0], predictions[i, 0]], 
                   [true_values[i, 1], predictions[i, 1]], 
                   'k-', alpha=0.3, linewidth=0.5)
        
        # 2. 力大小对比图
        ax = axes[0, 1]
        samples = range(min(50, len(force_true)))
        ax.plot(samples, force_true[samples], 'b-', label='真实力大小', 
                linewidth=2, marker='o', markersize=4)
        ax.plot(samples, force_predictions[samples], 'r--', label='预测力大小', 
                linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('样本索引', fontsize=12)
        ax.set_ylabel('力大小', fontsize=12)
        ax.set_title('力大小预测对比', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 3. 位置误差分布
        ax = axes[1, 0]
        position_errors = np.sqrt(np.sum((predictions - true_values)**2, axis=1))
        ax.hist(position_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('位置误差（欧氏距离）', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('位置误差分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = np.mean(position_errors)
        median_error = np.median(position_errors)
        ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                  label=f'均值: {mean_error:.2f}')
        ax.axvline(median_error, color='orange', linestyle='--', linewidth=2,
                  label=f'中位数: {median_error:.2f}')
        ax.legend(fontsize=11)
        
        # 4. 力大小误差分布
        ax = axes[1, 1]
        force_errors = np.abs(force_predictions.flatten() - force_true)
        ax.hist(force_errors, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('力大小绝对误差', fontsize=12)
        ax.set_ylabel('频数', fontsize=12)
        ax.set_title('力大小误差分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_force_error = np.mean(force_errors)
        median_force_error = np.median(force_errors)
        ax.axvline(mean_force_error, color='red', linestyle='--', linewidth=2,
                  label=f'均值: {mean_force_error:.2f}')
        ax.axvline(median_force_error, color='orange', linestyle='--', linewidth=2,
                  label=f'中位数: {median_force_error:.2f}')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # 保存图形
        save_path = f"results/figures/predictions_visualization_{timestamp}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"预测可视化图已保存: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")

if __name__ == "__main__":
    main()