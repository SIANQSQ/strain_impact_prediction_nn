#!/usr/bin/env python3
"""
快速开始脚本 - 最简使用方式
"""

import os
import sys
import numpy as np

# 添加src目录到路径
sys.path.append('src')

def main():
    """主函数"""
    print("="*60)
    print("应变到力神经网络 - 快速开始")
    print("="*60)
    
    print("\n请选择操作:")
    print("1. 生成示例数据并训练")
    print("2. 加载已有数据训练")
    print("3. 使用预训练模型预测")
    print("4. 退出")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == '1':
        generate_and_train()
    elif choice == '2':
        train_with_existing_data()
    elif choice == '3':
        predict_with_model()
    elif choice == '4':
        print("再见!")
        sys.exit(0)
    else:
        print("无效选择")

def generate_and_train():
    """生成数据并训练"""
    print("\n生成合成数据...")
    
    from src.data_processor import StrainDataProcessor
    StrainDataProcessor.generate_synthetic_data(
        n_samples=5000,
        save_path='data/raw/abaqus_strain_data.csv'
    )
    
    # 训练
    train_with_existing_data()

def train_with_existing_data():
    """使用现有数据训练"""
    print("\n加载配置...")
    
    import yaml
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改配置为快速训练
    config['training']['epochs'] = 50
    config['training']['batch_size'] = 16
    config['data']['sequence_length'] = 10
    
    print("\n数据处理...")
    from src.data_processor import StrainDataProcessor
    processor = StrainDataProcessor(config)
    
    try:
        datasets = processor.process_pipeline()
    except FileNotFoundError:
        print("\n数据文件未找到！")
        print("请先运行选项1生成数据，或将您的数据保存为:")
        print("data/raw/abaqus_strain_data.csv")
        return
    
    print("\n创建模型...")
    from src.models import PositionForceNet
    model = PositionForceNet(config)
    
    print("\n开始训练...")
    from src.trainer import StrainForceTrainer
    trainer = StrainForceTrainer(config)
    trainer.setup_model(model)
    
    dataloaders = trainer.create_dataloaders(datasets)
    
    results = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test']
    )
    
    print("\n训练完成！")
    print(f"最佳模型保存在: models/best_model_{trainer.timestamp}.pth")
    
    # 简单预测示例
    predict_example(trainer.timestamp)

def predict_with_model():
    """使用预训练模型预测"""
    print("\n查找可用模型...")
    
    model_files = [f for f in os.listdir('models') if f.startswith('best_model')]
    
    if not model_files:
        print("未找到训练好的模型！")
        print("请先训练模型。")
        return
    
    # 选择模型
    print("\n可用模型:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    try:
        choice = int(input("\n请选择模型 (输入序号): ")) - 1
        model_path = f"models/{model_files[choice]}"
    except:
        # 使用最新模型
        latest_model = max(model_files)
        model_path = f"models/{latest_model}"
        print(f"使用最新模型: {latest_model}")
    
    # 进行预测
    predict_example(model_path=model_path)

def predict_example(timestamp: str = None, model_path: str = None):
    """示例预测"""
    if model_path is None and timestamp is not None:
        model_path = f"models/best_model_{timestamp}.pth"
    elif model_path is None:
        print("未提供模型路径！")
        return
    
    print(f"\n使用模型进行预测: {model_path}")
    
    from src.predictor import StrainForcePredictor
    predictor = StrainForcePredictor(model_path)
    
    # 生成示例应变数据
    seq_length = 20
    time = np.linspace(0, 1, seq_length)
    
    # 模拟不同位置和大小的力
    print("\n示例1: 中心位置中等大小的力")
    strain_data1 = generate_strain_simulation(force_x=50, force_y=50, force_mag=200, time=time)
    result1 = predictor.predict(strain_data1)
    print_result("中心", result1)
    
    print("\n示例2: 右上角较大力的力")
    strain_data2 = generate_strain_simulation(force_x=80, force_y=80, force_mag=300, time=time)
    result2 = predictor.predict(strain_data2)
    print_result("右上", result2)
    
    print("\n示例3: 左下角较小的力")
    strain_data3 = generate_strain_simulation(force_x=20, force_y=20, force_mag=100, time=time)
    result3 = predictor.predict(strain_data3)
    print_result("左下", result3)
    
    # 保存预测结果
    import pandas as pd
    results_df = pd.DataFrame([
        {
            '示例': '中心位置',
            '预测X': result1['position_x'],
            '预测Y': result1['position_y'],
            '预测力': result1['force_magnitude']
        },
        {
            '示例': '右上位置',
            '预测X': result2['position_x'],
            '预测Y': result2['position_y'],
            '预测力': result2['force_magnitude']
        },
        {
            '示例': '左下位置',
            '预测X': result3['position_x'],
            '预测Y': result3['position_y'],
            '预测力': result3['force_magnitude']
        }
    ])
    
    results_df.to_csv('results/quick_predictions.csv', index=False)
    print(f"\n预测结果已保存: results/quick_predictions.csv")

def generate_strain_simulation(force_x: float, force_y: float, force_mag: float, time: np.ndarray):
    """生成应变模拟数据"""
    seq_length = len(time)
    
    # 初始化应变数组
    strain_data = np.zeros((1, seq_length, 4))
    
    # 模拟四个应变点的响应
    for t_idx, t in enumerate(time):
        # 简化的物理模型
        distance_top = np.sqrt((force_x - 50)**2 + (force_y - 100)**2)
        distance_bottom = np.sqrt((force_x - 50)**2 + (force_y - 0)**2)
        distance_left = np.sqrt((force_x - 0)**2 + (force_y - 50)**2)
        distance_right = np.sqrt((force_x - 100)**2 + (force_y - 50)**2)
        
        # 计算应变
        strain_top = force_mag / (distance_top + 1) * np.exp(-distance_top/50)
        strain_bottom = force_mag / (distance_bottom + 1) * np.exp(-distance_bottom/50)
        strain_left = force_mag / (distance_left + 1) * np.exp(-distance_left/50)
        strain_right = force_mag / (distance_right + 1) * np.exp(-distance_right/50)
        
        # 添加时间变化和噪声
        time_factor = 1 + 0.1 * np.sin(2 * np.pi * t)
        noise_scale = 0.02
        
        strain_data[0, t_idx, 0] = strain_top * time_factor + np.random.normal(0, strain_top * noise_scale)
        strain_data[0, t_idx, 1] = strain_bottom * time_factor + np.random.normal(0, strain_bottom * noise_scale)
        strain_data[0, t_idx, 2] = strain_left * time_factor + np.random.normal(0, strain_left * noise_scale)
        strain_data[0, t_idx, 3] = strain_right * time_factor + np.random.normal(0, strain_right * noise_scale)
    
    return strain_data

def print_result(location: str, result: dict):
    """打印预测结果"""
    print(f"  预测位置: ({result['position_x']:.1f}, {result['position_y']:.1f})")
    print(f"  预测力大小: {result['force_magnitude']:.1f}")

if __name__ == "__main__":
    main()