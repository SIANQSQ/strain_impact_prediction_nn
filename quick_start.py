#!/usr/bin/env python3
"""
快速开始脚本 - 支持事件数据格式
"""

import os
import sys
import numpy as np
# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设scripts在项目根目录下）
project_root = os.path.dirname(script_dir)
# 将src目录添加到sys.path
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)
# 添加src目录到路径
# sys.path.append('src')

def main():
    """主函数"""
    print("="*60)
    print("应变到力神经网络 - 快速开始（事件数据版本）")
    print("="*60)
    
    print("\n请选择操作:")
    print("1. 生成事件数据并训练")
    print("2. 加载已有事件数据训练")
    print("3. 使用预训练模型预测")
    print("4. 验证事件数据格式")
    print("5. 退出")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice == '1':
        generate_event_data_and_train()
    elif choice == '2':
        train_with_existing_event_data()
    elif choice == '3':
        predict_with_event_model()
    elif choice == '4':
        validate_event_data_format()
    elif choice == '5':
        print("再见!")
        sys.exit(0)
    else:
        print("无效选择")

def generate_event_data_and_train():
    """生成事件数据并训练"""
    print("\n生成事件数据...")
    
    from src.data_processor import StrainDataProcessor
    from src.utils import load_config
    
    # 加载配置
    config = load_config('configs/base_config.yaml')
    
    # 生成事件数据
    StrainDataProcessor.generate_event_data(
        n_events=1000,
        time_steps=config['data']['sequence_length'],
        save_path=config['data']['input_csv']
    )
    
    # 训练
    train_with_existing_event_data()

def train_with_existing_event_data():
    """使用已有事件数据训练"""
    print("\n加载配置...")
    
    from src.utils import load_config
    config = load_config('configs/base_config.yaml')
    
    print("\n数据处理...")
    from src.data_processor import StrainDataProcessor
    processor = StrainDataProcessor(config)
    
    try:
        datasets = processor.process_pipeline()
    except FileNotFoundError:
        print("\n数据文件未找到！")
        print("请先运行选项1生成数据，或将您的数据保存为:")
        print(config['data']['input_csv'])
        print("\n数据格式要求:")
        print("- 包含 'event_id' 列标识不同事件")
        print("- 每个事件有固定长度的时间序列")
        print("- 每个事件的所有时间步有相同的标签值")
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
    
    # 保存训练报告
    from src.utils import save_training_report
    save_training_report(config, results, trainer.timestamp)

def predict_with_event_model():
    """使用预训练模型预测事件数据"""
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
    predict_event_example(model_path=model_path)

def predict_event_example(model_path: str = None):
    """事件数据预测示例"""
    if model_path is None:
        print("未提供模型路径！")
        return
    
    print(f"\n使用模型进行预测: {model_path}")
    
    from src.predictor import StrainForcePredictor
    predictor = StrainForcePredictor(model_path)
    
    # 生成示例事件数据（一个完整事件的时间序列）
    from src.utils import load_config
    config = load_config('configs/base_config.yaml')
    seq_length = config['data']['sequence_length']
    
    print(f"\n示例1: 模拟一个完整冲击事件的 {seq_length} 个时间步")
    
    # 模拟一个完整事件的时间序列
    strain_data = generate_event_simulation(
        force_x=50, force_y=50, force_mag=200, 
        time_steps=seq_length
    )
    
    # 预测
    result = predictor.predict(strain_data)
    print_result("事件1", result)
    
    print(f"\n示例2: 模拟另一个冲击事件")
    strain_data2 = generate_event_simulation(
        force_x=80, force_y=80, force_mag=300,
        time_steps=seq_length
    )
    result2 = predictor.predict(strain_data2)
    print_result("事件2", result2)
    
    # 批量预测示例
    print(f"\n示例3: 批量预测多个事件")
    batch_data = []
    for i in range(5):
        fx = np.random.uniform(0, 100)
        fy = np.random.uniform(0, 100)
        fm = np.random.uniform(50, 500)
        event_data = generate_event_simulation(fx, fy, fm, seq_length)
        batch_data.append(event_data[0])  # 去掉批次维度
    
    batch_data = np.array(batch_data)  # [5, seq_length, 4]
    batch_results = predictor.predict(batch_data)
    
    if 'batch_results' in batch_results:
        for i, res in enumerate(batch_results['batch_results']):
            print(f"  事件{i}: 位置({res['position_x']:.1f}, {res['position_y']:.1f}), 力大小: {res['force_magnitude']:.1f}")
    
    # 保存预测结果
    import pandas as pd
    results_df = pd.DataFrame([
        {
            '事件': '示例1',
            '预测X': result['position_x'],
            '预测Y': result['position_y'],
            '预测力': result['force_magnitude']
        },
        {
            '事件': '示例2',
            '预测X': result2['position_x'],
            '预测Y': result2['position_y'],
            '预测力': result2['force_magnitude']
        }
    ])
    
    results_df.to_csv('results/event_predictions.csv', index=False)
    print(f"\n预测结果已保存: results/event_predictions.csv")

def generate_event_simulation(force_x: float, force_y: float, force_mag: float, 
                            time_steps: int = 100):
    """生成单个事件的应变模拟数据"""
    # 初始化应变数组
    strain_data = np.zeros((1, time_steps, 4))
    
    # 模拟四个应变点的响应
    for t in range(time_steps):
        time_step = t / time_steps
        
        # 简化的物理模型
        distance_top = np.sqrt((force_x - 50)**2 + (force_y - 100)**2)
        distance_bottom = np.sqrt((force_x - 50)**2 + force_y**2)
        distance_left = np.sqrt(force_x**2 + (force_y - 50)**2)
        distance_right = np.sqrt((force_x - 100)**2 + (force_y - 50)**2)
        
        # 计算应变
        strain_top = force_mag / (distance_top + 1) * np.exp(-distance_top/50)
        strain_bottom = force_mag / (distance_bottom + 1) * np.exp(-distance_bottom/50)
        strain_left = force_mag / (distance_left + 1) * np.exp(-distance_left/50)
        strain_right = force_mag / (distance_right + 1) * np.exp(-distance_right/50)
        
        # 添加时间变化和噪声
        time_factor = 1 + 0.1 * np.sin(2 * np.pi * time_step)
        noise_scale = 0.02
        
        strain_data[0, t, 0] = strain_top * time_factor + np.random.normal(0, strain_top * noise_scale)
        strain_data[0, t, 1] = strain_bottom * time_factor + np.random.normal(0, strain_bottom * noise_scale)
        strain_data[0, t, 2] = strain_left * time_factor + np.random.normal(0, strain_left * noise_scale)
        strain_data[0, t, 3] = strain_right * time_factor + np.random.normal(0, strain_right * noise_scale)
    
    return strain_data

def print_result(event_name: str, result: dict):
    """打印事件预测结果"""
    print(f"  {event_name} 预测:")
    print(f"    位置: ({result['position_x']:.1f}, {result['position_y']:.1f})")
    print(f"    力大小: {result['force_magnitude']:.1f}")

def validate_event_data_format():
    """验证事件数据格式"""
    print("\n验证事件数据格式...")
    
    from src.utils import load_config, validate_event_data
    import pandas as pd
    
    config = load_config('configs/base_config.yaml')
    data_path = config['data']['input_csv']
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        print("请先运行选项1生成示例数据，或准备您的事件数据")
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

if __name__ == "__main__":
    main()