import os
import torch
import numpy as np
from src.DIVE.main import DynamicDimBO
from Simulation.simulation.ota_two.ota_two import OTA_two_simulation_all
from src.DIVE.utils import seed_set, get_bounds_from_config, load_config


def main(seed):
    """
    优化二级放大器（180nm工艺）

    参数:
    seed: 随机种子
    """
    # 设置结果保存路径
    base_dir = "./result"
    os.makedirs(base_dir, exist_ok=True)

    # 设置文件名
    prefix = "DIVE_OTA_two"
    file_path = os.path.join(base_dir, f'{prefix}_seed_{seed}.csv')

    # Load configuration
    experiment = 'experiment1'
    config = load_config("../config.yaml")
    bounds = get_bounds_from_config(config, experiment)
    constraints = {
        'gain': float(config[experiment]['constraints']['gain']),
        'phase': float(config[experiment]['constraints']['phase']),
        'gbw': float(config[experiment]['constraints']['gbw'])
    }

    # 使用180nm工艺仿真函数
    objective_function = OTA_two_simulation_all

    # 设置优化器参数（去除初始可行点）
    bo_params = {
        'total_dim': len(bounds),
        'objective_function': objective_function,
        'bounds': bounds,
        'constraints': constraints,
        'seed': seed,
        'initial_samples': 20
    }

    print("开始优化二级运算放大器...")
    print(f"优化参数:")
    print(f"  总维度数: {len(bounds)}")
    print(f"  初始样本数: 20")
    print(
        f"  约束条件: gain > {constraints['gain']}dB, phase > {constraints['phase']}°, gbw > {np.exp(constraints['gbw']):.0e}Hz")

    # 初始化优化器
    bo = DynamicDimBO(**bo_params)

    # 运行优化
    bo.optimize(n_iter=380)

    # 输出最终结果
    print(f"\n最终优化结果:")
    if hasattr(bo, 'best_feasible_current') and bo.best_feasible_point is not None:
        print(f"找到的最佳可行电流: {np.exp(bo.best_feasible_current):.2e} A")
        print(f"最佳参数点:")
        for i, val in enumerate(bo.best_feasible_point):
            print(f"  参数 {i}: {val:.6e}")
    else:
        print("未找到可行解")

    print(f"最终活跃维度: {bo.active_dims}")
    print(f"维度重要性分数: {np.round(bo.dim_scores[bo.active_dims], 3)}")


if __name__ == "__main__":
    # 单次运行
    SEED = 1
    seed_set(seed=SEED)
    main(seed=SEED)
