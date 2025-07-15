import numpy as np
from sklearn.feature_selection import mutual_info_regression
import torch


def calculate_mutual_information(X, Y, n_neighbors=3, n_repeats=10):
    """
    计算输入X和输出Y之间的平均互信息。

    参数:
    X -- 输入数据，大小为(samples, n_inputs)的二维数组
    Y -- 输出数据，大小为(samples, n_outputs)的二维数组
    n_neighbors -- 使用的邻居数量
    n_repeats -- 重复计算次数以获得平均值

    返回:
    mi_avg -- 平均互信息值的数组
    """
    if len(X) == 0 or len(Y) == 0:
        # 注意这里返回二维数组
        n_inputs = X.shape[1]
        return np.zeros((1, n_inputs))

    if len(X) != len(Y):
        min_len = min(len(X), len(Y))
        X = X[:min_len]
        Y = Y[:min_len]

    n_samples, n_inputs = X.shape
    n_outputs = Y.shape[1] if Y.ndim > 1 else 1
    mi_avg = np.zeros((n_outputs, n_inputs))

    # 确保Y是二维数组
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # 使用较小的neighbors数量如果样本数量较少
    actual_n_neighbors = min(n_neighbors, len(X) - 1)

    try:
        for repeat in range(n_repeats):
            for output_index in range(n_outputs):
                mi = mutual_info_regression(X, Y[:, output_index],
                                            n_neighbors=actual_n_neighbors, random_state=42)
                mi_avg[output_index] += mi

        mi_avg /= n_repeats
    except Exception as e:
        print(f"Error in MI calculation: {e}")
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        # 返回形状正确的二维数组，避免后续索引错误
        return np.zeros((n_outputs, n_inputs))

    if n_outputs == 1:
        # 保证返回二维数组（1, n_inputs）
        return mi_avg.reshape(1, -1)
    return mi_avg

def cal_score(mi_results, index, weights):
    return sum(mi_results[i][index] * weight for i, weight in enumerate(weights))


# 计算mi分数
def calculate_scores(dbx, dby, gain_num, GBW_num, phase_num, iter, init_num, n_neighbors=3, n_repeats=10,
                     input_dim=12):
    """
    公式5: Iobj(xi) = I(xi; y0) - 目标相关MI
    公式6: Icon(xi) = Σ ωj · I(xi; yj) - 约束相关MI
    公式7: ωj = e^(-Nj/M) - 约束权重
    公式8: S(xi) = Iobj(xi) + Icon(xi) - 最终分数
    """
    if isinstance(dbx, torch.Tensor):
        dbx = dbx.numpy()
    if isinstance(dby, torch.Tensor):
        dby = dby.numpy()

    # 确保dby是2D数组
    if len(dby.shape) == 1:
        dby = dby.reshape(-1, 1)
    elif len(dby.shape) == 3:
        dby = dby.reshape(len(dby), -1)

    # 调用函数计算互信息 - 返回 (n_outputs, n_inputs) 的矩阵
    mi_results = calculate_mutual_information(dbx, dby, n_neighbors=n_neighbors, n_repeats=n_repeats)

    # 当前迭代总数 M
    M = iter + init_num + 1

    # 初始化分数数组
    scores_list = []

    # 对每个输入维度计算CawMI分数
    for i in range(input_dim):
        # 公式5: 目标相关MI - I_obj(xi) = I(xi; y0)
        # y0是电流目标 (索引1)
        I_obj = mi_results[1, i]  # 电流是主要优化目标

        # 公式6和7: 约束相关MI - I_con(xi) = Σ ωj · I(xi; yj)
        I_con = 0.0

        # 约束1: Gain (索引0)
        omega_gain = np.exp(-gain_num / M)  # 公式7
        I_con += omega_gain * mi_results[0, i]

        # 约束2: Phase (索引2)
        omega_phase = np.exp(-phase_num / M)  # 公式7
        I_con += omega_phase * mi_results[2, i]

        # 约束3: GBW (索引3)
        omega_gbw = np.exp(-GBW_num / M)  # 公式7
        I_con += omega_gbw * mi_results[3, i]

        # 公式8: 最终分数 S(xi) = I_obj(xi) + I_con(xi)
        S_xi = I_obj + I_con
        scores_list.append(S_xi)

    scores = torch.tensor(scores_list)

    # 打印调试信息
    print(f"CawMI分析结果 (迭代 {iter}):")
    print(
        f"  约束权重: gain={np.exp(-gain_num / M):.3f}, phase={np.exp(-phase_num / M):.3f}, gbw={np.exp(-GBW_num / M):.3f}")
    for i, mi in enumerate(mi_results):
        print(f"  与输出 {i + 1} 的互信息: {mi}")
    print(f"  最终CawMI分数: {scores[:5].numpy()}")  # 显示前5个维度

    return scores
