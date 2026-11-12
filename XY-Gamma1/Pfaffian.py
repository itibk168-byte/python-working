import numpy as np

def pfaffian(matrix):
    """
    输入一个偶数阶的反对称的方阵，返回他的Pfaffian值
    :param matrix: 偶数阶，反对称 M^T = -M，方阵，矩阵元素有限
    :return:
    """

    # 检测输入矩阵是实矩阵还是复矩阵
    if np.iscomplexobj(matrix):
        dtype = complex
    else:
        dtype = float

    M = np.array(matrix, dtype=dtype, copy=True)

    # 检测输入矩阵是否规范
    if M.ndim != 2:
        raise ValueError("输入矩阵必须是二维的")
    if M.shape[0] != M.shape[1]:
        raise ValueError("输入矩阵必须是方阵")
    if not np.allclose(M.T, -M, atol=1e-14, rtol=1e-12):
        raise ValueError("输入矩阵必须是反对称的")
    if not np.isfinite(M).all():
        raise ValueError("输入矩阵必须是有限的")
    if M.shape[0] % 2 != 0:
        raise ValueError("输入矩阵必须是偶数阶的")

    # 确保矩阵M是完全反对称的，或者说去掉输入中容差范围内的反对称误差
    M = 0.5 * (M - M.T)

    # 计算矩阵M的Pfaffian，使用主元相乘的算法，对于一个2^N阶的矩阵，需要把2^(N-1)个主元乘起来
    n = M.shape[0]                  # 矩阵M的阶数
    pf = dtype(1)                   # pf值初始化

    for i in range(0,n,2):                          # n/2次循环，每次乘上一个主元，i表示第 i 行
        j = i + 1 + np.argmax(M[i,i + 1:])          # 找当前行绝对值最大的候选主元，j表示第 j 列
        if M[i,j] == 0:
            return dtype(0)                         # 如果有一个主元为0，那么矩阵M的Pfaffian值为0

        if j != i + 1:                              # 如果主元不在 i + 1 的位置，那么需要把第 i + 1， j 行列都交换
            M[[i + 1, j], :] = M[[j, i + 1], :]     # 交换对应的两行
            M[:, [i + 1, j]] = M[:, [j, i + 1]]     # 同步交换对应的两列
            pf = -pf                                # 一次指标交换使 Pfaffian 变号

        p = M[i, i + 1]                             # p是当前主元
        pf *= p                                     # 乘上主元

        # 更新余块
        u = M[i, i + 2:]                            # 取第一行与余块的耦合
        v = M[i + 1, i + 2:]                        # 取第二行与余块的耦合
        M[i + 2:, i + 2:] += (np.outer(v, u) - np.outer(u, v)) / p      # 更新余块 C
    return pf

if __name__ == "__main__":
    M = np.array([[0, 1, 2, 3], [-1, 0, 4, 5], [-2, -4, 0, 6],[-3, -5, -6, 0]])
    print(pfaffian(M))
