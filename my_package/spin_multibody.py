# 构造自旋多体算符
import numpy as np

# 设置格点数
# N = 100

# 构造单位矩阵和泡利矩阵
I = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def kron_all(*args):
    """
    输入一个列表，列表视为一个元素存储在元组args的第一个元素中，
    所以要调取列表中的矩阵需要使用双重索引
    :param args: 输入后转化的元组
    :return: 返回一个矩阵
    """
    j = 1
    a = args[0][0]
    while j < len(args[0]):
        a = np.kron(a, args[0][j])
        j = j + 1
    return a


def one_site_kron(A, m, N):
    """
    构造单粒子相互作用算符
    :param A: 算符A
    :param m: 算符A的位置
    :param N: 格点数
    :return: 返回一个2^N维的矩阵
    """
    i = 0
    o = []
    while i < N:
        o.append(np.eye(2))
        i = i + 1
    o[m-1] = A
    o = kron_all(o)
    return o


def two_site_kron(A, m, B, n, N):
    """
    构造双粒子相互作用算符
    :param A: 算符A
    :param m: 算符A的位置
    :param B: 算符B
    :param n: 算符B的位置
    :param N: 格点数
    :return:返回一个2^N维的矩阵
    """
    i = 0
    o = []
    while i < N:
        o.append(np.eye(2))
        i = i + 1
    o[m - 1] = A
    o[n - 1] = B
    o = kron_all(o)
    return o

if __name__ == "__main__":
    print(one_site_kron(I, 2, 3))
    print(two_site_kron(I, 1, I, 2, 3))