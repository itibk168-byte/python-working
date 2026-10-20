from matplotlib import pyplot as plt
import my_package.spin_multibody as sp
import numpy as np
import energy_spectrum as esp


def energy_ground_state_num(J, gamma, Gamma, alpha, h, N):
    """
    输入参数以及格点数，返回体系的基态能量（数值解）
    :param J:
    :param gamma:
    :param Gamma:
    :param alpha:
    :param h: 磁场强度
    :param N: 格点数
    :return: 体系的基态能量（数值解）
    """
    j = 1
    H = 0
    while j < N:
        H_xy_1 = J * ((1 + gamma) / 2 * sp.two_site_kron(sp.sigma_x, j, sp.sigma_x, j + 1, N))
        H_Gamma_1 = Gamma * (sp.two_site_kron(sp.sigma_x, j, sp.sigma_y, j + 1, N) + alpha * sp.two_site_kron(sp.sigma_y, j, sp.sigma_x, j + 1, N))
        H_h_1 = h * sp.one_site_kron(sp.sigma_z, j, N)
        H = H + H_xy_1 + H_Gamma_1 + H_h_1
        j += 1

    # 边界项
    H_xy_2 = J * ((1 + gamma) / 2 * sp.two_site_kron(sp.sigma_x, N, sp.sigma_x, 1, N))
    H_Gamma_2 = Gamma * (sp.two_site_kron(sp.sigma_x, N, sp.sigma_y, 1, N) + alpha * sp.two_site_kron(sp.sigma_y, N, sp.sigma_x,  1, N))
    H_h_2 = h * sp.one_site_kron(sp.sigma_z, N, N)

    # 总哈密顿量
    H = H + H_xy_2 + H_Gamma_2 + H_h_2

    # 本征值和本征态
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # 基态能量
    energy = eigenvalues[0]
    return energy

def energy_ground_state_ana(J, gamma, Gamma, alpha, h, N):
    """
    输入参数以及格点数，返回体系的基态能量（解析解）
    :param J:
    :param gamma:
    :param Gamma:
    :param alpha:
    :param h:
    :param N:
    :return: 输入参数以及格点数，返回体系的基态能量（解析解）
    """
    # 遍历动量取值
    k = np.pi / N * np.arange(-(N - 1), N)
    for x in k:
        energy = -1 / 2 * np.abs(esp.epsilon(x,gamma,Gamma,alpha,h))
    return energy


if __name__ == "__main__":
    # 计算随着参数h改变而改变的基态能量
    h = np.linspace(-2,2,100)
    energy_num = list()
    energy_ana = list()
    for x in h:
        energy_num.append(energy_ground_state_num(1, 0.6, 0.6, -0.3, x, 8))
        energy_ana.append(energy_ground_state_num(1, 0.6, 0.6, -0.3, x, 8))

    # 画图
    plt.plot(h, energy_num)
    plt.plot(h, energy_ana,"bo",markerfacecolor="none",markersize=5,markevery=4)
    plt.grid(True,linestyle='--',alpha = 0.5)
    plt.xlabel("h")
    plt.ylabel("$E_0$")
    plt.title(r"$J=1,\Gamma=0.6,\gamma=0.6,\alpha=-0.3$")

    # 图片保存
    plt.savefig(r"D:\python-code\python-working\XY-Gamma\picture\energy_ground_state.png", dpi=300,
                bbox_inches="tight")
    plt.show()

