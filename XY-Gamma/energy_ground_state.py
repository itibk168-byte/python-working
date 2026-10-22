from matplotlib import pyplot as plt
import numpy as np
import my_package.spin_multibody as sp
import energy_spectrum as esp


def Hamiltonian(J, gamma, Gamma, alpha, h, N):
    j = 1
    H = 0
    while j < N:
        H_xy_1 = J * (
            (1 + gamma) / 2 * sp.two_site_kron(sp.sigma_x, j, sp.sigma_x, j + 1, N)
            + (1 - gamma) / 2 * sp.two_site_kron(sp.sigma_y, j, sp.sigma_y, j + 1, N)
        )
        H_Gamma_1 = Gamma * (
            sp.two_site_kron(sp.sigma_x, j, sp.sigma_y, j + 1, N)
            + alpha * sp.two_site_kron(sp.sigma_y, j, sp.sigma_x, j + 1, N)
        )
        H_h_1 = h * sp.one_site_kron(sp.sigma_z, j, N)
        H = H + H_xy_1 + H_Gamma_1 + H_h_1
        j += 1

    # 修改：边界项同样必须包含 yy 相互作用。
    H_xy_2 = J * (
        (1 + gamma) / 2 * sp.two_site_kron(sp.sigma_x, N, sp.sigma_x, 1, N)
        + (1 - gamma) / 2 * sp.two_site_kron(sp.sigma_y, N, sp.sigma_y, 1, N)
    )
    H_Gamma_2 = Gamma * (
        sp.two_site_kron(sp.sigma_x, N, sp.sigma_y, 1, N)
        + alpha * sp.two_site_kron(sp.sigma_y, N, sp.sigma_x, 1, N)
    )
    H_h_2 = h * sp.one_site_kron(sp.sigma_z, N, N)
    return H + H_xy_2 + H_Gamma_2 + H_h_2

def energy_ground_state_num(J, gamma, Gamma, alpha, h, N):
    H = Hamiltonian(J, gamma, Gamma, alpha, h, N)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return float(eigenvalues[0])

def ground_state_sector(J, gamma, Gamma, alpha, h, N, parity):
    """新增共用函数：返回该扇区的 (基态能量, 全动量, 占据数)。

    parity=1：偶宇称/APBC；parity=-1：奇宇称/PBC。
    先填负能模式，再在必要时只翻转一个代价最小的模式。
    """
    if N < 4 or N % 2 != 0:
        raise ValueError("此实现要求 N 为不小于 4 的偶数。")
    if parity == 1:
        k = np.pi / N * np.arange(-(N - 1), N, 2)
    elif parity == -1:
        k = np.pi / N * np.arange(-(N - 2), N + 1, 2)
    else:
        raise ValueError("parity 必须为 1 或 -1。")

    energy_values = esp.epsilon(k, gamma, Gamma, alpha, h,
                                J=J, parity=parity)
    occupation = (energy_values < 0).astype(int)

    energy = 0.0
    for value in energy_values:
        energy += -0.5 * abs(value)  # 修改：累加，而不是覆盖。

    N_minus = int(occupation.sum())
    correction = (1 - parity * (-1)**N_minus) // 2
    if correction:
        index = int(np.argmin(np.abs(energy_values)))
        energy += abs(energy_values[index])
        occupation[index] = 1 - occupation[index]

    return float(energy), k, occupation

def energy_ground_state_ana(J, gamma, Gamma, alpha, h, N):
    energy_even, _, _ = ground_state_sector(J, gamma, Gamma, alpha, h, N, 1)
    energy_odd, _, _ = ground_state_sector(J, gamma, Gamma, alpha, h, N, -1)
    return min(energy_even, energy_odd)
pyt



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    h = np.linspace(-2, 2, 100)
    energy_num, energy_ana = [], []
    for x in h:
        energy_num.append(energy_ground_state_num(1, 0.6, 0.6, -0.3, x, 8))
        energy_ana.append(energy_ground_state_ana(1, 0.6, 0.6, -0.3, x, 8))
    print("max energy error:", np.max(np.abs(np.array(energy_num) - energy_ana)))

    # 画图
    plt.plot(h, energy_num, label="Numerical")
    plt.plot(h, energy_ana, "bo", markerfacecolor="none", markersize=5,
             markevery=4, label="Analytic")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("h")
    plt.ylabel(r"$E_0$")
    plt.title(r"$J=1,\Gamma=0.6,\gamma=0.6,\alpha=-0.3$")
    plt.legend()

    # 图片保存
    plt.savefig(r"D:\python-code\python-working\XY-Gamma\picture\energy_ground_state.png", dpi=300,
    bbox_inches="tight")
    plt.show()

