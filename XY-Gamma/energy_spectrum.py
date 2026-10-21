import numpy as np
import matplotlib.pyplot as plt

# 能谱
def epsilon(k, gamma, Gamma, alpha, h, J=1, parity=None):
    """原来的五个位置参数不变，新增可选参数 J、parity。

    parity=None：连续色散图所用的通用分支。
    parity=1：偶扇区；parity=-1：奇扇区，单独处理 0、pi。
    """
    k = np.asarray(k, dtype=float)
    a2 = (J * gamma)**2 + Gamma**2 * (1 + alpha)**2
    symmetric_term = 2 * np.sqrt(a2 * np.sin(k)**2
                                 + (J * np.cos(k) - h)**2)
    asymmetric_term = 2 * Gamma * (1 - alpha) * np.sin(k)
    energy = symmetric_term + asymmetric_term

    if parity == -1:
        # 沿用论文 eta_0=c_0、eta_pi=c_pi 的单模约定。
        energy = np.where(np.isclose(k, 0, rtol=0, atol=1e-12),
                          2 * (J - h), energy)
        energy = np.where(np.isclose(np.abs(k), np.pi, rtol=0, atol=1e-12),
                          -2 * (J + h), energy)
    return energy


def N_(k, gamma, Gamma, alpha, h, J=1, parity=None):
    """同一宇称扇区中，负能模式的数目。"""
    energy = epsilon(k, gamma, Gamma, alpha, h, J=J, parity=parity)
    return int(np.count_nonzero(energy < 0))


def epsilon_min(k, gamma, Gamma, alpha, h, J=1, parity=None):
    """返回 min|epsilon|，结果非负。"""
    energy = epsilon(k, gamma, Gamma, alpha, h, J=J, parity=parity)
    return float(np.min(np.abs(energy)))




# 画图主程序
def plot():
    import matplotlib.pyplot as plt
    from pathlib import Path

    gamma, Gamma, alpha, h = 0.6, 0.6, -0.5, 0.9
    k = np.linspace(-np.pi, np.pi, 1001)
    energy = epsilon(k, gamma, Gamma, alpha, h)

    # 画图
    plt.plot(k, energy)
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    plt.fill_between(k, energy, 0, where=(energy < 0),
                     color="#D67B53", alpha=0.3)
    plt.xlabel("k")
    plt.ylabel(r"$\epsilon(k)$")

    # 保存图片
    plt.savefig(r"D:\python-code\python-working\XY-Gamma\picture\energy_spectrum.png", dpi=300,bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot()
