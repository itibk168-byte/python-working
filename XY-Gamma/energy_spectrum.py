import numpy as np
import matplotlib.pyplot as plt

# 能谱
def epsilon(k, gamma, Gamma, alpha, h):
    symmetric_term = 2 * np.sqrt((Gamma ** 2 * (1 + alpha) ** 2 + gamma ** 2) * (np.sin(k)) ** 2 + (np.cos(k) - h) ** 2)
    asymmetric_term = 2 * Gamma * (1 - alpha) * np.sin(k)
    return symmetric_term + asymmetric_term



# 画图主程序
def plot():
    # 参数设置
    gamma = 0.6
    Gamma = 0.6
    alpha = -0.5
    h = 0.9

    # 动量设置
    k = np.linspace(-np.pi,np.pi,1001)

    # 计算能谱
    energy = epsilon(k, gamma, Gamma, alpha, h)

    # 画图
    plt.plot(k,energy)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.fill_between(k, energy, 0, where=(energy < 0),color='#D67B53', alpha=0.3)
    plt.xlabel("k")
    plt.ylabel(r"$\epsilon(k)$")

    # 图片保存
    plt.savefig(r"D:\python-code\python-working\XY-Gamma\picture\energy_spectrum.png", dpi=300,
    bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot()
