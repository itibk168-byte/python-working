from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import my_package.spin_multibody as spin


# 模型参数
J = 1
gamma = 0.6
Gamma = 0.6
alpha = -0.5
h = 0.4

# 格点数和图片输出目录
N = 16
output_dir = Path(r"D:\python-code\python-working\XY-Gamma1\nQFI")


def Hamiltonian(J, gamma, Gamma, alpha, h):
    """构造周期链哈密顿量，返回基态能量和基态向量。"""
    H = sp.csr_matrix((1 << N, 1 << N), dtype=complex)

    for j in range(1, N):
        H += (
            J * (1 + gamma) / 2
            * spin.two_site_kron(spin.sigma_x, j, spin.sigma_x, j + 1, N)
            + J * (1 - gamma) / 2
            * spin.two_site_kron(spin.sigma_y, j, spin.sigma_y, j + 1, N)
            + Gamma
            * spin.two_site_kron(spin.sigma_x, j, spin.sigma_y, j + 1, N)
            + Gamma * alpha
            * spin.two_site_kron(spin.sigma_y, j, spin.sigma_x, j + 1, N)
            + h * spin.one_site_kron(spin.sigma_z, j, N)
        )

    # 周期边界键 N -> 1，以及第 N 个格点上的场项
    H += (
        J * (1 + gamma) / 2
        * spin.two_site_kron(spin.sigma_x, N, spin.sigma_x, 1, N)
        + J * (1 - gamma) / 2
        * spin.two_site_kron(spin.sigma_y, N, spin.sigma_y, 1, N)
        + Gamma
        * spin.two_site_kron(spin.sigma_x, N, spin.sigma_y, 1, N)
        + Gamma * alpha
        * spin.two_site_kron(spin.sigma_y, N, spin.sigma_x, 1, N)
        + h * spin.one_site_kron(spin.sigma_z, N, N)
    )

    energies, states = sp.linalg.eigsh(H, k=1, which="SA")
    return energies[0], states[:, 0]


E0, psi0 = Hamiltonian(J, gamma, Gamma, alpha, h)


def expect(operator):
    return np.vdot(psi0, operator @ psi0).real


def pair_expect(A, B, r):
    return expect(spin.two_site_kron(A, 1, B, 1 + r, N))


X, Y, Z = spin.sigma_x, spin.sigma_y, spin.sigma_z
m_z = expect(spin.one_site_kron(Z, 1, N))

# 四列依次为 C_xx、C_yy、对称 C_xy、连接 C_zz
C = np.empty((N, 4), dtype=float)
C[0] = (1.0, 1.0, 0.0, 1.0 - m_z**2)

for r in range(1, N):
    C[r] = (
        pair_expect(X, X, r),
        pair_expect(Y, Y, r),
        0.5 * (pair_expect(X, Y, r) + pair_expect(Y, X, r)),
        pair_expect(Z, Z, r) - m_z**2,
    )

q_grid = np.linspace(0, np.pi, 1000)
F_q = np.empty_like(q_grid)
nQFI_q = np.empty_like(q_grid)
sites = np.arange(1, N + 1)

for iq, q in enumerate(q_grid):
    weights = np.cos(q * sites)
    T = np.array([
        weights[:N - r] @ weights[r:] for r in range(N)
    ])
    T[1:] *= 2

    K_xx, K_yy, K_xy, K_zz = T @ C
    lambda_xy = 0.5 * (K_xx + K_yy + np.hypot(K_xx - K_yy, 2 * K_xy))
    F_q[iq] = max(lambda_xy, K_zz)
    nQFI_q[iq] = F_q[iq] / (weights @ weights)

i_best = np.argmax(F_q)
print("扫描结果：", "q/pi =", q_grid[i_best] / np.pi, "F_Q =", F_q[i_best])

q_infty = np.pi - np.arctan(
    np.sqrt(-J**2 * gamma**2 - 4 * alpha * Gamma**2) / J
)
print("理论结果：", "q_infty/pi =", q_infty / np.pi)

delta = (q_infty - q_grid[i_best]) / (np.pi * q_grid[i_best])
print("理论与扫描结果差距：", "delta =", delta, "粒子数为：", "N =", N)

q_over_pi = q_grid / np.pi
q_peak = q_over_pi[i_best]
F_peak = F_q[i_best]
nQFI_q_peak = nQFI_q[i_best]

fig, axes = plt.subplots(2, 1, figsize=(6, 8), layout="constrained")

# 第一张图：原始 QFI
ax = axes[0]
ax.plot(q_over_pi, F_q, color="#00647D", linewidth=2)
ax.axvline(q_peak, color="0.65", linestyle="--", linewidth=1)
ax.scatter(q_peak, F_peak, color="#A34830", s=35, zorder=3)

ax.annotate(
    f"$q/\\pi={q_peak:.4f}$\n$F_Q^{{\\mathrm{{opt}}}}={F_peak:.4f}$",
    xy=(q_peak, F_peak),
    xytext=(-14, 10),
    textcoords="offset points",
    ha="right",
    va="bottom",
)

ax.set(
    xlabel=r"$q/\pi$",
    ylabel=r"$F_Q^{\mathrm{opt}}(q)$",
    title=rf"$N={N},\ h={h:g},\ \alpha={alpha:g}$",
    xlim=(0, 1),
    xticks=[0, 0.25, 0.50, 0.75, 1],
)
ax.margins(y=0.18)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2)

# 第二张图：归一化 QFI，标出与上图相同的波矢
ax = axes[1]
ax.plot(q_over_pi, nQFI_q, color="#00647D", linewidth=2)
ax.axvline(q_peak, color="0.65", linestyle="--", linewidth=1)
ax.scatter(q_peak, nQFI_q_peak, color="#A34830", s=35, zorder=3)

ax.annotate(
    f"$q/\\pi={q_peak:.4f}$\n$nQFI_q^{{\\mathrm{{opt}}}}={nQFI_q_peak:.4f}$",
    xy=(q_peak, nQFI_q_peak),
    xytext=(-14, 10),
    textcoords="offset points",
    ha="right",
    va="bottom",
)

ax.set(
    xlabel=r"$q/\pi$",
    ylabel=r"$nQFI_q^{\mathrm{opt}}(q)$",
    title=rf"$N={N},\ h={h:g},\ \alpha={alpha:g}$",
    xlim=(0, 1),
    xticks=[0, 0.25, 0.50, 0.75, 1],
)
ax.margins(y=0.18)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2)

# 创建输出目录，在显示图形之前保存整张图
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"nQFI_vs_q_N{N}.png"
fig.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"图片已保存至 {output_file}")

plt.show()