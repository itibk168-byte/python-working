from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import my_package.spin_multibody as spin


# 模型参数
J = 1.0
gamma = 0.6
Gamma = 0.6
alpha = -0.5
h = 0.4

OUTPUT_DIR = Path(r"D:\python-code\python-working\XY-Gamma1\q_peak")

X, Y, Z = spin.sigma_x, spin.sigma_y, spin.sigma_z


def Hamiltonian(N, J, gamma, Gamma, alpha, h):
    """构造周期自旋链哈密顿量，返回基态能量和基态向量。"""
    dim = 1 << N
    H = sp.csr_matrix((dim, dim), dtype=complex)

    couplings = (
        (J * (1 + gamma) / 2, X, X),
        (J * (1 - gamma) / 2, Y, Y),
        (Gamma, X, Y),
        (Gamma * alpha, Y, X),
    )

    for j in range(1, N + 1):
        k = j % N + 1

        for coefficient, A, B in couplings:
            if j < k:
                operator = spin.two_site_kron(A, j, B, k, N)
            else:
                # 周期边界 N -> 1。调换格点顺序时，算符随格点调换。
                operator = spin.two_site_kron(B, k, A, j, N)

            H += coefficient * operator

        H += h * spin.one_site_kron(Z, j, N)

    energies, states = eigsh(H, k=1, which="SA")
    return energies[0], states[:, 0]


def q_max(N):
    """返回归一化 QFI 的峰位、理论参考值和有符号相对偏差。"""
    _, psi0 = Hamiltonian(N, J, gamma, Gamma, alpha, h)

    def expect(operator):
        return np.vdot(psi0, operator @ psi0).real

    def pair_expect(A, B, r):
        return expect(spin.two_site_kron(A, 1, B, 1 + r, N))

    m_z = expect(spin.one_site_kron(Z, 1, N))

    # 四列依次为 C_xx、C_yy、对称 C_xy 和连接 C_zz。
    C = np.empty((N, 4), dtype=float)
    C[0] = (1.0, 1.0, 0.0, 1.0 - m_z**2)

    for r in range(1, N):
        C[r] = (
            pair_expect(X, X, r),
            pair_expect(Y, Y, r),
            0.5 * (pair_expect(X, Y, r) + pair_expect(Y, X, r)),
            pair_expect(Z, Z, r) - m_z**2,
        )

    # 在 [0, pi] 上扫描余弦权重的波矢。
    q_grid = np.linspace(0.0, np.pi, 1000)
    sites = np.arange(1, N + 1)
    nQFI_q = np.empty_like(q_grid)

    for iq, q in enumerate(q_grid):
        weights = np.cos(q * sites)
        T = np.array([
            weights[:N - r] @ weights[r:]
            for r in range(N)
        ])
        T[1:] *= 2

        K_xx, K_yy, K_xy, K_zz = T @ C
        lambda_xy = 0.5 * (
            K_xx + K_yy + np.hypot(K_xx - K_yy, 2 * K_xy)
        )
        nQFI_q[iq] = max(lambda_xy, K_zz) / T[0]

    q_peak = q_grid[np.argmax(nQFI_q)]
    q_inf = np.pi - np.arctan(
        np.sqrt(-J**2 * gamma**2 - 4 * alpha * Gamma**2) / J
    )

    signed_pct = 100 * (q_peak - q_inf) / q_inf
    return q_peak, q_inf, signed_pct


def main():
    N_values = np.arange(8, 19, 2)
    results = []

    print(
        f"{'N':>3} {'q_peak/pi':>12} {'q_inf/pi':>12} "
        f"{'有符号偏差 (%)':>16} {'绝对偏差 (%)':>16}",
        flush=True,
    )

    for N in N_values:
        q_peak, q_inf, signed_pct = q_max(int(N))
        results.append((q_peak, q_inf, signed_pct))

        print(
            f"{N:3d} {q_peak / np.pi:12.8f} {q_inf / np.pi:12.8f} "
            f"{signed_pct:+16.6f} {abs(signed_pct):16.6f}",
            flush=True,
        )

    q_peaks, q_refs, signed_pcts = np.asarray(results).T
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

    axes[0].plot(
        N_values, q_peaks / np.pi, "o-",
        color="#00647D", label=r"$q_{\mathrm{peak}}/\pi$",
    )
    axes[0].axhline(
        q_refs[0] / np.pi, color="gray", linestyle="--",
        label=r"$q_\infty/\pi$",
    )
    axes[0].set(xlabel=r"$N$", ylabel=r"$q/\pi$", xticks=N_values)
    axes[0].legend()

    axes[1].plot(N_values, np.abs(signed_pcts), "o-", color="#A34830")
    axes[1].set(
        xlabel=r"$N$",
        ylabel="Absolute relative deviation (%)",
        xticks=N_values,
    )

    for ax in axes:
        ax.margins(y=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    # 创建输出目录，并在显示图形前保存 PNG。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "q_peak_vs_N.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"图片已保存至 {output_file}", flush=True)

    plt.show()


if __name__ == "__main__":
    main()