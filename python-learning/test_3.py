import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import schur


# 模型参数
J = 1.0
gamma = 0.6
Gamma = 0.6
alpha = -0.5
h = 0.4

# 扫描范围。步长为 2，只计算偶数 N。
N_MIN = 8
N_MAX = 128
N_STEP = 2
Q_POINTS = 1000


def majorana_matrix(N, parity):
    """构造 H = (i/4) w.T @ A @ w 中的实反对称矩阵 A。"""
    A = np.zeros((2 * N, 2 * N))

    def add_pair(u, v, value):
        A[u, v] += value
        A[v, u] -= value

    for j in range(N):
        k = (j + 1) % N

        # 自旋周期边界映射为 w[j+N] = -parity * w[j]。
        boundary = -parity if k == 0 else 1

        # w = (a_0, b_0, a_1, b_1, ...)。
        a, b = 2 * j, 2 * j + 1
        c, d = 2 * k, 2 * k + 1

        add_pair(a, b, -2 * h)                          # Z_j
        add_pair(b, c, -boundary * J * (1 + gamma))     # X_j X_k
        add_pair(a, d, boundary * J * (1 - gamma))      # Y_j Y_k
        add_pair(b, d, -boundary * 2 * Gamma)           # X_j Y_k
        add_pair(a, c, boundary * 2 * Gamma * alpha)    # Y_j X_k

    return A


def sector_ground_state(A, parity):
    """求指定物理奇偶扇区的最低能量和 Majorana 关联矩阵 M。"""
    T, U = schur(A, output="real")
    tol = 64 * np.finfo(float).eps * max(
        1.0, np.linalg.norm(A, ord=np.inf)
    )

    # 实 Schur 分解将 A 化为独立的 2×2 反对称块。
    # 数值为零的 Majorana 模式也要两两配对。
    left, right, zero, amplitudes = [], [], [], []
    i = 0
    while i < len(T):
        if i + 1 < len(T) and abs(T[i + 1, i]) > tol:
            left.append(i)
            right.append(i + 1)
            amplitudes.append(0.5 * (T[i, i + 1] - T[i + 1, i]))
            i += 2
        else:
            zero.append(i)
            i += 1

    left.extend(zero[::2])
    right.extend(zero[1::2])
    amplitudes.extend([0.0] * (len(zero) // 2))

    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    amplitudes = np.asarray(amplitudes)

    mode_energies = np.abs(amplitudes)
    pair_signs = np.where(amplitudes < 0, -1.0, 1.0)
    energy = -0.5 * mode_energies.sum()

    # 每个块取最低能量后，检查得到的态是否属于目标扇区。
    order = np.column_stack((left, right)).ravel()
    orientation = np.linalg.slogdet(U[:, order])[0]
    state_parity = orientation * np.prod(pair_signs)

    if state_parity != parity:
        # 翻转一个模式就能修正奇偶性，选能量代价最小的模式。
        mode = np.argmin(mode_energies)
        pair_signs[mode] *= -1
        energy += mode_energies[mode]

    rotated = (U[:, left] * pair_signs) @ U[:, right].T
    M = rotated - rotated.T
    return energy, M


def ground_state(N):
    """比较偶扇区的反周期边界与奇扇区的周期边界。"""
    even = sector_ground_state(majorana_matrix(N, +1), +1)
    odd = sector_ground_state(majorana_matrix(N, -1), -1)
    return min(even, odd, key=lambda state: state[0])


def pfaffian(matrix):
    """用带主元交换的消元法计算实反对称矩阵的 Pfaffian。"""
    work = matrix.copy()
    value = 1.0

    for k in range(0, len(work) - 1, 2):
        pivot_j = k + 1 + np.argmax(np.abs(work[k, k + 1:]))
        if work[k, pivot_j] == 0:
            return 0.0

        if pivot_j != k + 1:
            work[[k + 1, pivot_j], :] = work[[pivot_j, k + 1], :]
            work[:, [k + 1, pivot_j]] = work[:, [pivot_j, k + 1]]
            value = -value

        pivot = work[k, k + 1]
        value *= pivot

        u = work[k, k + 2:] / pivot
        v = work[k + 1, k + 2:]
        work[k + 2:, k + 2:] += np.outer(v, u) - np.outer(u, v)

    return value


def spin_correlations(M):
    """四列依次为 C_xx、C_yy、对称 C_xy 和连接 C_zz。"""
    N = len(M) // 2
    C = np.empty((N, 4))
    m_z = M[0, 1]
    C[0] = (1.0, 1.0, 0.0, 1.0 - m_z**2)

    for r in range(1, N // 2 + 1):
        a, b = 2 * r, 2 * r + 1
        indices = np.arange(1, a + 1)

        def string_pf(left, right):
            indices[0] = left
            indices[-1] = right
            return pfaffian(M[np.ix_(indices, indices)])

        xx = string_pf(1, a)
        yy = -string_pf(0, b)
        xy = 0.5 * (string_pf(1, b) - string_pf(0, a))
        zz = M[0, b] * M[1, a] - M[0, a] * M[1, b]

        C[r] = (xx, yy, xy, zz)
        # 平移对称和算符对易使这四个组合满足 C(r) = C(N-r)。
        C[N - r] = C[r]

    return C


def normalized_qfi(C, q_grid):
    N = len(C)
    sites = np.arange(1, N + 1)
    weights = np.cos(q_grid[:, None] * sites)

    # 零填充 FFT 批量计算原公式中的 sum_j w_j * w_{j+r}。
    # 零填充保证这里得到的是线性自相关。
    fft_size = 1 << (2 * N - 1).bit_length()
    spectrum = np.fft.rfft(weights, n=fft_size, axis=1)
    T = np.fft.irfft(np.abs(spectrum)**2, n=fft_size, axis=1)[:, :N]
    T[:, 1:] *= 2

    K_xx, K_yy, K_xy, K_zz = (T @ C).T
    lambda_xy = 0.5 * (
        K_xx + K_yy + np.hypot(K_xx - K_yy, 2 * K_xy)
    )
    return np.maximum(lambda_xy, K_zz) / T[:, 0]


def q_max(N):
    _, M = ground_state(N)
    C = spin_correlations(M)

    q_grid = np.linspace(0.0, np.pi, Q_POINTS)
    nQFI_q = normalized_qfi(C, q_grid)
    q_peak = q_grid[np.argmax(nQFI_q)]

    q_inf = np.pi - np.arctan(
        np.sqrt(-J**2 * gamma**2 - 4 * alpha * Gamma**2) / J
    )
    signed_pct = 100 * (q_peak - q_inf) / q_inf

    return q_peak, q_inf, signed_pct


def main():
    N_values = np.arange(N_MIN, N_MAX + 1, N_STEP)
    results = []

    print(
        f"{'N':>4} {'q_peak/pi':>12} {'q_inf/pi':>12} "
        f"{'有符号偏差 (%)':>16} {'绝对偏差 (%)':>16}",
        flush=True,
    )

    for N in N_values:
        q_peak, q_inf, signed_pct = q_max(int(N))
        results.append((q_peak, q_inf, signed_pct))

        print(
            f"{N:4d} {q_peak / np.pi:12.8f} {q_inf / np.pi:12.8f} "
            f"{signed_pct:+16.6f} {abs(signed_pct):16.6f}",
            flush=True,
        )

    q_peaks, q_refs, signed_pcts = np.asarray(results).T
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

    axes[0].plot(
        N_values, q_peaks / np.pi, "o-",
        color="#00647D", markersize=3, label=r"$q_{\mathrm{peak}}/\pi$",
    )
    axes[0].axhline(
        q_refs[0] / np.pi, color="gray", linestyle="--",
        label=r"$q_\infty/\pi$",
    )
    axes[0].set(xlabel=r"$N$", ylabel=r"$q/\pi$")
    axes[0].legend()

    axes[1].plot(
        N_values, np.abs(signed_pcts), "o-",
        color="#A34830", markersize=3,
    )
    axes[1].set(xlabel=r"$N$", ylabel="Absolute relative deviation (%)")

    for ax in axes:
        ax.margins(y=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    plt.show()


if __name__ == "__main__":
    main()