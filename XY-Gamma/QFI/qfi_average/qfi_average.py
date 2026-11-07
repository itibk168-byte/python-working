"""
S_x + S_y + S_z
"""

import os

# 在导入 NumPy 和 SciPy 前设置，减少小矩阵计算的线程开销。
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_legendre


J = 1.0
GAMMA_XY = 0.6  # 论文中的 γ
GAMMA = 0.6     # 论文中的 Γ
ALPHAS = (-0.50, -0.30, 0.30, 0.50)
N = 8
RADII = (128, 192)

OUTPUT_DIR = Path(r"D:\python-code\python-working\XY-Gamma\QFI\qfi_average\picture")


def finite_chain_qfi(alpha, fields):
    """计算周期链的基态 QFI 密度和相邻场值的基态重叠平方。"""
    basis = np.arange(2**N)
    z = 1 - 2 * ((basis[None, :] >> np.arange(N)[:, None]) & 1)
    sum_z = z.sum(axis=0)
    parity = (-1) ** ((N - sum_z) // 2)

    h0 = np.zeros((basis.size, basis.size), dtype=complex)
    sum_xy = np.zeros_like(h0)

    for j in range(N):
        jp = (j + 1) % N
        flip_j = basis ^ (1 << j)
        flip_both = flip_j ^ (1 << jp)

        h0[flip_both, basis] += (
            J / 2 * ((1 + GAMMA_XY) - (1 - GAMMA_XY) * z[j] * z[jp])
            + 1j * GAMMA * (z[jp] + alpha * z[j])
        )
        sum_xy[flip_j, basis] += 1 + 1j * z[j]

    blocks = []
    for p in (1, -1):
        same = np.flatnonzero(parity == p)
        other = np.flatnonzero(parity == -p)
        blocks.append((
            h0[np.ix_(same, same)],
            sum_z[same],
            sum_xy[np.ix_(other, same)],
        ))

    densities = []
    overlaps = []
    previous = None

    for h in fields:
        candidates = []
        for h_block, z_block, xy_block in blocks:
            matrix = h_block.copy()
            matrix[np.diag_indices_from(matrix)] += h * z_block
            energy, vectors = eigh(
                matrix, subset_by_index=[0, 0], check_finite=False
            )
            candidates.append((energy[0], vectors[:, 0], z_block, xy_block))

        # 比较两个宇称扇区的能量，简并时取偶宇称。
        sector = int(candidates[1][0] < candidates[0][0] - 1e-11)
        _, psi, z_block, xy_block = candidates[sector]

        # 2G = Σ_j(σ_x + σ_y + σ_z)。
        # 横向项翻转宇称，纵向项保持宇称，两部分正交。
        z_psi = z_block * psi
        z_psi -= np.vdot(psi, z_psi) * psi
        xy_psi = xy_block @ psi
        variance_2g = np.vdot(z_psi, z_psi).real + np.vdot(xy_psi, xy_psi).real
        densities.append(variance_2g / N)

        if previous is None:
            overlaps.append(1.0)
        elif sector == previous[0]:
            overlaps.append(abs(np.vdot(previous[1], psi)) ** 2)
        else:
            overlaps.append(0.0)
        previous = (sector, psi)

    return np.array(densities), np.array(overlaps)


@lru_cache(maxsize=None)
def quadrature(order):
    return roots_legendre(order)


def momentum_boundaries(h, alpha):
    """在准粒子能量过零处拆分积分区间。"""
    boundaries = [0.0, np.pi]
    delta = (J * GAMMA_XY) ** 2 + 4 * alpha * GAMMA**2
    discriminant = delta * (h**2 + delta - J**2)

    if delta < 0 and discriminant >= 0:
        roots = (
            J * h + np.array([-1, 1]) * np.sqrt(discriminant)
        ) / (J**2 - delta)
        boundaries.extend(np.arccos(roots[(roots > -1) & (roots < 1)]))

    a = np.hypot(J * GAMMA_XY, GAMMA * (1 + alpha))
    gap_width = abs(h - J) / a
    if 0 < gap_width < 0.05:
        boundaries.extend([gap_width, 10 * gap_width])

    return np.unique(boundaries)


def majorana_covariance(h, alpha, radius):
    """返回旋转后的实空间 Majorana 协方差和纵向 QFI 密度。"""
    nodes, weights = quadrature(max(192, radius + 32))
    onsite = np.array([[0.0, -2 * h], [2 * h, 0.0]])
    bond = np.array([
        [2 * alpha * GAMMA, J * (1 - GAMMA_XY)],
        [-J * (1 + GAMMA_XY), -2 * GAMMA],
    ])
    rotation = np.array([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2)
    distances = np.arange(radius + 1)
    blocks = np.zeros((radius + 1, 2, 2))
    f_z = 0.0
    a = np.hypot(J * GAMMA_XY, GAMMA * (1 + alpha))
    boundaries = momentum_boundaries(h, alpha)

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        k = (left + right) / 2 + (right - left) / 2 * nodes
        dk = (right - left) / 2 * weights
        phase = np.exp(1j * k[:, None, None])

        kernel = 1j * (onsite + bond * phase - bond.T * phase.conj())
        energy, modes = np.linalg.eigh(kernel)
        sign_matrix = (
            (modes * np.sign(energy)[:, None, :])
            @ modes.conj().swapaxes(1, 2)
        )
        symbol = 1j * (rotation @ sign_matrix @ rotation.T)
        fourier = np.exp(-1j * distances[:, None] * k) * (dk / np.pi)
        blocks += (fourier @ symbol.reshape(-1, 4)).real.reshape(-1, 2, 2)

        paired = energy[:, 0] * energy[:, 1] < 0
        numerator = (a * np.sin(k)) ** 2
        pair_weight = numerator / (numerator + (J * np.cos(k) - h) ** 2)
        f_z += (2 / np.pi) * np.dot(dk, paired * pair_weight)

    return blocks, f_z


def pfaffian(matrix):
    """带主元交换的 Pfaffian 消元。"""
    work = matrix.copy()
    value = 1.0

    for j in range(0, len(work), 2):
        pivot_index = j + 1 + np.argmax(abs(work[j, j + 1:]))
        if work[j, pivot_index] == 0:
            return 0.0

        if pivot_index != j + 1:
            work[[j + 1, pivot_index], :] = work[[pivot_index, j + 1], :]
            work[:, [j + 1, pivot_index]] = work[:, [pivot_index, j + 1]]
            value = -value

        pivot = work[j, j + 1]
        value *= pivot
        tail = slice(j + 2, None)
        x, y = work[j, tail], work[j + 1, tail]
        work[tail, tail] -= (np.outer(x, y) - np.outer(y, x)) / pivot

    return value


def leading_pfaffians(work):
    """复用消元，计算各个偶数阶前导子矩阵的 Pfaffian。"""
    size = len(work)
    values = np.zeros(size // 2)
    prefix = 1.0
    start = 0

    while start < size:
        stop = start + 2
        pivot = work[start, start + 1]
        values[stop // 2 - 1] = prefix * pivot

        if abs(pivot) < 1e-8:
            if np.max(abs(work[start:, start:])) < 1e-14:
                break
            while abs(pivot) < 1e-8 and stop < size:
                stop += 2
                pivot = pfaffian(work[start:stop, start:stop])
                values[stop // 2 - 1] = prefix * pivot
            if stop < size:
                block = work[start:stop, start:stop]
                cross = work[start:stop, stop:]
                work[stop:, stop:] += cross.T @ np.linalg.solve(block, cross)
        elif stop < size:
            x = work[start, stop:]
            y = work[start + 1, stop:]
            work[stop:, stop:] -= (np.outer(x, y) - np.outer(y, x)) / pivot

        prefix *= pivot
        start = stop

    return values


def thermodynamic_qfi(h, alpha):
    """用两个关联距离截断估计热力学极限的 QFI 密度。"""
    radius = max(RADII)
    blocks, f_z = majorana_covariance(h, alpha, radius)

    indices = np.arange(1, 2 * radius + 1)
    sites, species = indices // 2, indices % 2
    displacement = sites[None, :] - sites[:, None]
    entries = blocks[abs(displacement), species[:, None], species[None, :]]
    upper = np.triu(entries, 1)

    r = np.arange(1, radius + 1)
    corr_xy = 2 * (-1.0) ** r * leading_pfaffians(upper - upper.T)

    results = []
    for cutoff in RADII:
        t = np.maximum(0.0, 2 * r[:cutoff] / cutoff - 1)
        window = (1 - t) ** 3 * (1 + 3 * t + 6 * t**2)
        results.append(f_z + 2 + 2 * np.dot(window, corr_xy[:cutoff]))

    return np.array(results)


def limit_fields(alpha):
    fields = np.linspace(0.0, 2.5, 401)
    delta = (J * GAMMA_XY) ** 2 + 4 * alpha * GAMMA**2
    critical = [J]
    if delta < 0:
        critical.append(np.sqrt(J**2 - delta))

    offsets = [-0.002, -0.0005, -0.0001, 0.0, 0.0001, 0.0005, 0.002]
    return np.unique(np.r_[fields, np.add.outer(critical, offsets).ravel()])


def main():
    fig, (ax_ed, ax_limit) = plt.subplots(
        1, 2, figsize=(10, 4.4), sharey=True, layout="constrained"
    )
    fields_ed = np.linspace(-2, 2, 1601)
    styles = ("-", "--", "-.", ":")
    colors = ("#2166AC", "#AF5319", "#167D71", "#803D83")
    max_cutoff_difference = 0.0

    for alpha, style, color in zip(ALPHAS, styles, colors):
        print(f"正在计算 alpha = {alpha:+.2f}", flush=True)
        f_ed, overlaps = finite_chain_qfi(alpha, fields_ed)

        # 在基态分支切换处断开曲线。
        breaks = np.flatnonzero(overlaps < 0.5)
        for segment in np.split(np.arange(len(fields_ed)), breaks):
            ax_ed.plot(
                fields_ed[segment], f_ed[segment], style,
                color=color, linewidth=1.7
            )

        fields = limit_fields(alpha)
        f_limit = np.array([thermodynamic_qfi(h, alpha) for h in fields])
        ax_limit.plot(
            fields, f_limit[:, -1], style, color=color,
            linewidth=1.7, label=fr"$\alpha={alpha:+.2f}$"
        )
        max_cutoff_difference = max(
            max_cutoff_difference, np.max(abs(f_limit[:, 1] - f_limit[:, 0]))
        )

    ax_ed.set(
        title=rf"$N={N}$, exact diagonalization",
        xlim=(-2, 2),
        ylabel=r"QFI density $f_Q=F_Q/N$",
    )
    ax_limit.set(title="Thermodynamic-limit estimate", xlim=(0, 2.5))
    ax_limit.legend(frameon=False, loc="lower right")

    for ax in (ax_ed, ax_limit):
        ax.set_xlabel(r"Transverse field $h$")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(r"$G=S_x+S_y+S_z$    $J=1,\ \gamma=\Gamma=0.6$")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "qfi_Sxyz.png"
    fig.savefig(output_file, dpi=300)

    print(f"图片已保存到 {output_file}")
    print(f"Max |f_Q(R=192) - f_Q(R=128)| = {max_cutoff_difference:.3e}")
    plt.show()


if __name__ == "__main__":
    main()