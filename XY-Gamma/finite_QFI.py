"""
将 N=6、8、10 的两种 QFI 合并为一个 3 行 2 列的图。
每行对应一个 N，左列均匀 QFI，右列交错 QFI。
保留多个 alpha 的解析曲线和直接自旋 ED 空心圆。

依赖：numpy、matplotlib，以及同一项目中的
energy_ground_state.py、energy_spectrum.py 和 my_package/。
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import my_package.spin_multibody as sp
import energy_ground_state as egs


# ==================== 可修改参数 ====================
J_DEFAULT = 1.0
GAMMA_DEFAULT = 0.6
CAPITAL_GAMMA_DEFAULT = 0.6
ALPHAS = [-0.5, -0.3, 0.3, 0.5]
N_VALUES = [6, 8, 10]
NUM_POINTS = 121
OUTPUT_DIR = Path(r"D:\python-code\python-working\XY-Gamma\picture")
COLORS = ["#0072B2", "#D55E00", "#009E73", "#AA4499"]


# ==================== 两种生成元 ====================
@lru_cache(maxsize=4)
def s_z(N):
    """交错生成元 S_pi^z；保留原函数名。"""
    if N > 12:
        raise ValueError("完整自旋矩阵仅用于 N<=12 的小体系。")
    G = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(1, N + 1):
        G += 0.5 * (-1)**j * sp.one_site_kron(sp.sigma_z, j, N)
    return G


@lru_cache(maxsize=4)
def s_z_uniform(N):
    """均匀生成元 S^z。"""
    if N > 12:
        raise ValueError("完整自旋矩阵仅用于 N<=12 的小体系。")
    G = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(1, N + 1):
        G += 0.5 * sp.one_site_kron(sp.sigma_z, j, N)
    return G


# ==================== 数值计算 ====================
@lru_cache(maxsize=8)
def _ed_blocks(J, gamma, Gamma, alpha, N):
    """缓存无场 Hamiltonian 的两个宇称块及生成元对角元。"""
    if N < 4 or N % 2 or N > 12:
        raise ValueError("数值核对要求 N 是 4 到 12 之间的偶数。")
    H0 = egs.Hamiltonian(J, gamma, Gamma, alpha, 0.0, N)
    g0 = np.diag(s_z_uniform(N)).real
    gp = np.diag(s_z(N)).real
    spin_parity = np.array([(-1)**i.bit_count() for i in range(2**N)])
    blocks = []
    for parity in (1, -1):
        indices = np.flatnonzero(spin_parity == parity)
        blocks.append((
            parity, H0[np.ix_(indices, indices)],
            g0[indices], gp[indices]
        ))
    return blocks


def numerical_ground_state(J, gamma, Gamma, alpha, h, N):
    """分别对角化两个自旋宇称块，再独立比较最低能量。"""
    results = []
    for parity, H0, g0, gp in _ed_blocks(J, gamma, Gamma, alpha, N):
        H = H0.copy()
        H[np.diag_indices_from(H)] += 2 * h * g0
        values, vectors = np.linalg.eigh(H)
        probability = np.abs(vectors[:, 0])**2
        densities = []
        for diagonal in (g0, gp):
            mean = float(np.dot(probability, diagonal))
            second = float(np.dot(probability, diagonal**2))
            densities.append(4 * (second - mean**2) / N)
        results.append({
            "energy": float(values[0]),
            "next_energy": float(values[1]),
            "density": np.array(densities),
            "parity": parity
        })
    minimum = min(result["energy"] for result in results)
    tolerance = 100 * np.finfo(float).eps * max(1.0, abs(minimum))
    # 数值精度内两扇区等能时，约定选择偶宇称。
    chosen = next(r for r in results if r["energy"] <= minimum + tolerance)
    other = next(r for r in results if r["parity"] != chosen["parity"])
    first_excited = min(chosen["next_energy"], other["energy"])
    gap = max(0.0, first_excited - chosen["energy"])
    return {
        "energy": chosen["energy"],
        "density": chosen["density"],
        "parity": chosen["parity"],
        "gap": gap,
        "degenerate": gap <= tolerance
    }


def qfi_both_num(J, gamma, Gamma, alpha, h, N):
    return tuple(numerical_ground_state(J, gamma, Gamma, alpha, h, N)["density"])


def qfi_num(J, gamma, Gamma, alpha, h, N):
    """交错 QFI 数值密度。"""
    return qfi_both_num(J, gamma, Gamma, alpha, h, N)[1]


def qfi_uniform_num(J, gamma, Gamma, alpha, h, N):
    """均匀 QFI 数值密度。"""
    return qfi_both_num(J, gamma, Gamma, alpha, h, N)[0]


# ==================== 解析计算 ====================
def ground_state_data(J, gamma, Gamma, alpha, h, N):
    """比较两个宇称扇区，返回真实基态的数据。"""
    E_even, k_even, n_even = egs.ground_state_sector(
        J, gamma, Gamma, alpha, h, N, 1
    )
    E_odd, k_odd, n_odd = egs.ground_state_sector(
        J, gamma, Gamma, alpha, h, N, -1
    )
    if E_even <= E_odd:
        return E_even, k_even, n_even, 1
    return E_odd, k_odd, n_odd, -1


def momentum_correlations(J, gamma, Gamma, alpha, h, k, occupation, parity):
    """计算正常关联 nu_k 和异常关联的实振幅 t_k。"""
    N = len(k)
    indices = np.arange(N)
    if parity == 1:
        negative = (N - 1 - indices) % N
    else:
        negative = (N - 2 - indices) % N
    n_minus = occupation[negative]
    A = 1 - occupation - n_minus
    d = occupation - n_minus
    a = np.sqrt((J * gamma)**2 + Gamma**2 * (1 + alpha)**2)
    xi = J * np.cos(k) - h
    E = np.sqrt(xi**2 + a**2 * np.sin(k)**2)
    special = np.zeros(N, dtype=bool)
    if parity == -1:
        special = (
            np.isclose(k, 0, rtol=0, atol=1e-12)
            | np.isclose(np.abs(k), np.pi, rtol=0, atol=1e-12)
        )
    regular = ~special
    if np.any(E[regular] < 1e-14):
        raise ValueError("Bogoliubov 方向未定，需要另行指定基态。")
    nu = np.empty(N, dtype=float)
    t = np.zeros(N, dtype=float)
    nu[regular] = (
        1 + d[regular] - xi[regular] / E[regular] * A[regular]
    ) / 2
    t[regular] = a * np.sin(k[regular]) / (2 * E[regular]) * A[regular]
    nu[special] = occupation[special]
    return nu, t


def qfi_both_ana(J, gamma, Gamma, alpha, h, N):
    """返回同一个基态的 (均匀 QFI 密度, 交错 QFI 密度)。"""
    _, k, occupation, parity = ground_state_data(J, gamma, Gamma, alpha, h, N)
    nu, t = momentum_correlations(
        J, gamma, Gamma, alpha, h, k, occupation, parity
    )
    shifted = (np.arange(N) + N // 2) % N
    f0 = 4 / N * np.sum(nu * (1 - nu) + t**2)
    fp = 4 / N * np.sum(nu * (1 - nu[shifted]) + t * t[shifted])
    return float(f0), float(fp)


def qfi_ana(J, gamma, Gamma, alpha, h, N):
    """交错 QFI 解析密度。"""
    return qfi_both_ana(J, gamma, Gamma, alpha, h, N)[1]


def qfi_uniform_ana(J, gamma, Gamma, alpha, h, N):
    """均匀 QFI 解析密度。"""
    return qfi_both_ana(J, gamma, Gamma, alpha, h, N)[0]


def qfi_ana_sector(J, gamma, Gamma, alpha, h, N, parity):
    """指定宇称扇区的交错 QFI 密度。"""
    _, k, occupation = egs.ground_state_sector(
        J, gamma, Gamma, alpha, h, N, parity
    )
    nu, t = momentum_correlations(
        J, gamma, Gamma, alpha, h, k, occupation, parity
    )
    shifted = (np.arange(N) + N // 2) % N
    return float(4 / N * np.sum(
        nu * (1 - nu[shifted]) + t * t[shifted]
    ))


# ==================== 参数检查 ====================
def validate_alphas(alphas):
    values = [float(alpha) for alpha in alphas]
    if not 2 <= len(values) <= 4:
        raise ValueError("请提供 2、3 或 4 个 alpha。")
    if not np.all(np.isfinite(values)):
        raise ValueError("alpha 必须是有限实数。")
    if len(set(values)) != len(values):
        raise ValueError("alpha 不能重复。")
    return values


def validate_sizes(sizes):
    sizes = list(sizes)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("N 列表不能为空，且不能包含重复值。")
    if any(N < 4 or N > 12 or N % 2 for N in sizes):
        raise ValueError("每个 N 必须是 4 到 12 之间的偶数。")
    return sizes


# ==================== 合并绘图 ====================
def plot_combined(data_by_size, alphas, sizes, output_dir, show=True):
    """将各 N 的数据画入同一个 len(sizes) 行、2 列的 Figure。"""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False
    })
    fig, axes = plt.subplots(
        len(sizes), 2,
        figsize=(11, 3.05 * len(sizes) + 1.0),
        sharex=True, sharey="col", squeeze=False
    )

    for row, N in enumerate(sizes):
        table = data_by_size[N]
        for index, alpha in enumerate(alphas):
            # 数据列：N,alpha,h,f0_ana,fpi_ana,f0_ED,fpi_ED,
            #         E_ana,E_ED,gap_ED,degenerate
            values = table[np.isclose(table[:, 1], alpha, rtol=0, atol=1e-12)]
            values = values[np.argsort(values[:, 2])]
            if len(values) < 2:
                raise ValueError(f"N={N}, alpha={alpha}: 缺少绘图数据。")
            h_values = values[:, 2]
            ana = values[:, 3:5]
            num_plot = values[:, 5:7].copy()
            num_plot[values[:, 10].astype(bool)] = np.nan
            for column in (0, 1):
                ax = axes[row, column]
                ax.plot(
                    h_values, ana[:, column],
                    color=COLORS[index], linewidth=1.5
                )
                ax.plot(
                    h_values, num_plot[:, column], "o",
                    color=COLORS[index], fillstyle="none",
                    markersize=4.0, markeredgewidth=0.9,
                    markevery=max(1, len(values) // 28), linestyle="none"
                )

        for column in (0, 1):
            ax = axes[row, column]
            letter = chr(ord("a") + 2 * row + column)
            ax.set_title(rf"({letter})  $N={N}$", loc="left", fontsize=11)
            ax.set_xlim(0, 2)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2, linestyle="--")
        axes[row, 0].set_ylabel(r"$f_0^Q=F_Q[S^z]/N$")
        axes[row, 1].set_ylabel(r"$f_\pi^Q=F_Q[S_\pi^z]/N$")

    # 底行统一显示横坐标；同列共用纵坐标范围，便于比较尺寸效应。
    axes[-1, 0].set_xlabel(r"$h$")
    axes[-1, 1].set_xlabel(r"$h$")
    fig.suptitle(
        rf"$J={J_DEFAULT:g},\ \Gamma={CAPITAL_GAMMA_DEFAULT:g},"
        rf"\ \gamma={GAMMA_DEFAULT:g}$",
        y=0.994, fontsize=13
    )
    handles = [
        Line2D([0], [0], color=COLORS[i], linewidth=2,
               label=rf"$\alpha={alpha:g}$")
        for i, alpha in enumerate(alphas)
    ] + [
        Line2D([0], [0], color="black", linewidth=1.5, label="Analytic"),
        Line2D([0], [0], color="black", marker="o", fillstyle="none",
               linestyle="none", markersize=4.0, label="Spin ED")
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.966),
        ncol=len(handles), frameon=False, fontsize=9.5
    )
    fig.text(0.292, 0.914, r"Uniform $S^z$", ha="center", fontsize=12)
    fig.text(0.780, 0.914, r"Staggered $S_\pi^z$", ha="center", fontsize=12)
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.06, top=0.864,
        wspace=0.20, hspace=0.27
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "QFI_uniform_staggered_" + "_".join(f"N{N}" for N in sizes)
    for extension in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300)
        print("Saved:", path, flush=True)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ==================== 计算并绘图 ====================
def main(alphas, sizes, num_points, output_dir, show=True):
    alphas = validate_alphas(alphas)
    sizes = validate_sizes(sizes)
    if num_points < 2:
        raise ValueError("磁场采样点数至少为 2。")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    J, gamma, Gamma = J_DEFAULT, GAMMA_DEFAULT, CAPITAL_GAMMA_DEFAULT
    h_values = np.linspace(0, 2, num_points)
    summary = []
    data_by_size = {}

    for N in sizes:
        data_rows = []
        for alpha in alphas:
            print(f"Computing N={N}, alpha={alpha:g} ...", flush=True)
            ana = np.array([
                qfi_both_ana(J, gamma, Gamma, alpha, h, N)
                for h in h_values
            ])
            records = [
                numerical_ground_state(J, gamma, Gamma, alpha, h, N)
                for h in h_values
            ]
            num = np.array([r["density"] for r in records])
            E_num = np.array([r["energy"] for r in records])
            E_ana = np.array([
                ground_state_data(J, gamma, Gamma, alpha, h, N)[0]
                for h in h_values
            ])
            gap = np.array([r["gap"] for r in records])
            degenerate = np.array([r["degenerate"] for r in records])
            valid = ~degenerate
            errors = (
                np.max(np.abs(num[valid] - ana[valid]), axis=0)
                if np.any(valid) else np.array([np.nan, np.nan])
            )
            energy_error = float(np.max(np.abs(E_num - E_ana)))
            if energy_error > 1e-9:
                raise AssertionError(f"N={N}, alpha={alpha}: 能量核对失败。")
            if np.any(valid) and np.max(errors) > 1e-9:
                raise AssertionError(f"N={N}, alpha={alpha}: QFI 核对失败。")
            print(
                f"  max df0={errors[0]:.3e}, max dfpi={errors[1]:.3e}, "
                f"max dE={energy_error:.3e}, "
                f"omitted degenerate points={degenerate.sum()}",
                flush=True
            )
            summary.append([
                N, alpha, errors[0], errors[1], energy_error,
                int(degenerate.sum())
            ])
            data_rows.append(np.column_stack([
                np.full(num_points, N), np.full(num_points, alpha),
                h_values, ana, num, E_ana, E_num, gap,
                degenerate.astype(int)
            ]))

        data_by_size[N] = np.vstack(data_rows)
        np.savetxt(
            output_dir / f"QFI_comparison_N{N}.csv", data_by_size[N],
            delimiter=",", comments="",
            header=("N,alpha,h,f0_ana,fpi_ana,f0_ED,fpi_ED,"
                    "E_ana,E_ED,gap_ED,degenerate")
        )
        _ed_blocks.cache_clear()
        s_z.cache_clear()
        s_z_uniform.cache_clear()

    np.savetxt(
        output_dir / "QFI_errors.csv", np.array(summary),
        delimiter=",", comments="",
        header="N,alpha,max_df0,max_dfpi,max_dE,omitted_degenerate_points"
    )
    plot_combined(data_by_size, alphas, sizes, output_dir, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--sizes", nargs="+", type=int, default=N_VALUES)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    main(
        alphas=args.alphas, sizes=args.sizes, num_points=args.num_points,
        output_dir=args.output_dir, show=not args.no_show
    )


