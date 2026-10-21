"""
同时绘制多个 alpha 下的两种 QFI 密度。

上排：
    N_LARGE 的有限尺寸解析结果。

下排：
    N_NUM 的解析曲线和直接自旋对角化数值点。

原来的 qfi_num、qfi_ana 仍表示交错 QFI 密度。
uniform 表示均匀 QFI 密度。
both 返回：
    (均匀 QFI 密度, 交错 QFI 密度)

依赖：
    numpy
    matplotlib
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

# 输入 2、3 或 4 个互不相同的 alpha。
ALPHAS = [-0.5, -0.3, 0.3, 0.5]

# 大体系：只计算解析结果。
N_LARGE = 1000

# 小体系：同时计算数值和解析结果。
N_NUM = 8

OUTPUT_DIR = Path(
    r"D:\python-code\python-working\XY-Gamma\picture"
)


# ==================== 两种生成元 ====================

@lru_cache(maxsize=4)
def s_z(N):
    """
    交错生成元：
        S_pi^z = 1/2 * sum_j (-1)^j sigma_j^z

    保留上一版的函数名。
    """
    if N > 12:
        raise ValueError(
            "完整自旋矩阵仅用于小体系，"
            "大 N 请使用解析函数。"
        )

    G = np.zeros(
        (2**N, 2**N),
        dtype=complex
    )

    for j in range(1, N + 1):
        G += (
            0.5
            * (-1)**j
            * sp.one_site_kron(sp.sigma_z, j, N)
        )

    return G


@lru_cache(maxsize=4)
def s_z_uniform(N):
    """
    均匀生成元：
        S^z = 1/2 * sum_j sigma_j^z
    """
    if N > 12:
        raise ValueError(
            "完整自旋矩阵仅用于小体系，"
            "大 N 请使用解析函数。"
        )

    G = np.zeros(
        (2**N, 2**N),
        dtype=complex
    )

    for j in range(1, N + 1):
        G += (
            0.5
            * sp.one_site_kron(sp.sigma_z, j, N)
        )

    return G


# ==================== 数值计算 ====================

@lru_cache(maxsize=8)
def _ed_blocks(J, gamma, Gamma, alpha, N):
    """
    缓存无场哈密顿量及生成元。

    自旋宇称为：
        P = product_j sigma_j^z

    在两个宇称块内分别对角化，
    再独立比较两个块的最低能量。
    """
    if N < 4 or N % 2 != 0 or N > 12:
        raise ValueError(
            "数值核对要求 N 是 4 到 12 之间的偶数。"
        )

    H0 = egs.Hamiltonian(
        J, gamma, Gamma, alpha, 0.0, N
    )

    g0 = np.diag(s_z_uniform(N)).real
    gp = np.diag(s_z(N)).real

    spin_parity = np.array([
        (-1)**i.bit_count()
        for i in range(2**N)
    ])

    blocks = []

    for parity in (1, -1):
        indices = np.flatnonzero(
            spin_parity == parity
        )

        blocks.append((
            parity,
            H0[np.ix_(indices, indices)],
            g0[indices],
            gp[indices]
        ))

    return blocks


def numerical_ground_state(
    J, gamma, Gamma, alpha, h, N
):
    """
    独立自旋精确对角化。

    返回：
        energy：基态能量
        density：两种 QFI 密度
        parity：数值基态宇称
        gap：第一激发态能隙
        degenerate：数值精度内是否简并
    """
    results = []

    for parity, H0, g0, gp in _ed_blocks(
        J, gamma, Gamma, alpha, N
    ):
        H = H0.copy()

        # sum_j h*sigma_j^z = 2*h*S^z
        H[np.diag_indices_from(H)] += 2 * h * g0

        values, vectors = np.linalg.eigh(H)

        psi = vectors[:, 0]
        probability = np.abs(psi)**2

        densities = []

        # 两个生成元都在计算基底中对角。
        for diagonal in (g0, gp):
            mean = float(
                np.dot(probability, diagonal)
            )

            second = float(
                np.dot(probability, diagonal**2)
            )

            density = (
                4 * (second - mean**2) / N
            )

            densities.append(density)

        results.append({
            "energy": float(values[0]),
            "next_energy": float(values[1]),
            "density": np.array(densities),
            "parity": parity
        })

    minimum = min(
        result["energy"]
        for result in results
    )

    tolerance = (
        100
        * np.finfo(float).eps
        * max(1.0, abs(minimum))
    )

    # 能量在数值精度内相同时，约定选择偶宇称。
    # 这里不借用解析结果决定数值基态扇区。
    chosen = next(
        result
        for result in results
        if result["energy"] <= minimum + tolerance
    )

    other = next(
        result
        for result in results
        if result["parity"] != chosen["parity"]
    )

    first_excited = min(
        chosen["next_energy"],
        other["energy"]
    )

    gap = max(
        0.0,
        first_excited - chosen["energy"]
    )

    return {
        "energy": chosen["energy"],
        "density": chosen["density"],
        "parity": chosen["parity"],
        "gap": gap,
        "degenerate": gap <= tolerance
    }


def qfi_both_num(
    J, gamma, Gamma, alpha, h, N
):
    """
    返回：
        均匀数值密度、交错数值密度。
    """
    result = numerical_ground_state(
        J, gamma, Gamma, alpha, h, N
    )

    return tuple(result["density"])


def qfi_num(
    J, gamma, Gamma, alpha, h, N
):
    """交错 QFI 数值密度。"""
    return qfi_both_num(
        J, gamma, Gamma, alpha, h, N
    )[1]


def qfi_uniform_num(
    J, gamma, Gamma, alpha, h, N
):
    """均匀 QFI 数值密度。"""
    return qfi_both_num(
        J, gamma, Gamma, alpha, h, N
    )[0]


# ==================== 解析计算 ====================

def ground_state_data(
    J, gamma, Gamma, alpha, h, N
):
    """
    比较两个宇称扇区，
    返回真实基态的能量、动量、占据数和宇称。
    """
    E_even, k_even, n_even = (
        egs.ground_state_sector(
            J, gamma, Gamma, alpha,
            h, N, 1
        )
    )

    E_odd, k_odd, n_odd = (
        egs.ground_state_sector(
            J, gamma, Gamma, alpha,
            h, N, -1
        )
    )

    if E_even <= E_odd:
        return E_even, k_even, n_even, 1

    return E_odd, k_odd, n_odd, -1


def momentum_correlations(
    J, gamma, Gamma, alpha, h,
    k, occupation, parity
):
    """
    从修正后的准粒子占据数计算：
        nu_k = <c_k^dagger c_k>
        t_k：异常关联的实振幅。
    """
    N = len(k)
    indices = np.arange(N)

    if parity == 1:
        negative = (
            N - 1 - indices
        ) % N
    else:
        negative = (
            N - 2 - indices
        ) % N

    n_minus = occupation[negative]

    A = 1 - occupation - n_minus
    d = occupation - n_minus

    a = np.sqrt(
        (J * gamma)**2
        + Gamma**2 * (1 + alpha)**2
    )

    xi = J * np.cos(k) - h

    E = np.sqrt(
        xi**2
        + a**2 * np.sin(k)**2
    )

    special = np.zeros(
        N,
        dtype=bool
    )

    if parity == -1:
        special = (
            np.isclose(
                k, 0,
                rtol=0,
                atol=1e-12
            )
            |
            np.isclose(
                np.abs(k), np.pi,
                rtol=0,
                atol=1e-12
            )
        )

    regular = ~special

    if np.any(E[regular] < 1e-14):
        raise ValueError(
            "Bogoliubov 方向未定，"
            "需要另行指定基态。"
        )

    nu = np.empty(N, dtype=float)
    t = np.zeros(N, dtype=float)

    nu[regular] = (
        1
        + d[regular]
        - xi[regular] / E[regular] * A[regular]
    ) / 2

    t[regular] = (
        a
        * np.sin(k[regular])
        / (2 * E[regular])
        * A[regular]
    )

    # 特殊模沿用 eta_0=c_0、eta_pi=c_pi。
    nu[special] = occupation[special]

    return nu, t


def qfi_both_ana(
    J, gamma, Gamma, alpha, h, N
):
    """
    对同一个修正后的基态，同时计算两种密度。

    返回：
        f_uniform, f_staggered
    """
    energy, k, occupation, parity = (
        ground_state_data(
            J, gamma, Gamma, alpha, h, N
        )
    )

    nu, t = momentum_correlations(
        J, gamma, Gamma, alpha, h,
        k, occupation, parity
    )

    shifted = (
        np.arange(N) + N // 2
    ) % N

    f0 = 4 / N * np.sum(
        nu * (1 - nu)
        + t**2
    )

    fp = 4 / N * np.sum(
        nu * (1 - nu[shifted])
        + t * t[shifted]
    )

    return float(f0), float(fp)


def qfi_ana(
    J, gamma, Gamma, alpha, h, N
):
    """交错 QFI 解析密度。"""
    return qfi_both_ana(
        J, gamma, Gamma, alpha, h, N
    )[1]


def qfi_uniform_ana(
    J, gamma, Gamma, alpha, h, N
):
    """均匀 QFI 解析密度。"""
    return qfi_both_ana(
        J, gamma, Gamma, alpha, h, N
    )[0]


def qfi_ana_sector(
    J, gamma, Gamma, alpha, h, N, parity
):
    """指定宇称扇区的交错 QFI 密度。"""
    energy, k, occupation = (
        egs.ground_state_sector(
            J, gamma, Gamma, alpha,
            h, N, parity
        )
    )

    nu, t = momentum_correlations(
        J, gamma, Gamma, alpha, h,
        k, occupation, parity
    )

    shifted = (
        np.arange(N) + N // 2
    ) % N

    return float(
        4 / N
        * np.sum(
            nu * (1 - nu[shifted])
            + t * t[shifted]
        )
    )


# ==================== 参数检查 ====================

def validate_alphas(alphas):
    values = [
        float(alpha)
        for alpha in alphas
    ]

    if not 2 <= len(values) <= 4:
        raise ValueError(
            "请提供 2、3 或 4 个 alpha。"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "alpha 必须是有限实数。"
        )

    if len(set(values)) != len(values):
        raise ValueError(
            "alpha 不能重复。"
        )

    return values


# ==================== 计算和绘图 ====================

def main(
    alphas,
    N_large,
    N_num,
    points,
    num_points,
    output_dir,
    show=True
):
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    alphas = validate_alphas(alphas)

    if N_large < 4 or N_large % 2 != 0:
        raise ValueError(
            "N_large 必须是不小于 4 的偶数。"
        )

    if (
        N_num < 4
        or N_num % 2 != 0
        or N_num > 12
    ):
        raise ValueError(
            "N_num 必须是 4 到 12 之间的偶数，"
            "建议设为 8。"
        )

    if points < 2 or num_points < 2:
        raise ValueError(
            "磁场采样点数至少为 2。"
        )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    J = J_DEFAULT
    gamma = GAMMA_DEFAULT
    Gamma = CAPITAL_GAMMA_DEFAULT

    h_large = np.linspace(
        0, 2, points
    )

    h_num = np.linspace(
        0, 2, num_points
    )

    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#AA4499"
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10.6, 7.6)
    )

    large_rows = []
    check_rows = []
    summary = []

    for index, alpha in enumerate(alphas):
        color = colors[index]

        # 大体系解析结果。
        large = np.array([
            qfi_both_ana(
                J, gamma, Gamma, alpha,
                h, N_large
            )
            for h in h_large
        ])

        # 小体系解析结果。
        ana = np.array([
            qfi_both_ana(
                J, gamma, Gamma, alpha,
                h, N_num
            )
            for h in h_num
        ])

        # 小体系独立自旋 ED 结果。
        records = [
            numerical_ground_state(
                J, gamma, Gamma, alpha,
                h, N_num
            )
            for h in h_num
        ]

        num = np.array([
            record["density"]
            for record in records
        ])

        E_num = np.array([
            record["energy"]
            for record in records
        ])

        E_ana = np.array([
            ground_state_data(
                J, gamma, Gamma, alpha,
                h, N_num
            )[0]
            for h in h_num
        ])

        gap = np.array([
            record["gap"]
            for record in records
        ])

        degenerate = np.array([
            record["degenerate"]
            for record in records
        ])

        valid = ~degenerate

        if np.any(valid):
            errors = np.max(
                np.abs(
                    num[valid] - ana[valid]
                ),
                axis=0
            )
        else:
            errors = np.array([
                np.nan,
                np.nan
            ])

        energy_error = float(
            np.max(
                np.abs(E_num - E_ana)
            )
        )

        if energy_error > 1e-9:
            raise AssertionError(
                f"alpha={alpha}: "
                "数值和解析能量未通过核对。"
            )

        if (
            np.any(valid)
            and np.max(errors) > 1e-9
        ):
            raise AssertionError(
                f"alpha={alpha}: "
                "数值和解析 QFI 未通过核对。"
            )

        print(
            f"alpha={alpha:g}: "
            f"max df0={errors[0]:.3e}, "
            f"max dfpi={errors[1]:.3e}, "
            f"max dE={energy_error:.3e}, "
            f"omitted degenerate points="
            f"{degenerate.sum()}"
        )

        summary.append([
            alpha,
            errors[0],
            errors[1],
            energy_error,
            int(degenerate.sum())
        ])

        # 简并点的任意数值态未必匹配解析选态，
        # 因此图中不显示该点的 ED 标记。
        num_plot = num.copy()
        num_plot[degenerate] = np.nan

        for column in (0, 1):
            # 上排：大体系解析结果。
            axes[0, column].plot(
                h_large,
                large[:, column],
                color=color,
                linewidth=1.6
            )

            # 下排：同尺寸解析结果。
            axes[1, column].plot(
                h_num,
                ana[:, column],
                color=color,
                linewidth=1.4
            )

            # 下排：直接自旋 ED 空心圆。
            axes[1, column].plot(
                h_num,
                num_plot[:, column],
                "o",
                color=color,
                fillstyle="none",
                markersize=4.3,
                markeredgewidth=0.9,
                markevery=max(
                    1,
                    num_points // 28
                ),
                linestyle="none"
            )

        large_rows.append(
            np.column_stack([
                np.full(points, alpha),
                h_large,
                large
            ])
        )

        check_rows.append(
            np.column_stack([
                np.full(num_points, alpha),
                h_num,
                ana,
                num,
                E_ana,
                E_num,
                gap,
                degenerate.astype(int)
            ])
        )

    axes[0, 0].set_title(
        rf"a   Uniform $S^z$: "
        rf"analytic, $N={N_large}$"
    )

    axes[0, 1].set_title(
        rf"b   Staggered $S_\pi^z$: "
        rf"analytic, $N={N_large}$"
    )

    axes[1, 0].set_title(
        rf"c   Uniform $S^z$: "
        rf"analytic + ED, $N={N_num}$"
    )

    axes[1, 1].set_title(
        rf"d   Staggered $S_\pi^z$: "
        rf"analytic + ED, $N={N_num}$"
    )

    for row in (0, 1):
        axes[row, 0].set_ylabel(
            r"$f_0^Q=F_0^Q/N$"
        )

        axes[row, 1].set_ylabel(
            r"$f_\pi^Q=F_\pi^Q/N$"
        )

    for ax in axes.flat:
        ax.set_xlabel(r"$h$")
        ax.set_xlim(0, 2)
        ax.set_ylim(bottom=0)

        ax.grid(
            alpha=0.2,
            linestyle="--"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    color_legend = [
        Line2D(
            [0], [0],
            color=colors[i],
            linewidth=2,
            label=rf"$\alpha={alpha:g}$"
        )
        for i, alpha in enumerate(alphas)
    ]

    fig.legend(
        handles=color_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(alphas),
        frameon=False
    )

    fig.suptitle(
        rf"$J={J:g},\ "
        rf"\Gamma={Gamma:g},\ "
        rf"\gamma={gamma:g}$",
        y=0.99
    )

    fig.text(
        0.5,
        0.02,
        "Lines: analytic    |    "
        "Open circles: spin ED (bottom row)",
        ha="center",
        fontsize=10
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.98,
        top=0.84,
        bottom=0.10,
        hspace=0.38,
        wspace=0.25
    )

    plt.rcParams["svg.fonttype"] = "none"

    stem = "QFI_uniform_staggered_multi_alpha"

    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=250
    )

    fig.savefig(
        output_dir / f"{stem}.svg"
    )

    np.savetxt(
        output_dir / f"{stem}_large.csv",
        np.vstack(large_rows),
        delimiter=",",
        comments="",
        header="alpha,h,f0_ana,fpi_ana"
    )

    np.savetxt(
        output_dir / f"{stem}_ED.csv",
        np.vstack(check_rows),
        delimiter=",",
        comments="",
        header=(
            "alpha,h,"
            "f0_ana,fpi_ana,"
            "f0_ED,fpi_ED,"
            "E_ana,E_ED,"
            "gap_ED,degenerate"
        )
    )

    np.savetxt(
        output_dir / f"{stem}_errors.csv",
        np.array(summary),
        delimiter=",",
        comments="",
        header=(
            "alpha,"
            "max_df0,max_dfpi,"
            "max_dE,"
            "omitted_degenerate_points"
        )
    )

    print(
        "Saved figure:",
        output_dir / f"{stem}.png"
    )

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=ALPHAS
    )

    parser.add_argument(
        "--N-large",
        type=int,
        default=N_LARGE
    )

    parser.add_argument(
        "--N-num",
        type=int,
        default=N_NUM
    )

    parser.add_argument(
        "--points",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=121
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR
    )

    parser.add_argument(
        "--no-show",
        action="store_true"
    )

    args = parser.parse_args()

    main(
        alphas=args.alphas,
        N_large=args.N_large,
        N_num=args.N_num,
        points=args.points,
        num_points=args.num_points,
        output_dir=args.output_dir,
        show=not args.no_show
    )