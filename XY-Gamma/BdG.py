"""绘制 XY-Gamma 链的两条 BdG 能带。

依赖：
    pip install numpy matplotlib

运行：
    python plot_bdg_bands.py

显示绘图窗口：
    python plot_bdg_bands.py --show
"""

from pathlib import Path
import argparse
import json
import math

import numpy as np
import matplotlib


def parameters(gamma, Gamma, alpha):
    g = Gamma * (1.0 + alpha)
    lam = Gamma * (1.0 - alpha)
    a = math.hypot(gamma, g)
    D = a * a - lam * lam
    return g, lam, a, D


def bands(k, h, gamma, Gamma, alpha):
    """计算两条 BdG 能带与 Q(k)。"""
    _, lam, a, D = parameters(gamma, Gamma, alpha)

    A = np.hypot(
        np.cos(k) - h,
        a * np.sin(k),
    )

    shift = 2.0 * lam * np.sin(k)

    E_plus = shift + 2.0 * A
    E_minus = shift - 2.0 * A

    Q = (
        (h - np.cos(k)) ** 2
        + D * np.sin(k) ** 2
    )

    return E_plus, E_minus, Q


def bdg_matrix(k, h, gamma, Gamma, alpha):
    """构造 Nambu 基底 (c_k, c_-k^dagger) 下的矩阵。"""
    g, lam, _, _ = parameters(gamma, Gamma, alpha)

    sk = np.sin(k)
    ck = np.cos(k)

    shift = 2.0 * lam * sk
    z = 2.0 * (ck - h)
    off = -2.0 * (g - 1j * gamma) * sk

    matrix = np.zeros(
        np.shape(k) + (2, 2),
        dtype=complex,
    )

    matrix[..., 0, 0] = shift + z
    matrix[..., 1, 1] = shift - z
    matrix[..., 0, 1] = off
    matrix[..., 1, 0] = np.conjugate(off)

    return matrix


def positive_roots(h, D):
    """返回 [0, pi] 内所有不同的零能动量。"""
    tol = 1e-12

    if D < -tol:
        disc = D * (h * h + D - 1.0)

        if disc < -tol:
            return []

        root = math.sqrt(max(0.0, disc))

        candidates = [
            (h + root) / (1.0 - D),
            (h - root) / (1.0 - D),
        ]

    elif abs(D) <= tol:
        candidates = [h] if abs(h) <= 1.0 + tol else []

    else:
        candidates = (
            [h] if abs(abs(h) - 1.0) <= tol else []
        )

    roots = sorted(
        math.acos(float(np.clip(c, -1, 1)))
        for c in candidates
        if -1 - tol <= c <= 1 + tol
    )

    unique = []

    for value in roots:
        if not unique or abs(value - unique[-1]) > 1e-10:
            unique.append(value)

    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # 模型参数
    parser.add_argument(
        "--gamma", type=float, default=0.6
    )
    parser.add_argument(
        "--capital-gamma", type=float, default=0.6
    )
    parser.add_argument(
        "--alpha", type=float, default=-0.5
    )

    # 各子图对应的磁场
    parser.add_argument(
        "--fields",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 1.1, 1.3],
    )

    # 默认保存路径
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"D:\python-code\python-working\XY-Gamma\picture"
        ),
    )

    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gamma = args.gamma
    Gamma = args.capital_gamma
    alpha = args.alpha

    g, lam, a, D = parameters(gamma, Gamma, alpha)
    hc = math.sqrt(1.0 - D) if D < 0 else None

    k = np.linspace(-np.pi, np.pi, 4001)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })

    colors = ("#176B9A", "#CD7034")

    ncols = min(2, len(args.fields))
    nrows = math.ceil(len(args.fields) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.0, 2.65 * nrows + 0.70),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.11,
        top=0.90,
        hspace=0.36,
        wspace=0.17,
    )

    all_bands = [
        bands(k, h, gamma, Gamma, alpha)
        for h in args.fields
    ]

    maximum = max(
        float(np.max(np.abs(np.stack((ep, em)))))
        for ep, em, _ in all_bands
    )

    ymax = math.ceil(maximum / 2.0) * 2.0

    datasets = []
    audit = []

    for i, (h, band_data) in enumerate(
        zip(args.fields, all_bands)
    ):
        ep, em, Q = band_data
        ax = axes.flat[i]

        roots = positive_roots(h, D)

        # 灰色阴影：Q(k) < 0
        if D < 0 and h * h < 1.0 - D and len(roots) == 2:
            k1, k2 = roots

            ax.axvspan(
                k1, k2,
                color="#AEBBC9",
                alpha=0.30,
                lw=0,
                zorder=0,
            )

            ax.axvspan(
                -k2, -k1,
                color="#AEBBC9",
                alpha=0.30,
                lw=0,
                zorder=0,
            )

        # 零能参考线
        ax.axhline(
            0,
            color="#7F8790",
            ls=(0, (3, 3)),
            lw=0.85,
            zorder=1,
        )

        # 两条能带
        ax.plot(
            k, ep,
            color=colors[0],
            lw=1.75,
            zorder=3,
        )

        ax.plot(
            k, em,
            color=colors[1],
            lw=1.75,
            ls="--",
            zorder=3,
        )

        zero_momenta = sorted({
            x
            for root in roots
            for x in (-root, root)
        })

        # 标记零能点
        if zero_momenta:
            ax.scatter(
                zero_momenta,
                np.zeros(len(zero_momenta)),
                s=23,
                facecolors="white",
                edgecolors="#374754",
                lw=0.9,
                zorder=5,
            )

        # 子图标题只显示 h
        ax.set_title(
            rf"$h={h:g}$",
            pad=10,
        )

        # 子图编号
        ax.text(
            -0.09,
            1.07,
            chr(97 + i),
            transform=ax.transAxes,
            weight="bold",
            fontsize=12,
        )

        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-ymax, ymax)

        ax.set_xticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [
                r"$-\pi$",
                r"$-\pi/2$",
                "$0$",
                r"$\pi/2$",
                r"$\pi$",
            ],
        )

        ax.set_yticks(
            np.linspace(-ymax, ymax, 5)
        )

        ax.tick_params(labelbottom=True)

        if i % ncols == 0:
            ax.set_ylabel(r"$E_{\pm}(k)$")

        # 横轴只显示 k
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$k$")

        # 与直接矩阵对角化比较
        direct = np.linalg.eigvalsh(
            bdg_matrix(k, h, gamma, Gamma, alpha)
        )

        eigen_error = float(
            np.max(
                np.abs(
                    direct - np.column_stack((em, ep))
                )
            )
        )

        # 检查 E_+(k) = -E_-(-k)
        ph_error = float(
            np.max(np.abs(ep + em[::-1]))
        )

        root_error = 0.0

        if zero_momenta:
            rp, rm, _ = bands(
                np.array(zero_momenta),
                h,
                gamma,
                Gamma,
                alpha,
            )

            root_error = float(
                np.max(
                    np.minimum(np.abs(rp), np.abs(rm))
                )
            )

        assert eigen_error < 1e-11
        assert ph_error < 1e-11
        assert root_error < 1e-10

        audit.append({
            "h": h,
            "eigenvalue_max_abs_error": eigen_error,
            "particle_hole_max_abs_error": ph_error,
            "zero_energy_root_max_abs_error": root_error,
            "zero_energy_momenta": zero_momenta,
        })

        datasets.append(
            np.column_stack((
                np.full(k.size, h),
                k,
                ep,
                em,
                Q,
            ))
        )

    # 隐藏多余的子图
    for ax in list(axes.flat)[len(args.fields):]:
        ax.set_visible(False)

    # 公共图例
    handles = [
        Line2D(
            [], [],
            color=colors[0],
            lw=1.75,
            label=r"$E_+(k)$",
        ),
        Line2D(
            [], [],
            color=colors[1],
            lw=1.75,
            ls="--",
            label=r"$E_-(k)$",
        ),
        Patch(
            facecolor="#AEBBC9",
            alpha=0.30,
            label=r"$Q(k)<0$",
        ),
        Line2D(
            [], [],
            marker="o",
            ms=4.5,
            markerfacecolor="white",
            markeredgecolor="#374754",
            ls="",
            label="Zero energy",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        columnspacing=1.7,
        handlelength=2.3,
        fontsize=10,
    )

    stem = args.output_dir / "xy_gamma_bdg_bands"

    # 保存图片
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
    )
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))

    # 保存数据
    np.savetxt(
        stem.with_suffix(".csv"),
        np.concatenate(datasets),
        delimiter=",",
        header="h,k,E_plus,E_minus,Q",
        comments="",
        fmt="%.14g",
    )

    # 保存参数及数值核验结果
    report = {
        "gamma": gamma,
        "Gamma": Gamma,
        "alpha": alpha,
        "g": g,
        "lambda": lam,
        "a": a,
        "D": D,
        "h_c": hc,
        "n_k_per_panel": int(k.size),
        "checks": audit,
        "convention": (
            "Nambu basis (c_k, c_-k^dagger); "
            "signed BdG eigenvalues"
        ),
        "particle_hole_relation": (
            "E_plus(k) = -E_minus(-k)"
        ),
        "shading": (
            "Q(k)<0: singly occupied (k,-k) sectors"
        ),
    }

    stem.with_suffix(".json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("图片和数据已保存到：")
    print(args.output_dir.resolve())

    print(
        "能带核验最大误差：",
        max(
            item["eigenvalue_max_abs_error"]
            for item in audit
        ),
    )

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()