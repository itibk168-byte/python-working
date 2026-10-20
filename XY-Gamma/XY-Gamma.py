from pathlib import Path
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# 参数：固定 J = 1
GAMMA = 0.6                   # XY 各向异性参数 gamma
CAPITAL_GAMMA = 0.6           # 非对角交换参数 Gamma
ALPHAS = (-0.5, -0.3, 0.3, 0.5)

# 严格位于开区间 (0, 2.5) 内
H_MIN = 1.0e-5
H_MAX = 2.5 - 1.0e-5


def parameters(alpha, gamma=GAMMA, Gamma=CAPITAL_GAMMA):
    """计算 a、b、D；注意 a 包含平方根。"""
    a = math.hypot(gamma, Gamma * (1.0 + alpha))
    b = Gamma * (1.0 - alpha)
    D = gamma**2 + 4.0 * alpha * Gamma**2
    return a, b, D


def coherence_weight(k, h, a):
    """
    计算 B(k) = 1 - R(k)。

    R(k) = [h² - cos²(k) + a² sin²(k)] / [E_minus * E_plus]
    E_minus = sqrt[(cos(k) - h)² + a² sin²(k)]
    E_plus  = sqrt[(cos(k) + h)² + a² sin²(k)]
    """
    s = math.sin(k)
    c = math.cos(k)

    E_minus = math.hypot(c - h, a * s)
    E_plus = math.hypot(c + h, a * s)

    product = E_minus * E_plus
    numerator = h*h - c*c + a*a*s*s

    if product == 0.0:
        # a > 0 时，h = 1、k = 0 处的连续极限
        return 1.0

    if numerator > 0.0:
        # 利用恒等式避免 1 - R 的数值消减误差：
        # product² - numerator² = 4a² sin²(k) cos²(k)
        return (
            4.0 * a*a * s*s * c*c
            / (product * (product + numerator))
        )

    return 1.0 - numerator / product


def integrate_weight(h, a, lower, upper):
    """在指定区间对 B(k) 自适应积分。"""
    if upper <= lower:
        return 0.0

    # 在 h 接近 1 时，解析 k = 0 附近的狭窄变化区间
    scale = abs(h - 1.0) / a
    points = [
        x for x in (scale, 10.0 * scale, 100.0 * scale)
        if lower < x < upper
    ]

    value, _ = quad(
        coherence_weight,
        lower,
        upper,
        args=(h, a),
        points=sorted(set(points)),
        epsabs=2.0e-10,
        epsrel=2.0e-10,
        limit=200,
    )
    return value


def qfi_density(h, alpha, gamma=GAMMA, Gamma=CAPITAL_GAMMA):
    """
    交错 z 生成元对应的热力学极限 QFI 密度：
        f_pi^Q = lim_{N→∞} F_pi^Q / N

    使用解析确定的积分区间处理负能准粒子占据。
    """
    h = abs(float(h))
    a, b, D = parameters(alpha, gamma, Gamma)

    # 无配对极限；本次指定参数不会进入此分支
    if a == 0.0:
        bandwidth = math.hypot(1.0, b)
        if h >= bandwidth:
            return 0.0
        return 4.0 / math.pi * math.acos(h / bandwidth)

    # 无负能区间：
    # f = (2/pi) * integral_0^(pi/2) B(k) dk
    if D >= 0.0 or h >= math.sqrt(1.0 - D):
        return (
            2.0 / math.pi
            * integrate_weight(h, a, 0.0, math.pi / 2.0)
        )

    # 有负能区间：D < 0 且 h < sqrt(1-D)
    delta = math.sqrt(-D)
    root = math.sqrt(max(0.0, 1.0 + delta*delta - h*h))

    r_plus = (h + delta * root) / (1.0 + delta*delta)
    r_minus = (h - delta * root) / (1.0 + delta*delta)

    # 防止浮点误差令 arccos 的参数略超出 [-1, 1]
    k1 = math.acos(max(-1.0, min(1.0, r_plus)))
    k2 = math.acos(max(-1.0, min(1.0, r_minus)))

    integral = integrate_weight(h, a, 0.0, k1)

    if h >= delta:
        integral += integrate_weight(
            h, a, k2, math.pi / 2.0
        )

    # h < delta:
    # f = (2/pi) * [k2-k1 + integral_0^k1 B(k) dk]
    #
    # h >= delta:
    # 额外包含 integral_k2^(pi/2) B(k) dk
    return 2.0 / math.pi * (k2 - k1 + integral)


def make_h_grid():
    """构造网格，并在分段位置及其两侧加密。"""
    special = [1.0]

    for alpha in ALPHAS:
        _, _, D = parameters(alpha)
        if D < 0.0:
            special.extend([
                math.sqrt(-D),
                math.sqrt(1.0 - D),
            ])

    offsets = np.geomspace(1.0e-7, 0.025, 60)

    parts = [
        np.linspace(H_MIN, H_MAX, 1201),
        np.asarray(special),
    ]

    for center in special:
        parts.extend([
            center - offsets,
            center + offsets,
        ])

    h = np.unique(np.concatenate(parts))
    return h[(h >= H_MIN) & (h <= H_MAX)]


def main(show=True):
    h = make_h_grid()

    curves = np.array([
        [qfi_density(x, alpha) for x in h]
        for alpha in ALPHAS
    ])

    output_dir = Path(__file__).resolve().parent
    stem = output_dir / "staggered_qfi_vs_h"

    # 保存数据：第一列为 h，其余列为不同 alpha 的密度
    np.savetxt(
        stem.with_suffix(".csv"),
        np.column_stack([h, curves.T]),
        delimiter=",",
        comments="",
        header="h," + ",".join(
            f"f_alpha_{a:+.1f}" for a in ALPHAS
        ),
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8),
        layout="constrained",
    )

    colors = ("#0072B2", "#D55E00", "#009E73", "#AA4499")
    styles = ("-", "--", "-.", ":")

    for alpha, f, color, style in zip(
        ALPHAS, curves, colors, styles
    ):
        ax.plot(
            h,
            f,
            color=color,
            linestyle=style,
            linewidth=1.9,
            label=rf"$\alpha={alpha:.1f}$",
        )

    ax.set(
        xlim=(0.0, 2.5),
        ylim=(0.0, 1.90),
        xlabel=r"$h$",
        ylabel=r"$f_{\pi}^{Q}=F_{\pi}^{Q}/N$",
        title=r"$J=1,\quad \Gamma=\gamma=0.6$",
    )

    ax.set_xticks(np.arange(0.0, 2.51, 0.5))
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=3.0)

    fig.savefig(stem.with_suffix(".png"), dpi=400)
    fig.savefig(stem.with_suffix(".svg"))

    print(" alpha            a            D          h_c")
    for alpha in ALPHAS:
        a, _, D = parameters(alpha)
        hc = f"{math.sqrt(1.0 - D):.9f}" if D < 0.0 else "--"
        print(f"{alpha:6.1f} {a:12.9f} {D:12.9f} {hc:>12}")

    print("Saved: staggered_qfi_vs_h.png, .svg, .csv")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    main(show=not parser.parse_args().no_show)