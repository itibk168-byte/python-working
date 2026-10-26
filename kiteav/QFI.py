import os

# Set before importing numpy/scipy; small SVDs benefit from one BLAS thread.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.linalg import eigh, svd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def majorana_covariance(L, x, y, parity=None):
    """Return Q for <a_i b_j> = i Q_ij, using an OBC SVD.

    x=mu/t, y=Delta_0/t >= 0, t>0. parity=None is the absolute
    ground state; +1/-1 selects the lowest even/odd parity state.
    At an exactly zero edge energy, SVD chooses a pure Gaussian
    ground state. This is not a mixture over degenerate states.
    """
    if L < 2 or y < 0 or parity not in (None, 1, -1):
        raise ValueError("Need L>=2, y>=0 and parity=None,+1,-1")

    M = np.diag(np.full(L, -float(x)))
    M += np.diag(np.full(L - 1, float(y) - 1), 1)
    M += np.diag(np.full(L - 1, -(1 + float(y))), -1)

    U, s, Vh = svd(
        M,
        check_finite=False,
        lapack_driver="gesvd",
    )

    if parity is not None:
        p0 = int(
            np.linalg.slogdet(U)[0]
            * np.linalg.slogdet(Vh)[0]
        )

        if p0 != parity:
            # s is descending. Occupying the lowest mode changes parity.
            U[:, -1] *= -1

    return U @ Vh


def qfi_obc(L, x, y, parity=None):
    """Exact Gaussian-state F_Q for the finite open chain."""
    if y == 0:
        # Select a definite-number ground state even at a number crossing.
        return 0.0

    Q = majorana_covariance(L, x, y, parity)

    return float(0.5 * np.sum((Q - Q.T) ** 2))


def density_bulk(x, y):
    """Thermodynamic f_Q, using Eq. (71) of the supplied paper."""
    x, y = np.broadcast_arrays(
        np.asarray(x, float),
        np.asarray(y, float),
    )

    if np.any(y < 0):
        raise ValueError("y=Delta_0/t must be nonnegative")

    h = np.abs(x) / 2
    result = np.zeros_like(h)

    inside = (h <= 1) & (y > 0)
    result[inside] = 2 * y[inside] / (1 + y[inside])

    outside = (h > 1) & (y > 0)
    R = np.sqrt(h[outside] ** 2 + y[outside] ** 2 - 1)

    result[outside] = (
        2 * y[outside] ** 2
        / (R * (R + h[outside]))
    )

    return result.item() if result.ndim == 0 else result


def qfi_fock(L, x, y, parity=None):
    """Independent small-chain check in the full 2**L Fock space."""
    dim = 1 << L

    count = np.array([
        n.bit_count()
        for n in range(dim)
    ])

    operators = []

    for j in range(L):
        c = np.zeros((dim, dim))

        for n in range(dim):
            if (n >> j) & 1:
                sign = (-1) ** (
                    (n & ((1 << j) - 1)).bit_count()
                )

                c[n ^ (1 << j), n] = sign

        operators.append(c)

    H = np.diag(-x * (count - L / 2))

    for j in range(L - 1):
        hop = operators[j].T @ operators[j + 1]
        pair = operators[j] @ operators[j + 1]

        H += -(hop + hop.T) + y * (pair + pair.T)

    ids = np.arange(dim)

    if parity is not None:
        ids = np.flatnonzero((-1) ** count == parity)

    _, vectors = eigh(
        H[np.ix_(ids, ids)],
        subset_by_index=[0, 0],
    )

    probabilities = vectors[:, 0] ** 2
    n = count[ids]

    return float(
        4 * (
            probabilities @ n ** 2
            - (probabilities @ n) ** 2
        )
    )


def validate():
    ed_errors = []
    purity_errors = []
    integral_errors = []

    cases = [
        (6, -3.1, .37),
        (6, -1.3, .21),
        (6, .17, 1.2),
        (8, 2.2, .4),
        (8, 1.5, .2),
        (8, .3, .7),
    ]

    for L, x, y in cases:
        for parity in (None, 1, -1):
            fq = qfi_obc(L, x, y, parity)

            ed_errors.append(
                abs(fq - qfi_fock(L, x, y, parity))
            )

            Q = majorana_covariance(L, x, y, parity)

            C = np.eye(L) / 2 - (Q + Q.T) / 4
            F = (Q.T - Q) / 4

            purity_errors.append(
                abs(
                    np.trace(C @ (np.eye(L) - C))
                    - np.sum(F ** 2)
                )
            )

    integral_cases = [
        (0, .2),
        (1.4, .5),
        (2, .7),
        (2.01, .2),
        (3, 1),
        (4, 2),
    ]

    for x, y in integral_cases:

        def integrand(k):
            d2 = 4 * y ** 2 * np.sin(k) ** 2

            return d2 / (
                (2 * np.cos(k) + x) ** 2 + d2
            )

        f_int = (
            2 / np.pi
            * quad(
                integrand,
                0,
                np.pi,
                epsabs=1e-11,
                epsrel=1e-11,
            )[0]
        )

        integral_errors.append(
            abs(f_int - density_bulk(x, y))
        )

    report = {
        "fock_space_cases": 18,
        "max_FQ_error_vs_fock": max(ed_errors),
        "max_pure_gaussian_trace_identity_error": max(
            purity_errors
        ),
        "max_bulk_closed_form_error_vs_quadrature": max(
            integral_errors
        ),
        "sweet_spot_error_L40": abs(
            qfi_obc(40, 0, 1) - 40
        ),
        "normal_chain_FQ": qfi_obc(40, .17, 0),
    }

    assert report["max_FQ_error_vs_fock"] < 1e-9

    assert (
        report["max_pure_gaussian_trace_identity_error"]
        < 1e-10
    )

    assert (
        report["max_bulk_closed_form_error_vs_quadrature"]
        < 1e-9
    )

    assert report["sweet_spot_error_L40"] < 1e-10

    return report


def save_figure(fig, output, name):
    """Save each figure as PNG and SVG."""
    fig.savefig(
        output / f"{name}.png",
        dpi=300,
        facecolor="white",
    )

    fig.savefig(
        output / f"{name}.svg",
        facecolor="white",
    )

    plt.close(fig)


def make_plots(output, L, xs, ys, finite, bulk):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })

    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#7A5195",
    ]

    X, Y = np.meshgrid(xs, ys)

    # Figure 1: QFI density surface.
    fig = plt.figure(
        figsize=(8.0, 6.2),
        layout="constrained",
    )

    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        finite,
        cmap="cividis",
        linewidth=0,
        antialiased=True,
        rcount=len(ys),
        ccount=len(xs),
        vmin=0,
        vmax=1.4,
    )

    for xc in (-2, 2):
        j = np.argmin(abs(xs - xc))

        ax.plot(
            np.full_like(ys, xc),
            ys,
            finite[:, j] + .008,
            color="#A53D24",
            ls="--",
            lw=1.3,
        )

    ax.set(
        xlabel=r"$\mu/t$",
        ylabel=r"$\Delta_0/t$",
        zlabel="",
        zlim=(0, 1.45),
        xlim=(-4, 4),
        ylim=(0, 2),
    )

    # A 2D label avoids Matplotlib 3D's tight-bounding-box clipping.
    ax.text2D(
        1.10,
        .50,
        r"$f_L=F_Q/L$",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=12,
    )

    ax.set_title(
        f"Open Kitaev chain, L = {L}\n"
        r"$G=L/2-N_f,\quad F_Q=4\,\mathrm{Var}(N_f)$",
        pad=12,
    )

    ax.view_init(elev=27, azim=-55)
    ax.set_box_aspect((1.3, 1, .8))

    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([0, .5, 1, 1.5, 2])

    ax.tick_params(axis="x", pad=0)
    ax.tick_params(axis="y", pad=7)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    save_figure(fig, output, "qfi_density_surface")

    style_handles = [
        Line2D(
            [],
            [],
            color="0.25",
            lw=2,
            label=f"OBC, L={L}",
        ),
        Line2D(
            [],
            [],
            color="0.25",
            ls="--",
            lw=1.5,
            label="Thermodynamic limit",
        ),
    ]

    # Figure 2: QFI density versus chemical potential.
    fig, ax = plt.subplots(
        figsize=(7.6, 4.8),
        layout="constrained",
    )

    ax.axvspan(-2, 2, color="0.5", alpha=.07)

    for xc in (-2, 2):
        ax.axvline(
            xc,
            color=".6",
            ls=":",
            lw=1,
        )

    handles = []

    for color, y in zip(colors, (.2, .5, 1, 2)):
        i = np.argmin(abs(ys - y))

        line, = ax.plot(
            xs,
            finite[i],
            color=color,
            lw=2,
            label=rf"$\Delta_0/t={y:g}$",
        )

        ax.plot(
            xs,
            bulk[i],
            color=color,
            ls="--",
            lw=1.3,
        )

        handles.append(line)

    legend = ax.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        frameon=False,
    )

    ax.add_artist(legend)

    ax.legend(
        handles=style_handles,
        loc="upper right",
        frameon=False,
        fontsize=9,
    )

    ax.set(
        xlabel=r"$\mu/t$",
        ylabel=r"$f_L=F_Q/L$",
        title="QFI density versus chemical potential",
        xlim=(-4, 4),
        ylim=(0, 1.65),
    )

    ax.grid(
        axis="y",
        color=".9",
        lw=.6,
    )

    save_figure(fig, output, "qfi_density_vs_mu")

    # Figure 3: QFI density versus pairing amplitude.
    fig, ax = plt.subplots(
        figsize=(7.6, 4.8),
        layout="constrained",
    )

    handles = []

    # Markers separate nearly coincident x=0 and x=1 curves.
    markers = ["o", "s", "^", "D"]

    for color, marker, x in zip(
        colors,
        markers,
        (0, 1, 2, 3),
    ):
        j = np.argmin(abs(xs - x))

        line, = ax.plot(
            ys,
            finite[:, j],
            color=color,
            lw=1.8,
            marker=marker,
            markevery=10,
            ms=4,
            mfc="white",
            label=rf"$\mu/t={x:g}$",
        )

        ax.plot(
            ys,
            bulk[:, j],
            color=color,
            ls="--",
            lw=1.3,
        )

        handles.append(line)

    legend = ax.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        frameon=False,
    )

    ax.add_artist(legend)

    ax.legend(
        handles=style_handles,
        loc="lower right",
        frameon=False,
        fontsize=9,
    )

    ax.set(
        xlabel=r"$\Delta_0/t$",
        ylabel=r"$f_L=F_Q/L$",
        title="QFI density versus pairing amplitude",
        xlim=(0, 2),
        ylim=(0, 1.48),
    )

    ax.grid(
        axis="y",
        color=".9",
        lw=.6,
    )

    save_figure(fig, output, "qfi_density_vs_delta")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--sites",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=161,
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=81,
    )

    # 默认图片和数据保存路径。
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"D:\软件\PyCharm warehouse\python-working"
            r"\kiteav\assets\kitaev_QFI"
        ),
    )

    args = parser.parse_args()

    if (
        args.sites < 4
        or args.sites % 2
        or args.nx < 5
        or args.ny < 5
    ):
        parser.error(
            "Use an even number of sites >=4 and nx,ny >=5"
        )

    output = args.output.resolve()

    # 目录不存在时自动创建。
    output.mkdir(parents=True, exist_ok=True)

    report = validate()

    print(
        "Validation passed:",
        json.dumps(report),
        flush=True,
    )

    # Include requested cuts exactly, even with a custom grid size.
    xs = np.unique(
        np.r_[
            np.linspace(-4, 4, args.nx),
            -2,
            0,
            1,
            2,
            3,
        ]
    )

    ys = np.unique(
        np.r_[
            np.linspace(0, 2, args.ny),
            .2,
            .5,
            1,
            2,
        ]
    )

    finite = np.empty((len(ys), len(xs)))

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            finite[i, j] = (
                qfi_obc(args.sites, x, y) / args.sites
            )

        if i % 10 == 0 or i == len(ys) - 1:
            print(
                f"OBC grid: {i + 1}/{len(ys)} pairing rows",
                flush=True,
            )

    X, Y = np.meshgrid(xs, ys)
    bulk = density_bulk(X, Y)

    assert np.isfinite(finite).all()

    assert (
        finite.min() >= -1e-12
        and finite.max() <= 2 + 1e-12
    )

    # 保存网格数据。
    with (output / "qfi_density_grid.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "mu_over_t",
            "Delta0_over_t",
            "FQ_OBC",
            "fQ_OBC",
            "fQ_infinite",
        ])

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                writer.writerow([
                    x,
                    y,
                    args.sites * finite[i, j],
                    finite[i, j],
                    bulk[i, j],
                ])

    report["finite_density_range"] = [
        float(finite.min()),
        float(finite.max()),
    ]

    report["particle_hole_FQ_error"] = max(
        abs(
            qfi_obc(args.sites, x, y)
            - qfi_obc(args.sites, -x, y)
        )
        for x, y in [
            (.37, .2),
            (1.4, .5),
            (2.2, 1),
            (3.1, 2),
        ]
    )

    convergence = []

    for L in (40, 80, 120, 240):
        for x, y in [
            (0, .5),
            (2, .5),
            (3, 1),
        ]:
            convergence.append({
                "L": L,
                "x": x,
                "y": y,
                "fQ_OBC": qfi_obc(L, x, y) / L,
                "fQ_infinite": density_bulk(x, y),
            })

    report["size_convergence"] = convergence

    metadata = {
        "definition": (
            "F_Q=4 Var(N_f), G=L/2-N_f, f_L=F_Q/L"
        ),
        "source_equations": [59, 60, 62, 69, 71],
        "boundary": "open",
        "state": "pure absolute Gaussian ground state",
        "normal_line_selection": (
            "definite-number ground state when Delta0=0"
        ),
        "sites": args.sites,
        "grid_shape": [len(ys), len(xs)],
        "x_range": [-4, 4],
        "y_range": [0, 2],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "figure_conclusion": (
            "Bulk density is flat in mu inside |mu/t|<2; "
            "finite OBC corrects the plateau and rounds the cusp."
        ),
        "figure_archetype": (
            "quantitative parameter surface and two "
            "complementary line-cut comparisons"
        ),
        "export": (
            "Python matplotlib, editable SVG text "
            "and 300 dpi PNG, source CSV"
        ),
    }

    # 保存验证结果和参数设置。
    (output / "validation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    (output / "settings.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # 绘制并保存图片。
    make_plots(
        output,
        args.sites,
        xs,
        ys,
        finite,
        bulk,
    )

    print(
        f"Done. Outputs: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()