from pathlib import Path
import argparse
import numpy as np
import matplotlib

# Save figures without requiring a desktop display. --show enables a window.
if "--show" not in __import__("sys").argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

TRIVIAL, TOPOLOGICAL, GAPLESS = 0, 1, 2


def spectrum(k, mu_over_t, delta_over_t):
    """Positive BdG energy E(k)/t; t > 0 and delta = |Delta|."""
    return np.sqrt(
        (-2 * np.cos(k) - mu_over_t) ** 2
        + 4 * np.abs(delta_over_t) ** 2 * np.sin(k) ** 2
    )


def bulk_gap(mu_over_t, delta_over_t):
    """Exact min_k E(k)/t, obtained by minimizing a quadratic in cos(k)."""
    u, d = np.broadcast_arrays(
        np.asarray(mu_over_t, dtype=float),
        np.abs(np.asarray(delta_over_t, dtype=float)),
    )
    # The smaller endpoint value at cos(k) = +1 or -1.
    gap_squared = (np.abs(u) - 2) ** 2
    # For d < 1 the quadratic is convex; its vertex may lie in [-1, 1].
    x_star = np.zeros_like(u)
    np.divide(-u, 2 * (1 - d * d), out=x_star, where=(d < 1))
    interior = (d < 1) & (np.abs(x_star) <= 1)
    vertex_value = (-2 * x_star - u) ** 2 + 4 * d * d * (1 - x_star**2)
    gap_squared = np.where(interior, vertex_value, gap_squared)
    return np.sqrt(np.maximum(gap_squared, 0.0))


def classify_phase(mu_over_t, delta_over_t, atol=1e-10):
    """0: gapped trivial; 1: gapped topological; 2: gapless/boundary.

    atol is only a floating-point equality tolerance. It does not define a
    physical finite-width gapless region around the exact phase boundaries.
    """
    u, d = np.broadcast_arrays(
        np.asarray(mu_over_t, dtype=float),
        np.abs(np.asarray(delta_over_t, dtype=float)),
    )
    phase = np.where(np.abs(u) < 2, TOPOLOGICAL, TRIVIAL)
    boundary = np.isclose(np.abs(u), 2, rtol=0.0, atol=atol)
    normal_gapless = (d == 0) & (np.abs(u) <= 2 + atol)
    return np.where(boundary | normal_gapless, GAPLESS, phase)


def check_implementation():
    """Verify special points and compare the exact gap with dense k sampling."""
    for u, d, expected in (
        (0.0, 0.6, TOPOLOGICAL), (3.0, 0.6, TRIVIAL),
        (2.0, 0.6, GAPLESS), (-2.0, 1.2, GAPLESS),
        (0.0, 0.0, GAPLESS), (3.0, 0.0, TRIVIAL),
    ):
        assert int(classify_phase(u, d)) == expected
    assert np.isclose(bulk_gap(0, 1), 2)
    assert np.isclose(bulk_gap(2, 0.6), 0)
    rng = np.random.default_rng(20261025)
    points = list(zip(rng.uniform(-3.5, 3.5, 64), rng.uniform(0.02, 1.6, 64)))
    points += [(0.7, 0.0), (-1.3, 0.0), (0.4, 1.0)]
    k = np.linspace(0, np.pi, 20001)
    errors = []
    for u, d in points:
        exact = float(bulk_gap(u, d))
        sampled = float(np.min(spectrum(k, u, d)))
        assert sampled >= exact - 1e-9
        errors.append(abs(sampled - exact))
    # Finite k sampling overestimates the continuum minimum.
    assert max(errors) < 3e-4
    print(f"Gap check passed; max finite-grid error = {max(errors):.3e}")


def make_figure(out_dir):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    })
    u = np.linspace(-3.5, 3.5, 1401)
    d = np.linspace(0, 1.6, 501)
    U, D = np.meshgrid(u, d)
    phases = classify_phase(U, D)
    cmap = ListedColormap(["#E5E7EB", "#A5CAD8", "#D58B70"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.4), layout="constrained",
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    ax = axes[0]
    ax.pcolormesh(u, d, phases, cmap=cmap, norm=norm, shading="nearest",
                  rasterized=True)
    for edge in (-2, 2):
        ax.axvline(edge, color="#222222", lw=1.3)
    ax.plot([-2, 2], [0, 0], color="#B76143", lw=3.5,
            solid_capstyle="butt", clip_on=False)
    for x in (-2.78, 2.78):
        ax.text(x, 0.91, "Trivial\n" + r"$\mathcal{M}=+1$",
                ha="center", va="center", linespacing=1.5)
    ax.text(0, 0.91, "Topological\n" + r"$\mathcal{M}=-1$",
            ha="center", va="center", linespacing=1.5, fontsize=11)
    ax.annotate("Gapless normal line", xy=(0, 0.005), xytext=(0, 0.24),
                ha="center", color="#8C432D", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": "#8C432D", "lw": 0.8})
    ax.set(xlim=(-3.5, 3.5), ylim=(0, 1.6), xlabel=r"$\mu/t$",
           ylabel=r"$\Delta_0/t=|\Delta|/t$", title="Bulk phase diagram")
    ax.set_xticks([-3, -2, 0, 2, 3])
    ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])

    ax = axes[1]
    for ratio, color in ((0.3, "#287C9E"), (0.6, "#D18C28"), (1.2, "#7864A7")):
        ax.plot(u, bulk_gap(u, ratio), lw=1.8, color=color,
                label=rf"$\Delta_0/t={ratio}$")
    for edge in (-2, 2):
        ax.axvline(edge, color="#777777", ls="--", lw=0.8, zorder=0)
    ax.set(xlim=(-3.5, 3.5), ylim=(0, 2.2), xlabel=r"$\mu/t$",
           ylabel=r"$E_{\mathrm{gap}}/t$", title="Bulk-gap verification")
    ax.set_xticks([-3, -2, 0, 2, 3])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", color="#E4E4E4", lw=0.6)
    for letter, ax in zip(("a", "b"), axes):
        ax.text(-0.09, 1.04, letter, transform=ax.transAxes,
                fontsize=12, fontweight="bold")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        destination = out_dir / f"kitaev_phase_diagram.{suffix}"
        fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
        print(destination.resolve())
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "assets" / "kitaev_phase_diagram")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    check_implementation()
    fig = make_figure(args.out)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

