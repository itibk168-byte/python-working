from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


def _validate(t, delta):
    if not np.isfinite(t) or not np.isfinite(delta) or t <= 0 or delta <= 0:
        raise ValueError("Require finite t > 0 and delta > 0.")


def bands(k, Q, mu, t=1.0, delta=0.3):
    """Return the two BdG eigenvalues E_minus, E_plus."""
    _validate(t, delta)
    h0 = 2 * t * np.sin(Q) * np.sin(k)
    h2 = -2 * delta * np.sin(k)
    h3 = 2 * t * np.cos(Q) * np.cos(k) - mu
    radius = np.hypot(h2, h3)
    return h0 - radius, h0 + radius


def min_phi(Q, mu, t=1.0, delta=0.3):
    """Exact min over k of (R^2-h0^2)/(4*t^2).

    x=cos(k) lies in [-1,1]. Minimize the quadratic on that interval,
    including its endpoints. Inputs follow NumPy broadcasting rules.
    """
    _validate(t, delta)
    Q, m = np.broadcast_arrays(np.asarray(Q, float), np.asarray(mu, float)/(2*t))
    d = delta / t
    cosine = np.cos(Q)
    A = 1 - d*d
    result = np.minimum((cosine-m)**2, (cosine+m)**2)
    if A > 0:
        x = np.clip(m*cosine/A, -1, 1)
        value = (A*x - 2*m*cosine)*x + m*m + d*d - np.sin(Q)**2
        result = np.minimum(result, value)
    return result


def phase(Q, mu, t=1.0, delta=0.3, tol=1e-10):
    """0 trivial gapped; 1 topological gapped; 2 gapless; 3 critical.

    A Weyl point inside the gapless region retains label 2 because the
    minimum over all momenta is negative there. Critical labels refer to
    the boundary of the gapless region or an isolated bulk gap closing.
    """
    value = min_phi(Q, mu, t, delta)
    pfaffian_product = (np.asarray(mu)/(2*t))**2 - np.cos(Q)**2
    label = np.where(pfaffian_product < 0, 1, 0)
    label = np.where(value < -tol, 2, label)
    return np.where(np.abs(value) <= tol, 3, label).astype(np.int8)


def verify():
    """Check independent momentum spectra, exact boundaries and examples."""
    rng = np.random.default_rng(20261026)
    k = np.linspace(-np.pi, np.pi, 16385)
    largest_error = 0.0
    for _ in range(240):
        d = rng.uniform(0.12, 1.5)
        Q = rng.uniform(-np.pi, np.pi)
        mu = rng.uniform(-3.2, 3.2)
        exact = float(min_phi(Q, mu, delta=d))
        em, ep = bands(k, Q, mu, delta=d)
        dense = float(np.min(-em*ep/4))
        error = dense - exact
        assert error >= -2e-12
        assert error < 1e-6
        largest_error = max(largest_error, error)
        if abs(exact) > 1e-5:
            assert (dense < 0) == (exact < 0)

    d = 0.3
    q_star = np.arcsin(d)
    mu_c = 2*np.sqrt(1-d*d)
    examples = [
        (0.2, 0.4, d, 1),
        (0.9, 0.4, d, 2),
        (0.9, 2.4, d, 0),
        (q_star, 0.5, d, 3),
        (1.1, mu_c, d, 3),
        (0.2, 2*np.cos(0.2), d, 3),
        (np.pi/3, 1.0, d, 2),
        (0.4, 0.3, 1.2, 1),
        (1.1, 2.0, 1.2, 0),
        (np.pi/2, 0.0, 1.0, 3),
    ]
    for Q, mu, delta, expected in examples:
        assert int(phase(Q, mu, delta=delta)) == expected

    # Horizontal and vertical Lifshitz boundaries have exact zero modes.
    for Q, mu in [(q_star, 0.5), (1.1, mu_c)]:
        x0 = mu*np.cos(Q)/(2*(1-d*d))
        k0 = np.arccos(np.clip(x0, -1, 1))
        em, ep = bands(k0, Q, mu, delta=d)
        assert min(abs(em), abs(ep)) < 2e-12

    # All four Weyl nodes have both eigenvalues zero.
    mu = 1.0
    qw = np.arccos(mu/2)
    for k0, Q0 in [(0, qw), (0, -qw),
                   (np.pi, np.pi-qw), (np.pi, -np.pi+qw)]:
        em, ep = bands(k0, Q0, mu, delta=d)
        assert max(abs(em), abs(ep)) < 2e-12

    # At delta=t, mu=0, Q=pi/2, one BdG band is zero for every k.
    em, ep = bands(k, np.pi/2, 0, delta=1)
    assert np.max(np.minimum(np.abs(em), np.abs(ep))) < 2e-12
    report = (
        "240 random parameter sets: exact minimization vs 16385 k points.\n"
        f"Maximum discretization error in min_phi: {largest_error:.6e}\n"
        "10 phase examples, both Lifshitz boundaries, four Weyl nodes,\n"
        "and the delta=t flat-zero-band limit: all passed.\n"
    )
    return report


def plot_phase_diagram(output_dir):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    })
    colors = ["#E2E4E7", "#8CC5A9", "#C2ADD7", "#585858"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 4, 1), cmap.N)
    q = np.linspace(-np.pi, np.pi, 1401)
    u = np.linspace(-3, 3, 1101)  # u=mu/t
    fig, axes = plt.subplots(
        1, 3, figsize=(183/25.4, 93/25.4), sharex=True, sharey=True,
        gridspec_kw={"width_ratios": [1.3, 1, 1]},
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.87,
                        bottom=0.30, wspace=0.13)
    for index, (ax, d) in enumerate(zip(axes, [0.3, 0.6, 1.2])):
        labels = phase(q[None, :], u[:, None], delta=d)
        ax.imshow(labels, origin="lower", extent=(-np.pi, np.pi, -3, 3),
                  interpolation="nearest", aspect="auto", cmap=cmap, norm=norm)
        mu_c = 2*np.sqrt(max(0, 1-d*d))
        for sign in [-1, 1]:
            y = sign*2*np.cos(q)
            is_type2 = (d < 1) & (np.abs(y) < mu_c-1e-10)
            ax.plot(q, np.where(~is_type2, y, np.nan),
                    color="#202020", lw=0.9)
            ax.plot(q, np.where(is_type2, y, np.nan),
                    color="#202020", lw=0.9, ls=(0, (3, 2)))
        if d < 1:
            qs = np.arcsin(d)
            for lo, hi in [(qs, np.pi-qs), (-np.pi+qs, -qs)]:
                ax.add_patch(Rectangle(
                    (lo, -mu_c), hi-lo, 2*mu_c,
                    fill=False, edgecolor="#79558D", linewidth=1.15,
                ))
        ax.set(xlim=(-np.pi, np.pi), ylim=(-3, 3), xlabel=r"$Q$")
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                      [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax.tick_params(direction="out", length=2.5, width=0.6, labelsize=6.5)
        ax.set_title(rf"$\Delta_0/t={d:g}$", pad=7, fontsize=8)
        ax.text(-0.02, 1.08, "abc"[index], transform=ax.transAxes,
                fontsize=9, weight="bold", va="bottom")
    axes[0].set_ylabel(r"$\mu/t$")
    handles = [
        Patch(facecolor=colors[1], label="Gapped topological"),
        Patch(facecolor=colors[0], label="Gapped trivial"),
        Patch(facecolor=colors[2], label="Gapless"),
        Line2D([], [], color="#202020", lw=0.9, label="Type-I Weyl locus"),
        Line2D([], [], color="#202020", lw=0.9, ls=(0, (3, 2)),
               label="Type-II Weyl locus"),
        Line2D([], [], color="#79558D", lw=1.15, label="Lifshitz boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.06),
               ncol=3, fontsize=6.6, handlelength=2.0, columnspacing=1.8)
    fig.savefig(r"D:\软件\PyCharm warehouse\python-working\kiteav_supercurrent\assets\kitaev_phase_diagram\phase.png", dpi=300, facecolor="white")
    fig.savefig(r"D:\软件\PyCharm warehouse\python-working\kiteav_supercurrent\assets\kitaev_phase_diagram\phase.svg", facecolor="white")
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent
    report = verify()
    plot_phase_diagram(output_dir)
    print(report)
    print("Saved phase_diagram.png and phase_diagram.svg")


if __name__ == "__main__":
    main()