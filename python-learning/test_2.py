"""Particle-number QFI of a current-carrying Kitaev chain.

Generator: G = N_s/2 - N_f.  For the pure state, F = 4 Var(N_f).
This is NOT the QFI for estimating the supercurrent parameter Q.

All energies are divided by w: delta = Delta_0/w, mu_bar = mu/w.
Only NumPy and Matplotlib are needed. Matplotlib is imported when plotting.

Examples:
    python qfi_supercurrent.py
    python qfi_supercurrent.py --delta 0.3 --nq 1001 --nmu 801

Reference: Phys. Rev. Research 6, 033060 (2024).
DOI: 10.1103/PhysRevResearch.6.033060
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=8)
def gauss_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(order)


def qfi_density(Q, mu_over_w, delta: float = 0.6, nquad: int = 256):
    """Return F/N_s, broadcasting Q and mu_over_w like NumPy arrays.

    f = (2/pi) integral_0^pi R(k) Theta(P(k)) dk,
    R = delta^2 sin^2(k) / [(cos(Q) cos(k)-mu_bar/2)^2
                            + delta^2 sin^2(k)],
    P = (cos(Q) cos(k)-mu_bar/2)^2
        + [delta^2-sin^2(Q)] sin^2(k).

    We exploit f(Q,mu)=f(-Q,mu)=f(Q,-mu)=f(Q+pi,mu).
    Gapped integrals are analytic. In gapless regions, the integral is split
    at the exact pocket boundaries. No discrete step-function approximation
    is used, and no smoothing is applied.
    """
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    if nquad < 16:
        raise ValueError("nquad must be at least 16")

    q, mu = np.broadcast_arrays(np.asarray(Q, float), np.asarray(mu_over_w, float))
    shape = q.shape
    if not (np.isfinite(q).all() and np.isfinite(mu).all()):
        raise ValueError("Q and mu_over_w must be finite")
    c = np.abs(np.cos(q)).reshape(-1)
    c = np.where(c < 1e-14, 0.0, c)
    s2 = np.sin(q).reshape(-1) ** 2
    m = np.abs(mu).reshape(-1) / 2.0
    f = np.empty_like(m)

    # Strict inequalities describe the open gapless region. Critical lines
    # have the continuous limiting value; zero modes have zero measure.
    gapless = (s2 > delta**2) & (m**2 < 1.0 - delta**2)
    full_bz = ~gapless
    inner = full_bz & (m <= c)
    outer = full_bz & ~inner
    f[inner] = 2.0 * delta / (c[inner] + delta)
    r = np.sqrt(m[outer] ** 2 + delta**2 - c[outer] ** 2)
    f[outer] = 2.0 * delta**2 / (r * (r + m[outer]))

    ids = np.flatnonzero(gapless)
    if ids.size:
        nodes, weights = gauss_nodes(nquad)
        t = 1.0 - delta**2
        # Chunking limits the largest quadrature array to 2048*nquad values.
        for start in range(0, ids.size, 2048):
            ix = ids[start : start + 2048]
            cc, mm = c[ix], m[ix]
            ell = np.sqrt(np.maximum(s2[ix] - delta**2, 0.0))
            rr = np.sqrt(np.maximum(t - mm**2, 0.0))

            # The pocket is (k1,k2). Rationalizing 1-cos(k1) avoids
            # catastrophic cancellation near the Weyl line m=c.
            denominator = np.sqrt(np.maximum(2.0 * (t - cc * mm + ell * rr), 0.0))
            z = np.divide(np.abs(cc - mm), denominator,
                          out=np.zeros_like(mm), where=denominator > 0)
            k1 = 2.0 * np.arcsin(np.clip(z, 0.0, 1.0))
            k2 = np.arccos(np.clip((cc * mm - ell * rr) / t, -1.0, 1.0))

            def integrate(lo, hi):
                width = hi - lo
                k = lo[:, None] + 0.5 * width[:, None] * (nodes[None, :] + 1.0)
                numerator = delta**2 * np.sin(k) ** 2
                mass = (cc - mm)[:, None] - 2.0 * cc[:, None] * np.sin(k / 2.0) ** 2
                denominator_r = numerator + mass**2
                integrand = np.divide(numerator, denominator_r,
                                      out=np.zeros_like(numerator), where=denominator_r > 0)
                return 0.5 * width * (integrand @ weights)

            f[ix] = (2.0 / np.pi) * (
                integrate(np.zeros_like(k1), k1)
                + integrate(k2, np.full_like(k2, np.pi))
            )

    result = f.reshape(shape)
    return float(result) if result.ndim == 0 else result


def reference_phases(Q, mu_over_w, delta=0.6):
    """0 = gapped trivial, 1 = gapped topological, 2 = gapless.

    This classification uses analytic gap/topological criteria, NOT a
    threshold on QFI. Points exactly on a boundary inherit a limiting label.
    """
    q, mu = np.broadcast_arrays(Q, mu_over_w)
    gapless = (np.sin(q) ** 2 > delta**2) & ((mu / 2.0) ** 2 < 1.0 - delta**2)
    phase = np.zeros(q.shape, dtype=np.int8)
    phase[(np.abs(mu) < 2.0 * np.abs(np.cos(q))) & ~gapless] = 1
    phase[gapless] = 2
    return phase


def curvature_indicator(f, Q, mu_over_w):
    """Grid-dependent display diagnostic, not a topological invariant."""
    f_mu, f_q = np.gradient(f, mu_over_w, Q, edge_order=2)
    f_mumu = np.gradient(f_mu, mu_over_w, axis=0, edge_order=2)
    f_qq = np.gradient(f_q, Q, axis=1, edge_order=2)
    return np.abs(f_mumu) + np.abs(f_qq)


def draw_reference_lines(ax, delta):
    q = np.linspace(-np.pi, np.pi, 6001)
    type_i = np.abs(np.sin(q)) < delta
    type_ii = ~type_i
    for sign in (-1.0, 1.0):
        node_line = sign * 2.0 * np.cos(q)
        ax.plot(q, np.where(type_i, node_line, np.nan), color="#202020", lw=0.85)
        ax.plot(q, np.where(type_ii, node_line, np.nan), color="#202020",
                lw=0.85, linestyle=(0, (4, 3)))
    if delta < 1.0:
        qstar = np.arcsin(delta)
        muc = 2.0 * np.sqrt(1.0 - delta**2)
        for q0 in (-np.pi + qstar, -qstar, qstar, np.pi - qstar):
            ax.plot([q0, q0], [-muc, muc], color="#355c9a", lw=1.0)
        for lo, hi in ((-np.pi + qstar, -qstar), (qstar, np.pi - qstar)):
            for y in (-muc, muc):
                ax.plot([lo, hi], [y, y], color="#355c9a", lw=1.0)


def plot_comparison(Q, mu, f, curvature, phase, delta, outdir, formats, show=False):
    try:
        import matplotlib as mpl
        if not show:
            mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise SystemExit("Plotting needs matplotlib. Install: python -m pip install numpy matplotlib") from exc

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "text.usetex": False,
    })
    fig = plt.figure(figsize=(12.0, 4.5), layout="constrained")
    gs = fig.add_gridspec(3, 3, height_ratios=(1.0, 0.055, 0.10))
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    caxes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    titles = ("Particle-number QFI", "QFI curvature (no guides)", "Analytic reference phases")
    ticks = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
    ticklabels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    for i, ax in enumerate(axes):
        ax.set(xlim=(Q[0], Q[-1]), ylim=(mu[0], mu[-1]), xlabel=r"$Q$", title=titles[i])
        ax.set_xticks(ticks, ticklabels)
        ax.tick_params(direction="out", width=0.8, labelleft=(i == 0))
        ax.text(-0.10, 1.025, "abc"[i], transform=ax.transAxes, weight="bold", fontsize=11)
    axes[0].set_ylabel(r"$\mu/w$")

    im0 = axes[0].pcolormesh(Q, mu, f, shading="nearest", cmap="viridis",
                             vmin=0.0, vmax=max(1e-12, float(f.max())), rasterized=True)
    fig.colorbar(im0, cax=caxes[0], orientation="horizontal", label=r"$f_{\mathrm{QFI}}$")

    display = np.log10(1.0 + curvature)
    vmax = max(1e-12, float(np.percentile(display, 99.5)))
    im1 = axes[1].pcolormesh(Q, mu, display, shading="nearest", cmap="magma",
                             vmin=0.0, vmax=vmax, rasterized=True)
    fig.colorbar(im1, cax=caxes[1], orientation="horizontal",
                 label=r"$\log_{10}(1+|f_{QQ}|+|f_{\bar\mu\bar\mu}|)$")

    cmap = ListedColormap(["#dedede", "#d5e8a4", "#c6c7e8"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    im2 = axes[2].pcolormesh(Q, mu, phase, shading="nearest", cmap=cmap, norm=norm, rasterized=True)
    draw_reference_lines(axes[2], delta)
    cb = fig.colorbar(im2, cax=caxes[2], orientation="horizontal", ticks=[0, 1, 2])
    cb.ax.set_xticklabels(["Gapped trivial", "Gapped topological", "Gapless"], fontsize=8)

    footer = fig.add_subplot(gs[2, :])
    footer.axis("off")
    handles = [Line2D([], [], color="#202020", lw=0.9, label="Type-I Weyl nodes"),
               Line2D([], [], color="#202020", lw=0.9, linestyle=(0, (4, 3)), label="Type-II Weyl nodes"),
               Line2D([], [], color="#355c9a", lw=1.1, label="Gapless phase boundaries")]
    footer.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
                  title="Analytic guides appear only in panel c", title_fontsize=8)
    fig.suptitle(rf"Current-carrying Kitaev chain: $\Delta_0/w={delta:g}$; $F=4\,\mathrm{{Var}}(N_f)$",
                 fontsize=11)

    for suffix in formats:
        fig.savefig(outdir / f"qfi_phase_comparison.{suffix}", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return vmax


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delta", type=float, default=0.6, help="Delta_0/w (positive)")
    parser.add_argument("--nq", type=int, default=721)
    parser.add_argument("--nmu", type=int, default=601)
    parser.add_argument("--mu-max", type=float, default=2.4)
    parser.add_argument("--nquad", type=int, default=256)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--formats", nargs="+", choices=["png", "svg", "pdf"], default=["png", "svg"])
    parser.add_argument("--show", action="store_true", help="also open an interactive Matplotlib window")
    args = parser.parse_args()
    if args.nq < 5 or args.nmu < 5 or args.mu_max <= 0:
        parser.error("nq and nmu must be >= 5, mu-max must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)

    Q = np.linspace(-np.pi, np.pi, args.nq)
    mu = np.linspace(-args.mu_max, args.mu_max, args.nmu)
    print(f"Computing {args.nmu} x {args.nq} points, quadrature order {args.nquad} ...", flush=True)
    f = qfi_density(Q[None, :], mu[:, None], args.delta, args.nquad)
    phase = reference_phases(Q[None, :], mu[:, None], args.delta)
    curvature = curvature_indicator(f, Q, mu)
    np.savez_compressed(args.outdir / "qfi_data.npz", Q=Q, mu_over_w=mu,
                        qfi_density=f, curvature=curvature, reference_phase=phase)
    vmax = plot_comparison(Q, mu, f, curvature, phase, args.delta,
                           args.outdir, args.formats, args.show)
    metadata = {
        "delta": args.delta, "nq": args.nq, "nmu": args.nmu, "nquad": args.nquad,
        "generator": "G=N_s/2-N_f; F=4 Var(N_f)", "temperature": 0,
        "state": "thermodynamic-limit pure current-carrying state",
        "q_range": [-float(np.pi), float(np.pi)], "mu_over_w_range": [-args.mu_max, args.mu_max],
        "qfi_min": float(f.min()), "qfi_max": float(f.max()),
        "phase_codes": {"0": "gapped trivial", "1": "gapped topological", "2": "gapless"},
        "curvature_display": "log10(1+abs(f_QQ)+abs(f_mumu)); 99.5 percentile saturation",
        "curvature_display_max": vmax, "smoothing": "none",
        "note": "Reference phases use analytic criteria, not a QFI threshold. Curvature is grid dependent.",
    }
    if args.delta < 1:
        metadata.update(qstar=float(np.arcsin(args.delta)),
                        mu_critical_over_w=float(2 * np.sqrt(1 - args.delta**2)))
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Done. Outputs:", args.outdir.resolve())


if __name__ == "__main__":
    main()
