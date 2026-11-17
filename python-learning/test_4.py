"""Compute ground-state QFI and the nQFI peak of a periodic XY–Gamma chain.

Dependencies: NumPy and Matplotlib. Example: python xy_gamma_qfi.py
Use --help for model parameters, chain sizes, and the output directory.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


@dataclass(frozen=True)
class Model:
    J: float = 1.0
    gamma: float = 0.6
    Gamma: float = 0.6
    alpha: float = -0.5
    h: float = 0.4


def sector_ground_state(N, model, parity):
    """Find the lowest admissible mode occupation and its JW contractions."""
    a = model.J * model.gamma
    g = model.Gamma * (1 + model.alpha)
    d = model.Gamma * (1 - model.alpha)

    # APBC for even parity and PBC for odd parity.
    index = np.arange(N)
    shift = 0.5 if parity == 1 else 0.0
    k = 2 * np.pi * (index + shift) / N
    opposite = (-index - int(2 * shift)) % N
    special = index == opposite
    sin_k = np.sin(k)
    xi = model.J * np.cos(k) - model.h
    Delta_k = (-g + 1j * a) * sin_k
    E = np.hypot(xi, abs(Delta_k))
    epsilon = 2 * (E + d * sin_k)
    # In the odd sector k=0,pi are physical unpaired c modes.
    epsilon[special] = 2 * xi[special]

    occupation = (epsilon < 0).astype(int)
    if (-1) ** int(occupation.sum()) != parity:
        flip = np.argmin(np.abs(epsilon))
        occupation[flip] = 1 - occupation[flip]
    energy = float(epsilon @ (occupation - 0.5))

    R = 1 - occupation - occupation[opposite]
    D = occupation - occupation[opposite]
    # At an exactly zero paired BdG block, choose the bare c basis.
    cos_theta = np.divide(xi, E, out=np.ones_like(E), where=E > 0)
    n_k = (1 + D - R * cos_theta) / 2
    p_k = np.divide(R * Delta_k, 2 * E, out=np.zeros_like(Delta_k), where=E > 0)
    n_k[special] = occupation[special]
    p_k[special] = 0

    # G(r) = sum_k exp(+ikr) n_k/N, P(r) = sum_k exp(-ikr) p_k/N.
    # The APBC grid offset contributes the phase outside the FFT.
    phase = np.exp(2j * np.pi * shift * index / N)
    G = phase * np.fft.ifft(n_k)
    P = phase.conj() * np.fft.fft(p_k) / N
    return energy, G, P


def ground_state(N, model):
    """Compare physical parity sectors, selecting the even sector at a tie."""
    even = sector_ground_state(N, model, +1)
    odd = sector_ground_state(N, model, -1)
    parity, state = (+1, even) if even[0] <= odd[0] else (-1, odd)
    energy, G, P = state
    return energy, parity, G, P


def pfaffian(matrix):
    """Compute a real antisymmetric Pfaffian by pivoted elimination."""
    work = matrix.copy()
    value = 1.0
    for i in range(0, len(work), 2):
        j = i + 1 + np.argmax(np.abs(work[i, i + 1:]))
        if work[i, j] == 0:
            return 0.0
        if j != i + 1:
            work[[i + 1, j], :] = work[[j, i + 1], :]
            work[:, [i + 1, j]] = work[:, [j, i + 1]]
            value = -value
        pivot = work[i, i + 1]
        value *= pivot
        row_a = work[i, i + 2:]
        row_b = work[i + 1, i + 2:]
        work[i + 2:, i + 2:] += (
            np.outer(row_b, row_a) - np.outer(row_a, row_b)
        ) / pivot
    return value


def spin_correlations(G, P):
    """Return columns Cxx, Cyy, symmetric Cxy and connected Czz."""
    N = len(G)
    radius = N // 2
    g = G[:radius + 1]
    p = P[:radius + 1]
    delta = (np.arange(radius + 1) == 0).astype(float)

    # Wick contractions of the note's A = c† + c and B = c† - c.
    AA = delta + g - g.conj() + p - p.conj()
    BB = -delta - g + g.conj() + p - p.conj()
    AB = delta - g - g.conj() - p - p.conj()
    BA = -delta + g + g.conj() - p - p.conj()

    # In the Hermitian Majorana basis (A, iB), use M_uv = -i <w_u w_v>
    # for distinct indices. The upper triangle uses nonnegative distances.
    blocks = np.empty((radius + 1, 2, 2))
    blocks[:, 0, 0] = (-1j * AA).real
    blocks[:, 1, 1] = (1j * BB).real
    blocks[:, 0, 1] = AB.real
    blocks[:, 1, 0] = BA.real
    sites, species = divmod(np.arange(2 * (radius + 1)), 2)
    distances = np.maximum(sites[None, :] - sites[:, None], 0)
    upper = np.triu(blocks[distances, species[:, None], species[None, :]], 1)
    M = upper - upper.T

    correlations = np.empty((N, 4))
    mz = 1 - 2 * G[0].real
    correlations[0] = (1.0, 1.0, 0.0, 1 - mz**2)
    for r in range(1, radius + 1):
        string = np.arange(1, 2 * r + 1)

        def contraction(left, right):
            string[0], string[-1] = left, right
            return pfaffian(M[np.ix_(string, string)])

        a_r, b_r = 2 * r, 2 * r + 1
        xx = contraction(1, a_r)
        yy = -contraction(0, b_r)
        xy = (contraction(1, b_r) - contraction(0, a_r)) / 2
        zz = 4 * (abs(P[r])**2 - abs(G[r])**2)
        correlations[r] = (xx, yy, xy, zz)
        correlations[N - r] = correlations[r]
    return correlations


def largest_qfi(components):
    """Largest eigenvalue of the parity-reduced QFI matrix."""
    xx, yy, xy, zz = components.T
    transverse = (xx + yy + np.hypot(xx - yy, 2 * xy)) / 2
    return np.maximum(transverse, zz)


def qfi_curves(correlations, q):
    """Evaluate the note's Eq. (8) for arbitrary real q in [0, pi]."""
    N = len(correlations)
    weights = np.cos(np.multiply.outer(q, np.arange(1, N + 1)))
    fft_length = 1 << (2 * N - 2).bit_length()
    spectrum = np.fft.rfft(weights, n=fft_length, axis=1)
    T = np.fft.irfft(abs(spectrum)**2, n=fft_length, axis=1)[:, :N]
    T[:, 1:] *= 2
    F = largest_qfi(T @ correlations)
    return F, F / T[:, 0]


def numerical_peak(correlations):
    """Maximize nQFI using Eq. (9) at the exact periodic momenta."""
    N = len(correlations)
    # At each periodic momentum the factor N*eta_m cancels T_0(q).
    # The normalized matrix for any other q is a convex combination
    # of these Fourier matrices, so its largest eigenvalue is bounded
    # by the maximum on the periodic grid.
    spectrum = largest_qfi(np.fft.rfft(correlations, axis=0).real)
    mode = np.argmax(spectrum)
    return float(2 * np.pi * mode / N), float(spectrum[mode])


def theoretical_peak(model):
    """Half-width prediction for the single-pocket region |h| < J."""
    b_squared = -(model.J * model.gamma)**2 - 4 * model.alpha * model.Gamma**2
    pairing = np.hypot(model.J * model.gamma, model.Gamma * (1 + model.alpha))
    if model.J <= abs(model.h) or b_squared <= 0 or pairing == 0:
        return np.nan
    return float(np.pi - np.arctan2(np.sqrt(b_squared), model.J))


def configure_plots():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    })


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=450)
    plt.close(fig)


def plot_q_curves(output, model, q, curves, peaks, q_theory):
    colors = plt.cm.viridis(np.linspace(0.10, 0.72, len(curves)))
    settings = rf"$\alpha={model.alpha:g},\ h={model.h:g}$"
    theory_label = r"Theory $q_\infty$"
    figure_size = (4.8, 3.0)

    for column, filename, ylabel in (
        (0, "01_qfi_vs_q", r"$F_Q^{\rm opt}$"),
        (1, "02_nqfi_vs_q", r"$F_Q^{\rm opt}/\sum_j\cos^2(qj)$"),
    ):
        fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
        for color, (N, (F, normalized)) in zip(colors, curves.items()):
            ax.plot(q / np.pi, (F, normalized)[column], color=color, lw=1.15,
                    label=rf"$N={N}$")
            if column == 1:
                peak = peaks[N]
                ax.plot(peak.q_num / np.pi, peak.nqfi, "o", color=color,
                        markersize=4.5, markeredgecolor="white", markeredgewidth=.65)
        if np.isfinite(q_theory):
            ax.axvline(q_theory / np.pi, color="#747C82", ls="--", lw=.85,
                       label=theory_label, zorder=0)
        ax.set(xlabel=r"Probe wave vector $q/\pi$", ylabel=ylabel, xlim=(0, 1),
               title=settings)
        ax.legend(loc="upper left", ncols=2)
        save_figure(fig, output, filename)


@dataclass(frozen=True)
class Peak:
    energy: float
    parity: int
    q_num: float
    nqfi: float
    qfi: float


def scan_model(model, sizes, q, plot_sizes=()):
    curves, peaks = {}, {}
    for N in sizes:
        energy, parity, G, P = ground_state(N, model)
        C = spin_correlations(G, P)
        q_num, nqfi_peak = numerical_peak(C)
        qfi_at_peak = qfi_curves(C, np.array([q_num]))[0][0]
        peaks[N] = Peak(energy, parity, q_num, nqfi_peak, qfi_at_peak)
        if N in plot_sizes:
            curves[N] = qfi_curves(C, q)
    return theoretical_peak(model), peaks, curves


def relative_difference(q_num, q_theory):
    return 100 * (q_num - q_theory) / q_theory


def plot_size_curves(output, h, sizes, scans):
    sizes = np.asarray(sizes)
    marks = np.linspace(0, len(sizes) - 1, min(17, len(sizes)), dtype=int).tolist()
    height_fig, height_ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
    peak_fig, peak_ax = plt.subplots(figsize=(4.8, 3.0), layout="constrained")
    error_fig, error_ax = plt.subplots(figsize=(4.8, 3.0), layout="constrained")
    settings = rf"$h={h:g}$"
    colors, markers = ("#00647D", "#A34830"), ("o", "s")
    upper = float(sizes[-1])
    theory_available = False

    for (alpha, (q_theory, peaks)), color, marker in zip(scans.items(), colors, markers):
        q_num = np.array([peaks[N].q_num for N in sizes])
        relative = abs(relative_difference(q_num, q_theory))
        normalized_height = np.array([peaks[N].qfi for N in sizes]) / np.sqrt(sizes)
        style = dict(color=color, marker=marker, markevery=marks, markersize=3.8,
                     lw=1.05, label=rf"$\alpha={alpha:.2f}$")
        peak_ax.plot(sizes, q_num / np.pi, **style)
        if np.isfinite(q_theory):
            theory_available = True
            peak_ax.axhline(q_theory / np.pi, color=color, ls="--", lw=.85,
                            alpha=.65, zorder=0)
        error_ax.plot(sizes, relative, **style)
        height_ax.plot(sizes, normalized_height, **style)
        upper = max(upper, normalized_height.max())

    peak_ax.set(xlabel=r"Ring size $N$", ylabel=r"Peak wave vector $q/\pi$", title=settings)
    peak_ax.legend(loc="center right", title=r"Dashed: $q_\infty$" if theory_available else None)
    save_figure(peak_fig, output, "03_peak_q_vs_N")

    if theory_available:
        error_ax.axhline(0, color="#747C82", lw=.65, alpha=.55, zorder=0)
    else:
        error_ax.text(.5, .5, "Outside the range of the half-width prediction",
                      transform=error_ax.transAxes, ha="center")
    error_ax.set(xlabel=r"Ring size $N$",
                 ylabel=r"$100\,|q_{\rm num}-q_\infty|/q_\infty$ (%)", title=settings)
    error_ax.legend(loc="upper right")
    save_figure(error_fig, output, "04_relative_difference_vs_N")

    upper *= 1.045
    height_ax.plot([0, upper], [0, upper], ls="--", lw=.9, color="#666B72",
                   label=r"$y=N$", zorder=0)
    ticks = MaxNLocator(nbins=5, integer=True).tick_values(0, upper)
    ticks = ticks[(0 <= ticks) & (ticks <= upper)]
    height_ax.set_xticks(ticks)
    height_ax.set_yticks(ticks)
    height_ax.set(xlim=(0, upper), ylim=(0, upper), xlabel=r"Ring size $N$",
                  ylabel=r"$F_Q^{\rm opt}(q_{\rm num})/\sqrt{N}$", title=settings)
    height_ax.set_aspect("equal", adjustable="box")
    height_ax.legend(loc="upper left")
    save_figure(height_fig, output, "05_qfi_over_sqrt_N_vs_N")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--Gamma", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=-0.5,
                        help="primary alpha for all five figures")
    parser.add_argument("--compare-alpha", type=float, default=-0.4,
                        help="second alpha for figures 3–5")
    parser.add_argument("--h", type=float, default=0.4)
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=256)
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--plot-sizes", type=int, nargs="+", metavar="N",
                        help="ring sizes for the two q curves, chosen from the N scan")
    parser.add_argument("--q-points", type=int, default=2001,
                        help="number of continuous q samples for figures 1 and 2")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if (args.n_min < 4 or args.n_max < args.n_min or args.n_step < 2
            or any(n % 2 for n in (args.n_min, args.n_max, args.n_step))):
        parser.error("N bounds and step must be even, with 4 <= n-min <= n-max and n-step >= 2")
    if args.q_points < 2 or not np.isfinite([args.J, args.gamma, args.Gamma, args.alpha, args.compare_alpha, args.h]).all():
        parser.error("use at least 2 plot samples and finite model parameters")

    sizes = list(range(args.n_min, args.n_max + 1, args.n_step))
    plot_sizes = args.plot_sizes or [N for N in (32, 64, 128) if N in sizes]
    plot_sizes = sorted(set(plot_sizes or [sizes[-1]]))
    if not set(plot_sizes).issubset(sizes):
        parser.error("--plot-sizes must belong to the selected N scan")
    model = Model(args.J, args.gamma, args.Gamma, args.alpha, args.h)
    q = np.linspace(0.0, np.pi, args.q_points)
    q_theory, peaks, curves = scan_model(model, sizes, q, plot_sizes)
    scans = {model.alpha: (q_theory, peaks)}
    if args.compare_alpha != model.alpha:
        comparison = Model(args.J, args.gamma, args.Gamma, args.compare_alpha, args.h)
        other_theory, other_peaks, _ = scan_model(comparison, sizes, q)
        scans[comparison.alpha] = (other_theory, other_peaks)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "q_vs_N.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("alpha", "N", "parity", "ground_energy", "q_num", "q_theory",
                         "q_num_over_pi", "q_theory_over_pi", "nQFI_peak", "QFI_at_q_num",
                         "QFI_over_sqrt_N", "relative_signed_pct", "relative_absolute_pct"))
        for alpha, (predicted, series) in scans.items():
            for N, peak in series.items():
                relative = relative_difference(peak.q_num, predicted)
                writer.writerow((alpha, N, peak.parity, peak.energy, peak.q_num, predicted,
                                 peak.q_num / np.pi, predicted / np.pi, peak.nqfi, peak.qfi,
                                 peak.qfi / np.sqrt(N), relative, abs(relative)))
    with (args.output / "q_curves.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("N", "q", "q_over_pi", "QFI_opt", "nQFI_opt"))
        for N, (F, normalized) in curves.items():
            writer.writerows(zip(np.full(len(q), N), q, q / np.pi, F, normalized))

    configure_plots()
    plot_q_curves(args.output, model, q, curves, peaks, q_theory)
    plot_size_curves(args.output, model.h, sizes, scans)
    for alpha, (predicted, series) in scans.items():
        largest = series[sizes[-1]]
        print(f"alpha={alpha:g}  q_theory/pi={predicted/np.pi:.8f}  "
              f"N={sizes[-1]}  F/sqrt(N)={largest.qfi/np.sqrt(sizes[-1]):.8f}")
    print(f"Five PNG figures and two CSV files saved to {args.output.resolve()}")



if __name__ == "__main__":
    main()
