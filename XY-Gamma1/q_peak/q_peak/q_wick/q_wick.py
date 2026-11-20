"""Periodic XY-Gamma chain. Run with NumPy and Matplotlib.

Outputs four PNGs and one CSV. QFI and nQFI in the CSV are evaluated at
the nQFI peak q_N. Wave vectors are in radians. The primary alpha block
comes first, followed by the comparison alpha block, each in N order.
"""

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


OUTPUT = Path(r"D:\python-code\python-working\XY-Gamma1\q_peak\q_peak\q_wick")
COLORS = ("#00647D", "#A34830")
MARKERS = ("o", "s")


@dataclass(frozen=True)
class Model:
    J: float
    gamma: float
    Gamma: float
    alpha: float
    h: float


@dataclass(frozen=True)
class Peak:
    q: float
    nqfi: float
    qfi: float


def sector_ground_state(N, model, parity):
    """Return the lowest allowed energy and the JW contractions G and P."""
    a = model.J * model.gamma
    g = model.Gamma * (1 + model.alpha)
    d = model.Gamma * (1 - model.alpha)
    index = np.arange(N)
    shift = 0.5 if parity == 1 else 0.0  # APBC for even parity, PBC for odd.
    k = 2 * np.pi * (index + shift) / N
    opposite = (-index - int(2 * shift)) % N
    special = index == opposite  # Unpaired k = 0, pi.

    xi = model.J * np.cos(k) - model.h
    sin_k = np.sin(k)
    delta = (-g + 1j * a) * sin_k
    E = np.hypot(xi, abs(delta))
    epsilon = 2 * (E + d * sin_k)
    epsilon[special] = 2 * xi[special]

    occupation = (epsilon < 0).astype(int)
    if (-1) ** int(occupation.sum()) != parity:
        flip = np.argmin(abs(epsilon))
        occupation[flip] = 1 - occupation[flip]
    energy = float(epsilon @ (occupation - 0.5))

    R = 1 - occupation - occupation[opposite]
    D = occupation - occupation[opposite]
    cos_theta = np.divide(xi, E, out=np.ones_like(E), where=E > 0)
    n_k = (1 + D - R * cos_theta) / 2
    p_k = np.divide(R * delta, 2 * E, out=np.zeros_like(delta), where=E > 0)
    n_k[special] = occupation[special]
    p_k[special] = 0

    phase = np.exp(2j * np.pi * shift * index / N)
    G = phase * np.fft.ifft(n_k)
    P = phase.conj() * np.fft.fft(p_k) / N
    return energy, G, P


def pfaffian(matrix):
    """Real antisymmetric Pfaffian by pivoted elimination."""
    work = matrix.copy()
    value = 1.0
    for i in range(0, len(work), 2):
        j = i + 1 + np.argmax(abs(work[i, i + 1:]))
        if work[i, j] == 0:
            return 0.0
        if j != i + 1:
            work[[i + 1, j], :] = work[[j, i + 1], :]
            work[:, [i + 1, j]] = work[:, [j, i + 1]]
            value = -value
        pivot = work[i, i + 1]
        value *= pivot
        a, b = work[i, i + 2:], work[i + 1, i + 2:]
        work[i + 2:, i + 2:] += (np.outer(b, a) - np.outer(a, b)) / pivot
    return value


def spin_correlations(G, P):
    """Return Cxx, Cyy, symmetric Cxy and connected Czz."""
    N = len(G)
    radius = N // 2
    g, p = G[:radius + 1], P[:radius + 1]
    delta = (np.arange(radius + 1) == 0).astype(float)

    # Wick contractions in the Majorana basis (A, iB).
    AA = delta + g - g.conj() + p - p.conj()
    BB = -delta - g + g.conj() + p - p.conj()
    AB = delta - g - g.conj() - p - p.conj()
    BA = -delta + g + g.conj() - p - p.conj()
    blocks = np.empty((radius + 1, 2, 2))
    blocks[:, 0, 0], blocks[:, 1, 1] = (-1j * AA).real, (1j * BB).real
    blocks[:, 0, 1], blocks[:, 1, 0] = AB.real, BA.real
    sites, species = divmod(np.arange(2 * (radius + 1)), 2)
    distances = np.maximum(sites[None, :] - sites[:, None], 0)
    upper = np.triu(blocks[distances, species[:, None], species[None, :]], 1)
    M = upper - upper.T

    C = np.empty((N, 4))
    mz = 1 - 2 * G[0].real
    C[0] = (1, 1, 0, 1 - mz**2)
    for r in range(1, radius + 1):
        string = np.arange(1, 2 * r + 1)

        def contraction(left, right):
            string[0], string[-1] = left, right
            return pfaffian(M[np.ix_(string, string)])

        a, b = 2 * r, 2 * r + 1
        xx = contraction(1, a)
        yy = -contraction(0, b)
        xy = (contraction(1, b) - contraction(0, a)) / 2
        zz = 4 * (abs(P[r])**2 - abs(G[r])**2)
        C[r] = C[N - r] = (xx, yy, xy, zz)
    return C


def largest_qfi(components):
    xx, yy, xy, zz = components.T
    transverse = (xx + yy + np.hypot(xx - yy, 2 * xy)) / 2
    return np.maximum(transverse, zz)


def qfi_curves(C, q):
    """Evaluate optimized QFI and nQFI for arbitrary q in [0, pi]."""
    N = len(C)
    weights = np.cos(np.multiply.outer(q, np.arange(1, N + 1)))
    length = 1 << (2 * N - 2).bit_length()
    spectrum = np.fft.rfft(weights, n=length, axis=1)
    T = np.fft.irfft(abs(spectrum)**2, n=length, axis=1)[:, :N]
    T[:, 1:] *= 2
    F = largest_qfi(T @ C)
    return F, F / T[:, 0]


def numerical_peak(C):
    # Away from the exact periodic momenta the normalized QFI matrix is
    # a convex combination of the Fourier matrices at those momenta.
    spectrum = largest_qfi(np.fft.rfft(C, axis=0).real)
    mode = np.argmax(spectrum)
    q = float(2 * np.pi * mode / len(C))
    F, _ = qfi_curves(C, [q])
    return Peak(q, float(spectrum[mode]), float(F[0]))


def theoretical_peak(model):
    """Half-width prediction in the single-pocket region."""
    a = model.J * model.gamma
    b_squared = -a**2 - 4 * model.alpha * model.Gamma**2
    pairing = np.hypot(a, model.Gamma * (1 + model.alpha))
    if model.J <= abs(model.h) or b_squared <= 0 or pairing == 0:
        return np.nan
    return float(np.pi - np.arctan2(np.sqrt(b_squared), model.J))


def scan_model(model, sizes, q, plot_sizes=()):
    peaks, curves = {}, {}
    for N in sizes:
        even = sector_ground_state(N, model, +1)
        odd = sector_ground_state(N, model, -1)
        _, G, P = even if even[0] <= odd[0] else odd
        C = spin_correlations(G, P)
        peaks[N] = numerical_peak(C)
        if N in plot_sizes:
            curves[N] = qfi_curves(C, q)
    return theoretical_peak(model), peaks, curves


def save_plot(fig, output, filename, **options):
    fig.savefig(output / filename, dpi=450, **options)
    plt.close(fig)


def plot_q_curves(output, model, q, curves, peaks, q_inf):
    colors = plt.cm.viridis(np.linspace(0.10, 0.72, len(curves)))
    figures = (
        ("01_qfi_vs_q.png", r"$F_Q^{\rm opt}$"),
        ("02_nqfi_vs_q.png", r"$F_Q^{\rm opt}/\sum_j\cos^2(qj)$"),
    )
    for column, (filename, ylabel) in enumerate(figures):
        fig, ax = plt.subplots(figsize=(4.8, 3.0), layout="constrained")
        for color, (N, data) in zip(colors, curves.items()):
            ax.plot(q / np.pi, data[column], color=color, lw=1.15, label=f"$N={N}$")
            if column == 1:
                peak = peaks[N]
                ax.plot(peak.q / np.pi, peak.nqfi, "o", color=color, markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.65)
        if np.isfinite(q_inf):
            ax.axvline(q_inf / np.pi, color="#747C82", ls="--", lw=0.85,
                       label=r"Theory $q_\infty$", zorder=0)
        ax.set(xlabel=r"Probe wave vector $q/\pi$", ylabel=ylabel, xlim=(0, 1),
               title=rf"$\alpha={model.alpha:g},\ h={model.h:g}$")
        ax.legend(loc="upper left", ncols=2)
        save_plot(fig, output, filename)


def plot_size_curves(output, h, sizes, scans):
    N = np.asarray(sizes)
    title = f"$h={h:g}$"
    scale_fig, scale = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
    diff_fig, diff = plt.subplots(figsize=(5.6, 3.5), layout="constrained")
    marks = np.linspace(0, len(N) - 1, min(17, len(N)), dtype=int).tolist()
    upper = float(N[-1])

    for (alpha, (q_inf, peaks)), color, marker in zip(scans.items(), COLORS, MARKERS):
        height = np.array([peaks[n].qfi for n in N]) / np.sqrt(N)
        q_N = np.array([peaks[n].q for n in N])
        scale.plot(N, height, color=color, marker=marker, markevery=marks,
                   markersize=3.8, lw=1.05, label=rf"$\alpha={alpha:.2f}$")
        diff.plot(N, abs(q_N - q_inf) / np.pi, color=color, marker=marker,
                  markersize=2.8, lw=1.05, label=rf"$\alpha={alpha:g}$")
        upper = max(upper, height.max())

    upper *= 1.045
    scale.plot([0, upper], [0, upper], "--", lw=0.9, color="#666B72", label=r"$y=N$", zorder=0)
    ticks = MaxNLocator(nbins=5, integer=True).tick_values(0, upper)
    ticks = ticks[(0 <= ticks) & (ticks <= upper)]
    scale.set(xticks=ticks, yticks=ticks, xlim=(0, upper), ylim=(0, upper),
              xlabel=r"Ring size $N$", ylabel=r"$F_Q^{\rm opt}(q_N)/\sqrt{N}$", title=title)
    scale.set_aspect("equal", adjustable="box")
    scale.legend(loc="upper left")
    save_plot(scale_fig, output, "03_qfi_over_sqrt_N_vs_N.png")

    diff.plot(N, 2.0 / N, "--", color="#555555", lw=1.3, label=r"$2/N$")
    diff.set(xlabel=r"Ring size $N$", ylabel=r"$|q_N-q_\infty|/\pi$", title=title)
    diff.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    diff.margins(x=0.02, y=0.12)
    diff.set_ylim(bottom=0)
    diff.grid(axis="y", alpha=0.18, linewidth=0.6)
    diff.legend(loc="upper right")
    save_plot(diff_fig, output, "04_absolute_q_difference_vs_N.png", bbox_inches="tight")


def write_peak_data(path, scans):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("N", "QFI_opt", "nQFI", "q_N", "q_inf", "|q_N-q_inf|/pi", "2/N"))
        for q_inf, peaks in scans.values():
            for N, peak in peaks.items():
                difference = abs(peak.q - q_inf) / np.pi
                writer.writerow((N, peak.qfi, peak.nqfi, peak.q, q_inf, difference, 2.0 / N))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--Gamma", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=-0.5)
    parser.add_argument("--compare-alpha", type=float, default=-0.4)
    parser.add_argument("--h", type=float, default=0.4)
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=100)
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--plot-sizes", type=int, nargs="+", metavar="N")
    parser.add_argument("--q-points", type=int, default=2001)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not 4 <= args.n_min <= args.n_max or args.n_step < 2:
        parser.error("require 4 <= n-min <= n-max and n-step >= 2")
    if any(n % 2 for n in (args.n_min, args.n_max, args.n_step)):
        parser.error("N bounds and step must be even")
    model = Model(args.J, args.gamma, args.Gamma, args.alpha, args.h)
    parameters = (args.J, args.gamma, args.Gamma, args.alpha, args.compare_alpha, args.h)
    if args.q_points < 2 or not np.isfinite(parameters).all():
        parser.error("require at least two q samples and finite model parameters")

    sizes = range(args.n_min, args.n_max + 1, args.n_step)
    plot_sizes = args.plot_sizes or [N for N in (32, 64, 128) if N in sizes] or [sizes[-1]]
    if not set(plot_sizes).issubset(sizes):
        parser.error("--plot-sizes must belong to the selected N scan")
    q = np.linspace(0, np.pi, args.q_points)
    q_inf, peaks, curves = scan_model(model, sizes, q, plot_sizes)
    scans = {model.alpha: (q_inf, peaks)}
    if args.compare_alpha != model.alpha:
        comparison = replace(model, alpha=args.compare_alpha)
        other_inf, other_peaks, _ = scan_model(comparison, sizes, q)
        scans[comparison.alpha] = (other_inf, other_peaks)

    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    })
    plot_q_curves(args.output, model, q, curves, peaks, q_inf)
    plot_size_curves(args.output, model.h, sizes, scans)
    write_peak_data(args.output / "q_peak_data.csv", scans)
    print(f"Saved four PNGs and q_peak_data.csv to {args.output.resolve()}")


if __name__ == "__main__":
    main()
