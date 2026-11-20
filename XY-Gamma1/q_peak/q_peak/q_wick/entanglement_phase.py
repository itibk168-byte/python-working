"""Plot the QFI-certified depth of a periodic XY-Gamma chain.

Requires NumPy and Matplotlib. Writes 05_entanglement_depth.png.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import repeat
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Model:
    J: float
    gamma: float
    Gamma: float
    alpha: float = 0.0
    h: float = 0.0


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


def ground_correlations(N, model):
    even = sector_ground_state(N, model, +1)
    odd = sector_ground_state(N, model, -1)
    _, G, P = even if even[0] <= odd[0] else odd
    return spin_correlations(G, P)


def certified_depth(C):
    """Depth certified by the cosine probe at the nQFI peak."""
    N = len(C)
    # Every off-grid normalized QFI matrix is a convex combination
    # of the matrices at the exact periodic momenta.
    xx, yy, xy, zz = np.fft.rfft(C, axis=0).real.T
    transverse = (xx + yy + np.hypot(xx - yy, 2 * xy)) / 2
    spectrum = np.maximum(transverse, zz)
    mode = np.argmax(spectrum)

    q = 2 * np.pi * mode / N
    weights = np.sort(abs(np.cos(q * np.arange(1, N + 1))))[::-1]
    reliable_qfi = (spectrum[mode] - 1e-9) * (weights @ weights)
    for k in range(1, N):
        # Sorting and grouping the absolute weights gives the tight bound.
        sums = np.add.reduceat(weights, np.arange(0, N, k))
        if reliable_qfi <= sums @ sums:
            return k
    return N


def depth_row(h, alphas, N, model):
    row = []
    for alpha in alphas:
        point = replace(model, alpha=alpha, h=h)
        row.append(certified_depth(ground_correlations(N, point)))
    return row


def plot_depth_map(output, model, N, points):
    alphas = np.linspace(-1, 1, points)
    fields = np.linspace(0, 2, points)
    with ProcessPoolExecutor(max_workers=min(8, points, os.cpu_count() or 1)) as pool:
        rows = pool.map(depth_row, fields, repeat(alphas), repeat(N), repeat(model))
        depth = np.array(list(rows))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
    })
    fig, ax = plt.subplots(figsize=(5.1, 3.7), layout="constrained")
    maximum = int(depth.max())
    norm = PowerNorm(gamma=0.5, vmin=0.5, vmax=maximum + 0.5)
    mesh = ax.pcolormesh(alphas, fields, depth, cmap="cividis", norm=norm, shading="nearest")
    ticks = [k for k in (1, 2, 5, 10, 20, 40, 60, 80) if norm(maximum) - norm(k) > 0.055]
    fig.colorbar(mesh, ax=ax, ticks=ticks + [maximum], label="Certified entanglement depth")

    alpha_line = np.linspace(-1, 1, 401)
    a_squared = (model.J * model.gamma)**2
    g_squared = model.Gamma**2
    if g_squared:
        alpha_c = -a_squared / (4 * g_squared)
        if -1 <= alpha_c <= 1:
            alpha_line = np.sort(np.append(alpha_line, alpha_c))
            ax.vlines(alpha_c, 0, abs(model.J), colors="white", linestyles="--", lw=1)
    h_c = np.sqrt(model.J**2 + np.maximum(-a_squared - 4 * alpha_line * g_squared, 0))
    ax.plot(alpha_line, h_c, "w--", lw=1)
    ax.set(xlabel=r"$\alpha$", ylabel=r"$h$", xlim=(-1, 1), ylim=(0, 2),
           title=rf"XY–$\Gamma$ chain, $N={N}$  ($J={model.J:g}$, "
                 rf"$\gamma={model.gamma:g}$, $\Gamma={model.Gamma:g}$)")
    fig.savefig(output / "05_entanglement_depth.png", dpi=450)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--Gamma", type=float, default=0.6)
    parser.add_argument("--N", type=int, default=100, help="even ring size")
    parser.add_argument("--phase-points", type=int, default=101, help="grid points per axis")
    parser.add_argument("--output", type=Path, default=Path("."), help="output directory")
    args = parser.parse_args()

    if args.N < 4 or args.N % 2:
        parser.error("--N must be even and at least 4")
    if args.phase_points < 2:
        parser.error("--phase-points must be at least 2")
    if not np.isfinite([args.J, args.gamma, args.Gamma]).all():
        parser.error("model parameters must be finite")

    model = Model(args.J, args.gamma, args.Gamma)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Computing a {args.phase_points} x {args.phase_points} grid at N={args.N}", flush=True)
    plot_depth_map(args.output, model, args.N, args.phase_points)
    print(f"Saved {args.output.resolve() / '05_entanglement_depth.png'}")


if __name__ == "__main__":
    main()
