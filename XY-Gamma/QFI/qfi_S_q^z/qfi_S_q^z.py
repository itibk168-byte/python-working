"""QFI density for S_q^z = sum_{j=1}^N cos(q*j) sigma_j^z / 2.

Install: python -m pip install numpy scipy matplotlib
Run:     python qfi_cosine_z_picture_dir.py

The script computes a periodic N=8 spin chain and its thermodynamic limit.
Plots and numerical data are saved in the output directory OUT.
"""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from functools import lru_cache
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.linalg import eigh
from scipy.special import roots_legendre

matplotlib.use("Agg")
import matplotlib.pyplot as plt


N = 8
J = 1.0
GAMMA_XY = 0.6
GAMMA = 0.6
ALPHAS = (-0.50, -0.30, 0.30, 0.50)
H_FINITE = np.linspace(-2.0, 2.0, 801)
Q_GRID = np.linspace(0.0, np.pi, 401)
Q_CUTS = np.pi * np.arange(5) / 4
Q_LABELS = (r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$")
STYLES = (
    ("#254C76", "-"), ("#7570B3", "--"), ("#25837C", "-."),
    ("#A47E27", (0, (1.5, 1.5))), ("#B63E30", "-"),
)
OUT = Path(r"D:\python-code\python-working\XY-Gamma\QFI\qfi_S_q^z\picture")


def spin_basis(n):
    basis = np.arange(2**n)
    bit_masks = 1 << np.arange(n)[:, None]
    z = 1 - 2 * ((basis & bit_masks) != 0)
    flips = basis ^ bit_masks
    parity = (-1)**((n - z.sum(axis=0)) // 2)
    return basis, z, flips, [np.flatnonzero(parity == p) for p in (1, -1)]


def zero_field_hamiltonian(basis, z, flips, alpha):
    h0 = np.zeros((len(basis), len(basis)), dtype=complex)
    for j in range(len(z)):
        jp = (j + 1) % len(z)
        h0[flips[j, flips[jp]], basis] += (
            0.5 * J * ((1 + GAMMA_XY) - (1 - GAMMA_XY) * z[j] * z[jp])
            + 1j * GAMMA * (z[jp] + alpha * z[j])
        )
    return h0


def parity_ground_state(blocks, field):
    solutions = []
    for h0_block, diagonal_z in blocks:
        hamiltonian = h0_block.copy()
        hamiltonian[np.diag_indices_from(hamiltonian)] += field * diagonal_z
        solutions.append(eigh(hamiltonian, subset_by_index=[0, 1], check_finite=False))
    # Compare both physical parity sectors. Select the even state at a tie.
    sector = int(solutions[1][0][0] < solutions[0][0][0] - 1e-11)
    energies, vectors = solutions[sector]
    gap = max(0.0, min(energies[1], solutions[1-sector][0][0]) - energies[0])
    return sector, vectors[:, 0], energies[0], gap


def scan_finite(alpha, fields, q_grid, n=N):
    basis, z, flips, sectors = spin_basis(n)
    h0 = zero_field_hamiltonian(basis, z, flips, alpha)
    blocks = [(h0[np.ix_(idx, idx)], z[:, idx].sum(axis=0)) for idx in sectors]
    weights = np.cos(q_grid[:, None] * np.arange(1, n + 1))
    weighted_z = [weights @ z[:, idx] for idx in sectors]
    density = np.empty((len(fields), len(q_grid)))
    energies, gaps = np.empty((2, len(fields)))
    parity = np.empty(len(fields), dtype=int)
    overlap = np.ones(len(fields))
    previous_sector, previous_state = None, None
    for i, h in enumerate(fields):
        sector, state, energies[i], gaps[i] = parity_ground_state(blocks, h)
        parity[i] = 1 - 2 * sector
        probability = abs(state)**2
        diagonal = weighted_z[sector]  # eigenvalues of 2 S_q^z in this sector
        centered = diagonal - (diagonal @ probability)[:, None]
        density[i] = centered**2 @ probability / n
        if previous_state is not None:
            overlap[i] = abs(np.vdot(previous_state, state))**2 if sector == previous_sector else 0
        previous_sector, previous_state = sector, state
    return density, energies, gaps, parity, overlap


@lru_cache
def quadrature(order):
    return roots_legendre(order)


def momentum_edges(h, alpha):
    """Resolve the Fermi points and narrow near-critical features."""
    anisotropy = np.hypot(J * GAMMA_XY, GAMMA * (1 + alpha))
    delta = (J * GAMMA_XY)**2 + 4 * alpha * GAMMA**2
    discriminant = delta * (h*h + delta - J*J)
    positive = []
    if delta < 0 and discriminant > 0:
        cosine = (J*h + np.array([-1, 1])*np.sqrt(discriminant)) / (J*J - delta)
        positive.extend(np.arccos(cosine[np.abs(cosine) < 1]))
    width = abs(h - J) / anisotropy
    if 0 < width < 0.05:
        positive.extend((width, 10*width))
    positive = np.asarray(positive)
    return np.unique(np.r_[-np.pi, -positive, 0., positive, np.pi])


def normal_and_pairing(k, h, alpha):
    """Return n(k) and the anomalous amplitude with its constant phase removed."""
    a = np.hypot(J * GAMMA_XY, GAMMA * (1 + alpha))
    xi = J * np.cos(k) - h
    pairing = a * np.sin(k)
    amplitude = np.hypot(xi, pairing)
    tilt = GAMMA * (1 - alpha) * np.sin(k)
    occupied_plus = (amplitude + tilt < 0).astype(float)
    occupied_minus = (amplitude - tilt < 0).astype(float)
    paired = 1 - occupied_plus - occupied_minus
    # A zero at a boundary has zero quadrature weight. The limiting values
    # inside each nonzero-width interval are used by Gauss-Legendre integration.
    ratio = np.divide(paired, amplitude, out=np.zeros_like(amplitude), where=amplitude > 0)
    normal = (1 + occupied_plus - occupied_minus - ratio*xi) / 2
    return normal, ratio*pairing / 2


def infinite_density(h, alpha, q_grid, order=64):
    """Integrate the longitudinal structure factor and apply the cosine weight."""
    edges = momentum_edges(h, alpha)
    q = q_grid[:, None]
    shifted = (edges[None, :] - q + np.pi) % (2*np.pi) - np.pi
    all_edges = np.sort(np.column_stack((np.broadcast_to(edges, shifted.shape), shifted)))
    midpoint = 0.5 * (all_edges[:, 1:] + all_edges[:, :-1])
    half_width = 0.5 * np.diff(all_edges, axis=1)
    nodes, weights = quadrature(order)
    k = midpoint[..., None] + half_width[..., None]*nodes
    n0, p0 = normal_and_pairing(k, h, alpha)
    nq, pq = normal_and_pairing(k + q[..., None], h, alpha)
    # Wick's theorem for sigma_z=1-2 c†c. The anomalous amplitudes have the
    # same constant phase, so Re[p(k) p(k+q)*] is the product below.
    integrand = n0*(1 - nq) + p0*pq
    structure = (2/np.pi) * np.sum(half_width * (integrand @ weights), axis=1)
    # For fixed q, the spatial mean of cos²(q*j) is 1 at 0 and pi, else 1/2.
    cosine_factor = np.where((q_grid == 0) | (q_grid == np.pi), 1., 0.5)
    return cosine_factor * structure


def limit_fields():
    delta = (J * GAMMA_XY)**2 + 4 * GAMMA**2 * np.array(ALPHAS)
    critical = np.r_[J, np.sqrt(J**2 - delta[delta < 0])]
    offsets = np.array([-0.002, -0.0005, 0., 0.0005, 0.002])
    return np.unique(np.r_[np.linspace(0.0, 2.5, 251), (critical[:, None] + offsets).ravel()])


def save_figure(figure, name):
    for extension in ("png", "svg"):
        figure.savefig(OUT / f"{name}.{extension}", dpi=320)
    plt.close(figure)


def plot_curves(fields, density, cut_indices, title, name, overlap=None):
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.10, top=0.79,
                        hspace=0.35, wspace=0.15)
    fig.suptitle(title, fontsize=13, y=0.975)
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=1.6, label=label)
               for (color, style), label in zip(STYLES, Q_LABELS)]
    fig.legend(handles=handles, title=r"Wavevector $q$", loc="upper center",
               bbox_to_anchor=(0.5, 0.934), ncols=5, handlelength=2.5)
    for i, ax in enumerate(axes.flat):
        breaks = [] if overlap is None else np.flatnonzero(overlap[i] < 0.5)
        segments = np.split(np.arange(len(fields)), breaks)
        for index, (color, style) in zip(cut_indices, STYLES):
            for segment in segments:
                ax.plot(fields[segment], density[i, segment, index], color=color,
                        linestyle=style, linewidth=1.5)
        ax.set_title(fr"$\bf{{{'abcd'[i]}}}$   $\alpha={ALPHAS[i]:+.2f}$", loc="left")
        ax.set(xlim=(fields[0], fields[-1]), ylim=(0, 1.07*density.max()))
        ax.xaxis.set_major_locator(MultipleLocator(1.0 if fields[0] < 0 else 0.5))
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
        ax.set_axisbelow(True)
    fig.supxlabel(r"Transverse field $h$", y=0.025)
    fig.supylabel(r"QFI density $f_Q(S_q^z)$", x=0.018)
    save_figure(fig, name)


def plot_map(density, title):
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.0), sharex=True, sharey=True,
                            layout="constrained")
    fig.suptitle(title, fontsize=13)
    for i, ax in enumerate(axes.flat):
        mesh = ax.pcolormesh(H_FINITE, Q_GRID/np.pi, density[i].T, cmap="viridis",
                             vmin=0, vmax=density.max(), shading="nearest", rasterized=True)
        ax.set_title(fr"$\bf{{{'abcd'[i]}}}$   $\alpha={ALPHAS[i]:+.2f}$", loc="left")
        ax.set(xlim=(-2, 2), ylim=(0, 1))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
    fig.colorbar(mesh, ax=axes, label=r"QFI density $f_Q(S_q^z)$", pad=0.025)
    fig.supxlabel(r"Transverse field $h$")
    fig.supylabel(r"Wavevector $q/\pi$")
    save_figure(fig, "qfi_cosine_z_q_scan")


def save_curves(fields, density, cut_indices, filename):
    alpha, field, wavevector = np.meshgrid(ALPHAS, fields, Q_GRID[cut_indices]/np.pi,
                                         indexing="ij")
    data = np.column_stack([column.ravel() for column in
                            (alpha, field, wavevector, density[:, :, cut_indices])])
    np.savetxt(OUT / filename, data, delimiter=",", header="alpha,h,q_over_pi,f_Q_z",
               comments="", fmt="%.12g")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False,
    })
    scans = [scan_finite(alpha, H_FINITE, Q_GRID) for alpha in ALPHAS]
    finite, energy, gap, parity, overlap = map(np.stack, zip(*scans))
    h_limit = limit_fields()
    limit = np.array([[infinite_density(h, alpha, Q_GRID) for h in h_limit]
                      for alpha in ALPHAS])
    np.savez_compressed(OUT / "qfi_cosine_z_scan.npz", alpha=ALPHAS, q=Q_GRID,
                        N=N, h_finite=H_FINITE, h_limit=h_limit, f_finite=finite,
                        f_limit=limit, energy=energy, gap=gap, parity=parity)
    cut_indices = np.argmin(abs(Q_GRID[None, :] - Q_CUTS[:, None]), axis=1)
    save_curves(H_FINITE, finite, cut_indices, "qfi_cosine_z_N8.csv")
    save_curves(h_limit, limit, cut_indices, "qfi_cosine_z_limit.csv")

    model = rf"$J={J:g},\ \gamma={GAMMA_XY:g},\ \Gamma={GAMMA:g}$"
    title_finite = rf"Longitudinal cosine QFI   |   $N={N}$, " + model
    plot_curves(H_FINITE, finite, cut_indices, title_finite, "qfi_cosine_z_vs_h", overlap)
    plot_curves(h_limit, limit, cut_indices, "Thermodynamic limit   |   " + model,
                "qfi_cosine_z_limit_vs_h")
    plot_map(finite, title_finite)


if __name__ == "__main__":
    main()
