"""Direction-optimized cosine-weighted QFI of the periodic XY–Gamma chain.
S_q^n = sum_{j=1}^N \frac12 cos{(qj)}sigma_j^n

Install: python -m pip install numpy scipy matplotlib
Run:     python qfi_cosine_thesis_picture_dir.py
Optional infinite-chain estimate: python qfi_cosine_thesis_picture_dir.py --thermo

The script saves PNG and SVG to PICTURE_DIR and numerical data beside itself.
The spin operators
are S_q^a = sum_{j=1}^N cos(q*j) sigma_j^a / 2, with no rescaling of the weights.
"""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
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
OUT = Path(__file__).resolve().parent
PICTURE_DIR = Path('D:\\python-code\\python-working\\XY-Gamma\\QFI\\qfi_S_q^n\\picture')
DEGENERACY_TOL = 1e-11
H_GRID = np.linspace(-2.0, 2.0, 801)
Q_GRID = np.linspace(0.0, np.pi, 401)
Q_CUTS = np.pi * np.arange(5) / 4
Q_LABELS = (r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$")
STYLES = (
    ("#254C76", "-"), ("#7570B3", "--"), ("#25837C", "-."),
    ("#A47E27", (0, (1.5, 1.5))), ("#B63E30", "-"),
)


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
        two_flips = flips[j, flips[jp]]
        h0[two_flips, basis] += (
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
    # Use the lower-energy physical parity sector. Choose even at ties.
    sector = int(solutions[1][0][0] < solutions[0][0][0] - DEGENERACY_TOL)
    energies, vectors = solutions[sector]
    gap = max(0.0, min(energies[1], solutions[1-sector][0][0]) - energies[0])
    return sector, vectors[:, 0], energies[0], gap


def qfi_matrix(psi, z, flips, weight_products):
    """Return 4 Cov(S_q^a,S_q^b)/N for every q, as real 3x3 matrices."""
    flipped = psi[flips]
    local = np.stack((flipped, -1j * z * flipped, z * psi))
    # These are centered Pauli operators acting on |GS>. Using sigma rather
    # than sigma/2 here absorbs the factor of 4 in the pure-state QFI.
    local -= (local @ psi.conj())[..., None] * psi
    n = len(z)
    centered = local.reshape(3*n, -1)
    pairs = (centered.conj() @ centered.T).real.reshape(3, n, 3, n)
    pairs = pairs.transpose(1, 3, 0, 2).reshape(n*n, 9)
    return (weight_products @ pairs).reshape(-1, 3, 3)


def optimize_direction(matrices):
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    axes = eigenvectors[..., -1]
    # n and -n give the same QFI. Pick a reproducible sign for export.
    pivot = np.argmax(abs(axes), axis=-1, keepdims=True)
    axes *= np.sign(np.take_along_axis(axes, pivot, axis=-1))
    return np.maximum(eigenvalues[..., -1], 0.0), axes


def scan(alpha, fields, q_grid, n=N):
    basis, z, flips, sectors = spin_basis(n)
    h0 = zero_field_hamiltonian(basis, z, flips, alpha)
    blocks = [(h0[np.ix_(idx, idx)], z[:, idx].sum(axis=0)) for idx in sectors]
    weights = np.cos(q_grid[:, None] * np.arange(1, n + 1))
    weight_products = (weights[:, :, None] * weights[:, None, :]).reshape(-1, n*n) / n
    matrices = np.empty((len(fields), len(q_grid), 3, 3))
    gaps, energies, parities = np.empty((3, len(fields)))
    overlaps = np.ones(len(fields))
    previous = None
    for index, h in enumerate(fields):
        sector, state, energies[index], gaps[index] = parity_ground_state(blocks, h)
        parities[index] = 1 - 2 * sector
        psi = np.zeros(len(basis), dtype=complex)
        psi[sectors[sector]] = state
        if previous is not None:
            overlaps[index] = abs(np.vdot(previous, psi))**2
        matrices[index] = qfi_matrix(psi, z, flips, weight_products)
        previous = psi
    density, axes = optimize_direction(matrices)
    return density, axes, matrices, energies, gaps, parities.astype(int), overlaps


def plot_scan(density, overlaps, cut_indices):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False,
    })
    title = (r"Direction-optimized cosine QFI"
             rf"   |   $N={N},\ J={J:g},\ \gamma=\Gamma={GAMMA:g}$")
    labels = "abcd"
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.10, top=0.79,
                        hspace=0.35, wspace=0.15)
    fig.suptitle(title, fontsize=13, y=0.975)
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=1.6,
                      label=label) for (color, style), label in zip(STYLES, Q_LABELS)]
    fig.legend(handles=handles, title=r"Wavevector $q$", loc="upper center",
               bbox_to_anchor=(0.5, 0.934), ncols=5, handlelength=2.5)
    for i, ax in enumerate(axes.flat):
        breaks = np.flatnonzero(overlaps[i] < 0.5)
        segments = np.split(np.arange(len(H_GRID)), breaks)
        for index, (color, style) in zip(cut_indices, STYLES):
            for segment in segments:
                ax.plot(H_GRID[segment], density[i, segment, index], color=color,
                        linestyle=style, linewidth=1.5)
        ax.set_title(fr"$\bf{{{labels[i]}}}$   $\alpha={ALPHAS[i]:+.2f}$", loc="left")
        ax.set(xlim=(-2, 2), ylim=(0, 1.06*density.max()))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
        ax.set_axisbelow(True)
    fig.supxlabel(r"Transverse field $h$", y=0.025)
    fig.supylabel(r"Optimized QFI density $f_Q^{\rm max}$", x=0.015)
    save_figure(fig, "qfi_cosine_vs_h")

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.0), sharex=True, sharey=True,
                            layout="constrained")
    fig.suptitle(title, fontsize=13)
    for i, ax in enumerate(axes.flat):
        mesh = ax.pcolormesh(H_GRID, Q_GRID/np.pi, density[i].T, cmap="viridis",
                             vmin=0, vmax=density.max(), shading="nearest", rasterized=True)
        ax.set_title(fr"$\bf{{{labels[i]}}}$   $\alpha={ALPHAS[i]:+.2f}$", loc="left")
        ax.set(xlim=(-2, 2), ylim=(0, 1))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
    fig.colorbar(mesh, ax=axes, label=r"Optimized QFI density $f_Q^{\rm max}$", pad=0.025)
    fig.supxlabel(r"Transverse field $h$")
    fig.supylabel(r"Wavevector $q/\pi$")
    save_figure(fig, "qfi_cosine_q_scan")


def save_figure(figure, name):
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(PICTURE_DIR / f"{name}.{extension}", dpi=320)
    plt.close(figure)


# Infinite-chain correlation functions used by --thermo.


@lru_cache
def quadrature(order):
    return roots_legendre(order)


def majorana_blocks(h, alpha, radius, J, gamma, Gamma):
    """Compute M_ab(r)=i<w_(0,a) w_(r,b)> off the diagonal."""
    d = (J * gamma)**2 + 4 * alpha * Gamma**2
    a = np.hypot(J * gamma, Gamma * (1 + alpha))
    boundaries = [0., np.pi]
    discriminant = d * (h*h + d - J*J)
    if d < 0 and discriminant >= 0:
        roots = (J*h + np.sqrt(discriminant) * np.array([-1., 1.])) / (J*J - d)
        boundaries.extend(np.arccos(roots[abs(roots) < 1]))
    width = abs(h - J) / a
    if 0 < width < 0.05:
        boundaries.extend([width, 10 * width])
    boundaries = np.unique(boundaries)

    onsite = np.array([[0., -2*h], [2*h, 0.]])
    bond = np.array([[2*alpha*Gamma, J*(1-gamma)], [-J*(1+gamma), -2*Gamma]])
    nodes, weights = quadrature(max(192, radius+32))
    distances = np.arange(radius + 1)
    blocks = np.zeros((radius+1, 2, 2))
    fz_uniform = 0.
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        midpoint, half_width = (left+right)/2, (right-left)/2
        k = midpoint + half_width*nodes
        dk = half_width*weights
        phase = np.exp(1j*k[:, None, None])
        kernel = 1j*(onsite + bond*phase - bond.T*phase.conj())
        energies, modes = np.linalg.eigh(kernel)
        sign = (modes*np.sign(energies)[:, None, :]) @ modes.conj().swapaxes(1, 2)
        transform = np.exp(-1j*distances[:, None]*k)*(dk/np.pi)
        blocks += (transform @ (1j*sign.reshape(-1, 4))).real.reshape(-1, 2, 2)
        paired = energies[:, 0] * energies[:, 1] < 0
        numerator = (a*np.sin(k))**2
        fz_uniform += (2/np.pi)*np.dot(dk, paired*numerator/(numerator+(J*np.cos(k)-h)**2))
    return blocks, fz_uniform


def pfaffian(matrix):
    """Evaluate a small skew block using partial pivoting."""
    work = matrix.copy()
    result = 1.
    for i in range(0, len(work), 2):
        j = i+1+np.argmax(abs(work[i, i+1:]))
        if work[i, j] == 0:
            return 0.
        if j != i+1:
            work[[i+1, j]] = work[[j, i+1]]
            work[:, [i+1, j]] = work[:, [j, i+1]]
            result = -result
        pivot = work[i, i+1]
        result *= pivot
        x, y = work[i, i+2:], work[i+1, i+2:]
        work[i+2:, i+2:] += (np.outer(y, x)-np.outer(x, y))/pivot
    return result


def leading_pfaffians(work):
    """Reuse skew elimination for the Pfaffian at each even leading order."""
    size = len(work)
    values = np.zeros(size//2)
    prefix = 1.
    start = 0
    while start < size:
        stop = start+2
        pivot = work[start, start+1]
        values[stop//2-1] = prefix*pivot
        if abs(pivot) < 1e-8:
            if np.max(abs(work[start:, start:])) < 1e-14:
                break
            while abs(pivot) < 1e-8 and stop < size:
                stop += 2
                pivot = pfaffian(work[start:stop, start:stop])
                values[stop//2-1] = prefix*pivot
            if stop < size:
                cross = work[start:stop, stop:]
                work[stop:, stop:] += cross.T @ np.linalg.solve(work[start:stop, start:stop], cross)
        elif stop < size:
            x, y = work[start, stop:], work[start+1, stop:]
            work[stop:, stop:] += (np.outer(y, x)-np.outer(x, y))/pivot
        prefix *= pivot
        start = stop
    return values


def transverse_correlation(blocks, angle):
    """Return <(cos(angle) X + sin(angle) Y)_0 (same spin)_r> for r>=1."""
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, s], [-s, c]])
    blocks = rotation @ blocks @ rotation.T
    radius = len(blocks)-1
    # The first 2r Majoranas are V_0,U_1,V_1,...,U_r.
    indices = np.arange(1, 2*radius+1)
    sites, species = indices//2, indices%2
    displacement = sites[None, :] - sites[:, None]
    upper = np.triu(blocks[abs(displacement), species[:, None], species[None, :]], 1)
    return (-1.)**np.arange(1, radius+1) * leading_pfaffians(upper-upper.T)


def correlation_components(h, alpha, radius, J, gamma, Gamma):
    """Return (XX,YY,XY,ZZ) symmetrized connected Pauli correlators."""
    blocks, fz_uniform = majorana_blocks(h, alpha, radius, J, gamma, Gamma)
    correlations = np.zeros((radius+1, 4))
    correlations[0] = [1., 1., 0., 1.-blocks[0, 0, 1]**2]
    correlations[1:, 3] = -np.linalg.det(blocks[1:])
    d = (J*gamma)**2 + 4*alpha*Gamma**2
    has_fermi_sea = d < 0 and h*h < J*J-d
    order = np.zeros(4)
    if has_fermi_sea:
        xx = transverse_correlation(blocks, 0.)
        yy = transverse_correlation(blocks, np.pi/2)
        plus = transverse_correlation(blocks, np.pi/4)
        correlations[1:, :3] = np.column_stack((xx, yy, plus-(xx+yy)/2))
    else:
        # With no blocked modes, the state is a rotated ordinary XY ground state.
        angle = 0.5*np.arctan2(Gamma*(1+alpha), J*gamma)
        c, s = np.cos(angle), np.sin(angle)
        easy = transverse_correlation(blocks, angle)
        hard = transverse_correlation(blocks, angle+np.pi/2)
        easy_axis = np.array([c*c, s*s, c*s])
        hard_axis = np.array([s*s, c*c, -c*s])
        correlations[1:, :3] = easy[:, None]*easy_axis + hard[:, None]*hard_axis
        if h < J:
            effective_gamma = np.hypot(J*gamma, Gamma*(1+alpha))/J
            # Staggered Pauli order from Barouch and McCoy, Phys. Rev. A 3, 786,
            # Eq. (4.1), including the conversion from spin to Pauli normalization.
            amplitude = 2*(effective_gamma**2*(1-(h/J)**2))**0.25/(1+effective_gamma)
            order[:3] = amplitude*easy_axis
    divergent_pi = not has_fermi_sea and h <= J
    return correlations, fz_uniform, order, divergent_pi


def qfi_scan_point(h, alpha, q, radii=(96, 192), J=1., gamma=.6, Gamma=.6):
    """Return windowed covariance components (R,q,4) and known divergence mask.

    The components are (xx,yy,xy,zz) for the F_Q/N matrix. The omitted
    xz and yz entries vanish in a parity eigenstate. Finite-range estimates
    can remain cutoff-sensitive at incommensurate gapless momenta.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    radius = max(radii)
    correlations, fz, order, divergent_pi = correlation_components(
        abs(h), alpha, radius, J, gamma, Gamma,
    )
    is_zero, is_pi = q == 0., q == np.pi
    endpoint_scale = np.where(is_zero | is_pi, 1., .5)
    divergent = is_pi & divergent_pi
    # Remove the explicit staggered Bragg peak. Away from pi it contributes
    # zero to the Fourier density, including r=0 of its formal continuation.
    r = np.arange(radius+1)
    regular = correlations - (-1.)**r[:, None]*order
    cos_kernel = np.cos(q[:, None]*r[None, 1:])
    values = []
    for cutoff in radii:
        t = np.maximum(0., 2*r[1:cutoff+1]/cutoff-1)
        window = (1-t)**3*(1+3*t+6*t*t)
        result = endpoint_scale[:, None] * (
            regular[0] + 2*(cos_kernel[:, :cutoff]*window)@regular[1:cutoff+1]
        )
        result[is_zero, 3] = fz
        values.append(result)
    return np.array(values), divergent


def max_density(components):
    """Largest eigenvalue of the covariance matrix (xx,yy,xy,zz)."""
    xx, yy, xy, zz = np.moveaxis(components, -1, 0)
    return np.maximum(zz, 0.5*(xx+yy+np.hypot(xx-yy, 2*xy)))


THERMO_RADII = (96, 192)


def scan_infinite(alpha):
    fields = np.linspace(0.0, 2.5, 251)
    density = np.empty((len(fields), len(THERMO_RADII), len(Q_GRID)))
    divergent = np.empty((len(fields), len(Q_GRID)), dtype=bool)
    for index, h in enumerate(fields):
        components, divergent[index] = qfi_scan_point(
            h, alpha, Q_GRID, THERMO_RADII, J, GAMMA_XY, GAMMA,
        )
        density[index] = max_density(components)
    return density, divergent


def plot_infinite(cut_indices):
    fields = np.linspace(0.0, 2.5, 251)
    with ProcessPoolExecutor(max_workers=len(ALPHAS)) as pool:
        density, divergent = map(np.stack, zip(*pool.map(scan_infinite, ALPHAS)))
    coarse, fine = np.moveaxis(density, 2, 0)
    # This is a cutoff diagnostic, not a certified error bound. Leave points
    # blank when the two truncations differ substantially.
    sensitive = abs(fine - coarse) > 0.02 * np.maximum(1.0, abs(fine))
    estimate = np.where(divergent, np.inf, np.where(sensitive, np.nan, fine))
    np.savez_compressed(OUT / "qfi_cosine_thermo_scan.npz", alpha=ALPHAS, h=fields,
                        q=Q_GRID, radii=THERMO_RADII, cutoff_estimates=density,
                        estimate=estimate, divergent=divergent, cutoff_sensitive=sensitive)

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.13, top=0.79,
                        hspace=0.33, wspace=0.15)
    fig.suptitle(r"Thermodynamic-limit estimates   |   $J=1,\ \gamma=\Gamma=0.6$",
                 y=0.975, fontsize=13)
    handles = [Line2D([], [], color=color, linestyle=style, linewidth=1.6, label=label)
               for (color, style), label in zip(STYLES, Q_LABELS)]
    fig.legend(handles=handles, title=r"Wavevector $q$", loc="upper center",
               bbox_to_anchor=(0.5, 0.934), ncols=5, handlelength=2.5)
    for i, ax in enumerate(axes.flat):
        for index, (color, style) in zip(cut_indices, STYLES):
            values = estimate[i, :, index]
            ax.plot(fields, np.where(np.isfinite(values), values, np.nan), color=color,
                    linestyle=style, linewidth=1.5)
        divergent_fields = fields[divergent[i, :, -1]]
        if divergent_fields.size:
            ax.axvspan(divergent_fields.min(), divergent_fields.max(), color="#B63E30",
                       alpha=0.10, linewidth=0)
            ax.text(0.18, 0.88, r"$q=\pi:\ f_Q^{\rm max}\to\infty$", color="#912D22",
                    transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.set_title(fr"$\alpha={ALPHAS[i]:+.2f}$", loc="left")
        ax.set(xlim=(0, 2.5), yscale="log")
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
        ax.set_axisbelow(True)
    fig.supxlabel(r"Transverse field $h$", y=0.055)
    fig.supylabel(r"Optimized QFI density $f_Q^{\rm max}$", x=0.015)
    fig.text(0.5, 0.015, "Cutoff-sensitive points are omitted.", ha="center", fontsize=8)
    save_figure(fig, "qfi_cosine_thermo_vs_h")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermo", action="store_true", help="Also estimate the infinite-chain curves.")
    args = parser.parse_args()
    scans = [scan(alpha, H_GRID, Q_GRID) for alpha in ALPHAS]
    density, direction, matrix, energy, gap, parity, overlap = map(np.stack, zip(*scans))
    np.savez_compressed(OUT / "qfi_cosine_scan.npz", alpha=ALPHAS, h=H_GRID, q=Q_GRID,
                        N=N, density=density, direction=direction, matrix=matrix,
                        energy=energy, gap=gap, parity=parity, overlap=overlap)
    cut_indices = np.argmin(abs(Q_GRID[None, :] - Q_CUTS[:, None]), axis=1)
    columns = []
    for i, alpha in enumerate(ALPHAS):
        for k in cut_indices:
            columns.append(np.column_stack((np.full_like(H_GRID, alpha), H_GRID,
                            np.full_like(H_GRID, Q_GRID[k]/np.pi), density[i, :, k],
                            direction[i, :, k], gap[i], parity[i])))
    np.savetxt(OUT / "qfi_cosine_curves.csv", np.vstack(columns), delimiter=",",
               header="alpha,h,q_over_pi,f_Q_max,n_x,n_y,n_z,gap,parity", comments="",
               fmt="%.12g")
    plot_scan(density, overlap, cut_indices)
    if args.thermo:
        plot_infinite(cut_indices)


if __name__ == "__main__":
    main()
