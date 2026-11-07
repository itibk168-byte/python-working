"""Plot direction-optimized thermodynamic QFI versus transverse field.

Install: python -m pip install numpy scipy matplotlib
Run:     python plot_thermo_qfi.py

The standalone script writes its CSV, PNG and SVG to OUTPUT_DIR.
"""

import csv
import os
from functools import lru_cache
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullFormatter
from scipy.special import roots_legendre

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(r"D:\python-code\python-working\XY-Gamma\QFI\qfi_S_q^n\picture")
CACHE_PATH = (Path(__file__).resolve().parent.parent / "qfi_cosine_thesis"
              / "qfi_cosine_thermo_scan.npz")
J = 1.0
GAMMA_XY = 0.6
GAMMA = 0.6
ALPHAS = (-0.50, -0.30, 0.30, 0.50)
Q = np.pi * np.arange(5) / 4
LABELS = (r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$")
STYLES = (
    ("#254C76", "-"), ("#7570B3", "--"), ("#25837C", "-."),
    ("#A47E27", (0, (1.5, 1.5))), ("#B63E30", "-"),
)
TOLERANCE = 0.01
BASE_RADII = (96, 192)


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


def evaluate(alpha, h, radii):
    components, divergence = qfi_scan_point(h, alpha, Q, radii, J, GAMMA_XY, GAMMA)
    return max_density(components), divergence


def cutoff_sensitive(values, divergent):
    discrepancy = abs(values[1] - values[0]) / np.maximum(1.0, abs(values[1]))
    return (~divergent) & (discrepancy > TOLERANCE)


def refinement_radii(h, alpha):
    if alpha > 0 and 1 < h <= 1.015:
        return 576, 768
    if alpha > 0 and 1 < h <= 1.025:
        return 384, 576
    return 192, 384


def field_grid(alpha):
    if alpha < 0:
        delta = (J * GAMMA_XY)**2 + 4 * alpha * GAMMA**2
        boundary = np.sqrt(J**2 - delta)
        extras = boundary + np.array([-0.003, -0.001, 0, 0.001, 0.003])
    else:
        extras = [1.015, 1.025, 1.035]
    return np.unique(np.r_[np.linspace(0, 2.5, 251), extras])


def initial_scan():
    if not CACHE_PATH.exists():
        return [{} for _ in ALPHAS]
    with np.load(CACHE_PATH) as data:
        cuts = np.argmin(abs(data["q"][:, None] - Q[None, :]), axis=0)
        if (not np.array_equal(data["alpha"], ALPHAS)
                or not np.array_equal(data["radii"], BASE_RADII)
                or not np.allclose(data["q"][cuts], Q)):
            raise ValueError("Cached scan has a different parameter or wavevector grid.")
        values = data["cutoff_estimates"][..., cuts]
        divergence = data["divergent"][..., cuts]
        return [dict(zip(data["h"], zip(a, d))) for a, d in zip(values, divergence)]


def scan_alpha(alpha, cached):
    rows = []
    for h in field_grid(alpha):
        radii = BASE_RADII
        values, divergent = cached[h] if h in cached else evaluate(alpha, h, radii)
        if np.any(cutoff_sensitive(values, divergent)):
            radii = refinement_radii(h, alpha)
            values, divergent = evaluate(alpha, h, radii)
        sensitive = cutoff_sensitive(values, divergent)
        for i, q in enumerate(Q):
            if divergent[i]:
                estimate, status = np.inf, "divergent"
            elif sensitive[i]:
                estimate, status = np.nan, "cutoff_sensitive"
            else:
                estimate, status = values[1, i], "finite_estimate"
            coarse, fine = values[:, i] if not divergent[i] else (None, None)
            rows.append((alpha, h, q, q/np.pi, estimate, status, *radii, coarse, fine))
    return rows


def draw(rows):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.2, 5.1))
    fig.subplots_adjust(left=0.114, right=0.984, top=0.81, bottom=0.153,
                        hspace=0.32, wspace=0.16)
    fig.suptitle(r"Thermodynamic-limit cosine QFI  |  $J=1,\ \gamma=\Gamma=0.6$",
                 fontsize=12, y=0.97)
    handles = [Line2D([], [], color=color, linestyle=style, label=label, linewidth=1.5)
               for (color, style), label in zip(STYLES, LABELS)]
    fig.legend(handles=handles, title=r"Wavevector $q$", ncols=5, loc="upper center",
               bbox_to_anchor=(0.53, 0.926), handlelength=2.5)
    for i, (alpha, panel, ax) in enumerate(zip(ALPHAS, rows, axes.flat)):
        h = np.array([row[1] for row in panel[::len(Q)]])
        values = np.array([row[4] for row in panel]).reshape(-1, len(Q))
        for curve, (color, style) in zip(values.T, STYLES):
            ax.plot(h, np.where(np.isfinite(curve), curve, np.nan), color=color,
                    linestyle=style, linewidth=1.5)
        if np.any(np.isinf(values[:, -1])):
            ax.axvspan(0, 1, color="#B63E30", alpha=0.085, linewidth=0, zorder=0)
            ax.text(0.2, 0.90, r"$q=\pi$ diverges", color="#922F25", ha="center",
                    va="top", transform=ax.transAxes, fontsize=8.5)
        ax.set_title(fr"$\bf{{{'abcd'[i]}}}$   $\alpha={alpha:+.2f}$", loc="left")
        ax.set(xlim=(0, 2.5), yscale="log", ylim=(0.24, 52))
        ax.set_yticks([0.3, 1, 3, 10, 30], labels=["0.3", "1", "3", "10", "30"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
    fig.supxlabel(r"Transverse field $h$", y=0.072)
    fig.supylabel(r"Optimized QFI density $\max_{\mathbf{n}} F_Q(S_q^{\mathbf{n}})/N$",
                  x=0.016)
    fig.text(0.53, 0.023, r"Shading marks divergence of the $q=\pi$ mode only.",
             ha="center", fontsize=7.4, color="#555555")
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"qfi_thermo_vs_h.{extension}", dpi=340)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [scan_alpha(alpha, cached) for alpha, cached in zip(ALPHAS, initial_scan())]
    with (OUTPUT_DIR / "qfi_thermo_vs_h.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha", "h", "q", "q_over_pi", "f_Q_max", "status",
                         "R_coarse", "R_fine", "f_R_coarse", "f_R_fine"])
        writer.writerows(row for panel in rows for row in panel)
    draw(rows)
    print(f"Saved CSV, PNG and SVG to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
