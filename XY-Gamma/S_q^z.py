import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters
J, gamma, Gamma = 1.0, 0.6, 0.6
N = 24000                       # Must be divisible by 24
STATE = "ground"                 # "ground" or "vacuum"
alphas = [-0.5, -0.3, 0.3, 0.5]
qs = [0, np.pi, np.pi/2, np.pi/4, np.pi/3]
q_labels = ["0", r"\pi", r"\pi/2", r"\pi/4", r"\pi/3"]
q_names = ["0", "pi", "pi_over_2", "pi_over_4", "pi_over_3"]

h_values = np.unique(np.r_[
    np.linspace(0, 2, 1001), np.sqrt(1.36), np.sqrt(1.072)
])
out = Path("xy_gamma_variance_figures")
out.mkdir(parents=True, exist_ok=True)

assert N > 0 and N % 24 == 0
k = -np.pi + (np.arange(N) + 0.5) * 2*np.pi/N
sin_k, cos_k = np.sin(k), np.cos(k)


def correlations(h, alpha):
    a = J * gamma
    b = Gamma * (1 + alpha)
    d = Gamma * (1 - alpha)
    xi = J * cos_k - h
    A = np.sqrt(xi**2 + (a*a + b*b) * sin_k**2)
    epsilon = 2*A + 2*d*sin_k

    n_eta = np.zeros(N)
    if STATE == "ground":
        n_eta = (epsilon < 0).astype(float)

        # Lowest-energy state in the even-parity sector
        if int(n_eta.sum()) % 2:
            index = np.argmin(np.abs(epsilon))
            n_eta[index] = 1 - n_eta[index]
    elif STATE != "vacuum":
        raise ValueError("STATE must be 'ground' or 'vacuum'.")

    # On this APBC grid, reversing the array maps k to -k.
    R = 1 - n_eta - n_eta[::-1]
    D = n_eta - n_eta[::-1]

    n_c = (1 + D - R*xi/A) / 2
    p_c = R*(-b + 1j*a)*sin_k / (2*A)
    return n_c, p_c


def variance(n_c, p_c, q):
    shift = int(round(q*N/(2*np.pi))) % N
    n_shift = np.roll(n_c, -shift)
    p_shift = np.roll(p_c, -shift)

    prefactor = 1.0 if (2*shift) % N == 0 else 0.5
    return prefactor * np.sum(
        n_c*(1 - n_shift) + np.real(np.conj(p_c)*p_shift)
    )


# results[alpha index, q index, h index]
results = np.empty((len(alphas), len(qs), len(h_values)))

for ia, alpha in enumerate(alphas):
    for ih, h in enumerate(h_values):
        n_c, p_c = correlations(h, alpha)
        for iq, q in enumerate(qs):
            results[ia, iq, ih] = variance(n_c, p_c, q)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"]
styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]


def save(fig, stem):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{stem}.{extension}",
                    dpi=320, bbox_inches="tight")
    plt.close(fig)


def format_axes(ax, title):
    ax.set(
        xlabel=r"$h/J$",
        ylabel=r"$\mathrm{Var}(S_q^z)$",
        title=title,
        xlim=(0, 2),
        ylim=(0, None),
    )
    ax.axvline(1, color="0.65", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2, fontsize=9)


# Five separate figures, each comparing four alpha values
for iq, label in enumerate(q_labels):
    fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
    for ia, alpha in enumerate(alphas):
        ax.plot(
            h_values, results[ia, iq],
            color=colors[ia], linestyle=styles[ia], linewidth=1.7,
            label=rf"$\alpha={alpha}$",
        )
    format_axes(ax, rf"$q={label},\quad N={N}$")
    save(fig, f"variance_q_{q_names[iq]}")

# Sixth figure: alpha=-0.5, comparing all five q values
fig, ax = plt.subplots(figsize=(6.6, 4.4), layout="constrained")
for iq, label in enumerate(q_labels):
    ax.plot(
        h_values, results[0, iq],
        color=colors[iq], linestyle=styles[iq], linewidth=1.7,
        label=rf"$q={label}$",
    )
format_axes(ax, rf"$\alpha=-0.5,\quad N={N}$")
save(fig, "variance_alpha_minus_0p5_all_q")

print(f"Six figures saved to: {out.resolve()}")