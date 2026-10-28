from pathlib import Path
from functools import lru_cache
import csv
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kitaev_qfi_mpl"))
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DELTAS = (0.3, 0.6, 1.2)
OUT = Path(r"D:\软件\PyCharm warehouse\python-working\kiteav_supercurrent\assets\kiteav_QFI")
OUT.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def gauss_rule(order):
    return leggauss(order)


def full_support(q, mu_over_t, delta_over_t):
    """Exact integral when every nonzero-energy momentum pair contributes."""
    q, r = np.broadcast_arrays(np.asarray(q, float), np.asarray(mu_over_t, float))
    a, m, d = np.abs(np.cos(q)), np.abs(r) / 2, float(delta_over_t)
    if d <= 0 or not np.isfinite(d):
        raise ValueError("Require finite Delta_0/t > 0.")
    plateau = m <= a
    # Evaluate the outside expression only in its domain.
    s = np.sqrt(np.maximum(m*m + d*d - a*a, np.finfo(float).tiny))
    denominator = s * (s + m)
    outside = np.divide(2*d*d, denominator, out=np.zeros_like(s), where=~plateau)
    return np.where(plateau, 2*d/(a+d), outside)


def qfi_density(q, mu_over_t, delta_over_t, order=128):
    """Bulk f_Q; exact full-support part and split Gaussian quadrature otherwise.

    E_+=h0+R, E_-=h0-R. Only R>|h0| contributes.
    With x=cos(k), the excluded interval has roots of
    (1-d^2)x^2-r*cos(Q)*x+r^2/4+d^2-sin(Q)^2=0.
    Isolated equality points follow the full-support limiting convention.
    """
    q, r = np.broadcast_arrays(np.asarray(q, float), np.asarray(mu_over_t, float))
    shape = q.shape
    q, r = q.ravel(), r.ravel()
    d = float(delta_over_t)
    result = np.asarray(full_support(q, r, d)).copy()
    tol = 1e-13
    if d < 1:
        active = (np.abs(np.sin(q)) > d + tol) & (np.abs(r) < 2*np.sqrt(1-d*d)-tol)
        indices = np.flatnonzero(active)
        nodes, weights = gauss_rule(order)
        for start in range(0, len(indices), 2048):
            ii = indices[start:start+2048]
            cq, sq, rr = np.cos(q[ii]), np.sin(q[ii]), r[ii]
            disc = np.maximum((sq*sq-d*d)*(4*(1-d*d)-rr*rr), 0)
            root = np.sqrt(disc)
            xminus = np.clip((rr*cq-root)/(2*(1-d*d)), -1, 1)
            xplus = np.clip((rr*cq+root)/(2*(1-d*d)), -1, 1)
            for x in (xminus, xplus):
                x[np.abs(x-1) < 1e-14] = 1
                x[np.abs(x+1) < 1e-14] = -1
            k1, k2 = np.arccos(xplus), np.arccos(xminus)
            integrals = np.zeros(len(ii))
            # Integrate only [0,k1] and [k2,pi]; no discontinuous step is sampled.
            for lo, hi in ((np.zeros_like(k1), k1), (k2, np.full_like(k2, np.pi))):
                half = (hi-lo)/2
                k = (lo+hi)[:, None]/2 + half[:, None]*nodes
                numerator = (d*np.sin(k))**2
                denominator = (cq[:, None]*np.cos(k)-rr[:, None]/2)**2 + numerator
                g = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator>0)
                integrals += half*(g @ weights)
            result[ii] = 2*integrals/np.pi
    # This exceptional point has an extensive zero-energy degeneracy.
    if abs(d-1) < tol:
        exceptional = (np.abs(r)<tol) & (np.abs(np.cos(q))<tol)
        result[exceptional] = np.nan
    return result.reshape(shape)


def reference_integral(q, r, d):
    """Independent adaptive integral, with zero-energy roots as breakpoints."""
    c, s = np.cos(q), np.sin(q)
    coefficients = [1-d*d, -r*c, r*r/4+d*d-s*s]
    if abs(coefficients[0]) < 1e-13:
        roots = [] if abs(coefficients[1]) < 1e-13 else [-coefficients[2]/coefficients[1]]
    else:
        roots = np.roots(coefficients)
    cuts = [0.0, np.pi]
    for x in roots:
        if abs(np.imag(x)) < 1e-10 and -1 < np.real(x) < 1:
            cuts.append(float(np.arccos(np.real(x))))
    cuts = sorted(cuts)
    def integrand(k):
        numerator = (d*np.sin(k))**2
        xi = c*np.cos(k)-r/2
        radius2 = xi*xi+numerator
        return numerator/radius2 if radius2 > s*s*np.sin(k)**2 and radius2>0 else 0.0
    return 2/np.pi*sum(quad(integrand, lo, hi, epsabs=2e-11, epsrel=2e-11)[0]
                       for lo, hi in zip(cuts[:-1], cuts[1:]))


def verify():
    rng = np.random.default_rng(20261026)
    checks = []
    for d in DELTAS:
        for q, r in zip(rng.uniform(-np.pi,np.pi,32), rng.uniform(-3,3,32)):
            value = float(qfi_density(q,r,d))
            reference = reference_integral(q,r,d)
            checks.append(abs(value-reference))
    assert max(checks) < 2e-8, max(checks)
    q = rng.uniform(-np.pi,np.pi,300)
    r = rng.uniform(-3,3,300)
    symmetry = 0.0
    for d in DELTAS:
        f = qfi_density(q,r,d)
        assert np.all((f >= -1e-12) & (f <= 2+1e-12))
        symmetry = max(symmetry, float(np.max(np.abs(f-qfi_density(-q,r,d)))),
                       float(np.max(np.abs(f-qfi_density(q,-r,d)))),
                       float(np.max(np.abs(f-qfi_density(q+np.pi,r,d)))))
        assert abs(float(qfi_density(0,0,d))-2*d/(1+d)) < 1e-12
    assert symmetry < 1e-10
    assert abs(float(qfi_density(np.pi/2,0,0.3))) < 1e-12
    assert abs(float(qfi_density(np.pi/2,0,0.6))) < 1e-12
    assert abs(float(qfi_density(np.pi/2,0,1.2))-2) < 1e-12
    # Convergence of the split quadrature near phase boundaries.
    convergence = 0.0
    for d in (0.3,0.6):
        q = np.linspace(np.arcsin(d)+1e-6,np.pi/2-1e-6,81)
        r = np.linspace(0,2*np.sqrt(1-d*d)-1e-6,79)
        qq,rr = np.meshgrid(q,r)
        convergence = max(convergence,float(np.max(np.abs(qfi_density(qq,rr,d,128)-qfi_density(qq,rr,d,256)))))
    assert convergence < 2e-7, convergence
    report = {"adaptive_reference_checks":len(checks),"max_absolute_error":max(checks),
              "max_symmetry_error":symmetry,"quadrature_128_vs_256_max_error":convergence,
              "definition":"F_Q=4 Var(N_f), density=F_Q/N, pure ground state",
              "boundary":"thermodynamic bulk; finite OBC independently checked",
              "delta_over_t":list(DELTAS)}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def style():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":7,"axes.labelsize":8,"axes.titlesize":8,"axes.linewidth":0.65,
        "axes.spines.top":False,"axes.spines.right":False,"xtick.direction":"out",
        "ytick.direction":"out","legend.frameon":False,"legend.fontsize":7,
        "svg.fonttype":"none","pdf.fonttype":42,"lines.linewidth":1.35,
        "savefig.facecolor":"white"})


def save(fig, name):
    fig.savefig(OUT/(name+".png"),dpi=300)
    fig.savefig(OUT/(name+".svg"))
    fig.savefig(OUT/(name+".pdf"))
    plt.close(fig)


def q_ticks(ax, full=False):
    if full:
        ax.set_xticks(np.linspace(-np.pi,np.pi,5),[r"$-\pi$",r"$-\pi/2$","0",r"$\pi/2$",r"$\pi$"])
    else:
        ax.set_xticks(np.linspace(-np.pi/2,np.pi/2,5),[r"$-\pi/2$",r"$-\pi/4$","0",r"$\pi/4$",r"$\pi/2$"])
    ax.set_xlabel(r"$Q$")


def make_figures():
    style()
    q = np.linspace(-np.pi,np.pi,721)
    r = np.linspace(-3,3,601)
    qq,rr = np.meshgrid(q,r)
    grids = np.stack([qfi_density(qq,rr,d) for d in DELTAS])
    np.savez_compressed(OUT/"qfi_density_grid.npz",Q=q,mu_over_t=r,delta_over_t=DELTAS,f_Q=grids)
    for d,grid in zip(DELTAS,grids):
        np.savetxt(OUT/f"qfi_grid_delta_{d:.1f}.csv",np.column_stack((r,grid)),delimiter=",",
                   header="mu_over_t,"+",".join(f"Q={v:.12g}" for v in q),comments="",fmt="%.10g")
    fig,axes = plt.subplots(1,3,figsize=(7.205,2.85),sharex=True,sharey=True)
    fig.subplots_adjust(left=.065,right=.895,bottom=.19,top=.86,wspace=.16)
    for i,(ax,d,grid) in enumerate(zip(axes,DELTAS,grids)):
        im=ax.pcolormesh(q,r,grid,cmap="magma",vmin=0,vmax=2,shading="auto",rasterized=True)
        ax.set_title(rf"$\Delta_0/t={d}$",pad=7)
        ax.text(-.11,1.10,"abc"[i],transform=ax.transAxes,weight="bold",fontsize=9)
        q_ticks(ax,full=True)
        ax.set_yticks([-3,-2,-1,0,1,2,3])
        ax.set_xlim(-np.pi,np.pi)
        ax.set_ylim(-3,3)
    axes[0].set_ylabel(r"$\mu/t$")
    cax=fig.add_axes([.922,.19,.016,.67])
    fig.colorbar(im,cax=cax,label=r"$f_Q=F_Q/N$",ticks=[0,.5,1,1.5,2])
    save(fig,"qfi_density_map")

    colors=["#293D5A","#007E87","#739651","#C1843D","#AF5472"]
    styles=["-","--","-.",":","-"]
    qcuts=[0,np.pi/12,np.pi/6,np.pi/4,np.pi/2]
    labels=["0",r"\pi/12",r"\pi/6",r"\pi/4",r"\pi/2"]
    rline=np.linspace(-3,3,1801)
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(7.205,2.85),sharex=True,sharey=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.78,wspace=.16)
    for i,(ax,d) in enumerate(zip(axes,DELTAS)):
        for j,(qc,label) in enumerate(zip(qcuts,labels)):
            values=qfi_density(qc,rline,d)
            ax.plot(rline,values,color=colors[j],linestyle=styles[j],label=rf"$Q={label}$")
            rows.extend((d,qc,float(r0),float(f)) for r0,f in zip(rline,values))
        ax.set(xlabel=r"$\mu/t$",xlim=(-3,3),ylim=(0,2.05),title=rf"$\Delta_0/t={d}$")
        ax.set_xticks([-3,-1.5,0,1.5,3])
        ax.set_yticks([0,.5,1,1.5,2])
        ax.text(-.12,1.11,"abc"[i],transform=ax.transAxes,weight="bold",fontsize=9)
    axes[0].set_ylabel(r"$f_Q=F_Q/N$")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.54,1.005),ncol=5)
    save(fig,"qfi_vs_mu")
    with (OUT/"qfi_vs_mu.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.writer(file); writer.writerow(["delta_over_t","Q","mu_over_t","f_Q"]); writer.writerows(rows)

    rcuts=[0,0.8,1.6,2.4]
    qline=np.linspace(-np.pi/2,np.pi/2,1801)
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(7.205,2.85),sharex=True,sharey=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.78,wspace=.16)
    for i,(ax,d) in enumerate(zip(axes,DELTAS)):
        for j,rc in enumerate(rcuts):
            values=qfi_density(qline,rc,d)
            ax.plot(qline,values,color=colors[j],linestyle=styles[j],label=rf"$\mu/t={rc:g}$")
            rows.extend((d,rc,float(q0),float(f)) for q0,f in zip(qline,values))
        q_ticks(ax)
        ax.set(xlim=(-np.pi/2,np.pi/2),ylim=(0,2.05),title=rf"$\Delta_0/t={d}$")
        ax.set_yticks([0,.5,1,1.5,2])
        ax.text(-.12,1.11,"abc"[i],transform=ax.transAxes,weight="bold",fontsize=9)
    axes[0].set_ylabel(r"$f_Q=F_Q/N$")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.54,1.005),ncol=4)
    save(fig,"qfi_vs_Q")
    with (OUT/"qfi_vs_Q.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.writer(file); writer.writerow(["delta_over_t","mu_over_t","Q","f_Q"]); writer.writerows(rows)


if __name__=="__main__":
    print(json.dumps(verify(),indent=2))
    make_figures()
    print("Saved all QFI figures, grids, and line-cut data.")