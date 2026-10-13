"""
Figuras de los pasos del algoritmo de reconstrucción de w(z) con procesos gaussianos.

Uso:  python figuras_algoritmo_gp.py            (guarda los PNG en ./figuras)
Requiere: numpy, scipy, matplotlib
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import cholesky

# ---------------------------------------------------------------- estilo
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.serif": ["STIXGeneral", "Times New Roman"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "figure.figsize": (8, 5),
})
BG, PK, PU = "#fbf7fb", "#d65fb0", "#8e55c8"      # fondo, rosa, morado
TX, MU, GR = "#2b2233", "#6d6275", "#b9aec0"      # texto, gris oscuro, gris claro
mpl.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TX, "axes.labelcolor": TX, "xtick.color": MU, "ytick.color": MU,
    "axes.edgecolor": GR, "font.size": 14, "axes.titlesize": 14,
})
OUT = "figuras"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(3)


def limpiar(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------ ingredientes comunes
m = 50                                   # nodos de la malla en z
z = np.linspace(0, 3, m)
kappa, ell = 1.0, 0.5                    # hiperparámetros del kernel RBF


def kernel_rbf(u, up, kappa=kappa, ell=ell):
    return kappa**2 * np.exp(-(u[:, None] - up[None, :])**2 / (2 * ell**2))


K = kernel_rbf(z, z) + 1e-8 * np.eye(m)  # Σ22 (con jitter para estabilidad)
U = cholesky(K, lower=True)              # Σ22 = U Uᵀ


# ------------------------------------------------ paso 1: inicialización
def fig_inicializacion():
    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2.2, 1]},
                           constrained_layout=True)
    idx = np.arange(1, m + 1)
    for c in (PK, PU, "#e9a3d3"):
        ax[0].plot(idx, rng.standard_normal(m), color=c, lw=1.3, marker="o", ms=3)
    ax[0].axhspan(-2, 2, color=GR, alpha=.25)
    ax[0].axhline(0, color=MU, lw=.8)
    ax[0].set(xlabel="Nodo $i$ de la malla en $z$", ylabel=r"$\theta^*_i$",
              title=r"3 muestras de $\theta^*\sim$MVN(0, I)")
    limpiar(ax[0])

    muestras = rng.standard_normal(5000)
    ax[1].hist(muestras, bins=30, orientation="horizontal", color=PK,
               alpha=.8, density=True)
    t = np.linspace(-4, 4, 100)
    ax[1].plot(np.exp(-t**2 / 2) / np.sqrt(2 * np.pi), t, color=PU)
    ax[1].set(xlabel="Densidad", title=r"Cada $\theta_i$", yticks=[])
    limpiar(ax[1])
    fig.savefig(f"{OUT}/paso1_inicializacion.png", dpi=120)


# ------------------------------------------------ paso 2: covarianza
def fig_covarianza():
    fig, ax = plt.subplots(
    1, 2,
    figsize=(10, 5),
    gridspec_kw={"width_ratios": [1.3, 1]},
    constrained_layout=True,
    )    
    fig.set_constrained_layout_pads(wspace=0.2, hspace=0.1)
    im = ax[0].imshow(K, extent=[0, 3, 3, 0], cmap="RdPu", vmin=0, vmax=1)
    ax[0].set(xlabel="$u$", ylabel="$u'$", title=r"$\Sigma_{22}=K_w(u,u')$")
    fig.colorbar(im, ax=ax[0], shrink=.7)

    r = np.linspace(0, 2, 200)
    for l, c in ((0.25, GR), (0.5, PK), (1.0, PU)):
        ax[1].plot(r, np.exp(-r**2 / (2 * l**2)), color=c,
                   lw=2.2 if l == 0.5 else 1.5, label=fr"$\ell$ = {l}")
    ax[1].set(xlabel="$|u-u'|$", ylabel=r"$\frac{k(r)}{\kappa^2}$",
              title="RBF: correlación vs distancia")
    ax[1].legend(frameon=False)
    limpiar(ax[1])
    fig.savefig(f"{OUT}/paso2_covarianza.png", dpi=120)


# ------------------------------------------------ paso 3: Cholesky
def fig_cholesky():
    fig, ax = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.1]},
                           constrained_layout=True)
    ax[0].imshow(U, extent=[0, 3, 3, 0], cmap="RdPu")
    ax[0].set(title="U triangular\n" + r"($\Sigma_{22}=UU^T$)", xticks=[], yticks=[])

    theta = rng.standard_normal(m)
    w = U @ theta
    ax[1].plot(theta, z, color=GR, lw=1.2)
    ax[1].set(title=r"Ruido blanco $\theta$", ylabel="$z$")
    ax[1].invert_yaxis()
    limpiar(ax[1])
    ax[2].plot(w, z, color=PK, lw=2)
    ax[2].set(title=r"$w=U\theta$" + "\n(suave)", yticks=[])
    ax[2].invert_yaxis()
    limpiar(ax[2])
    fig.savefig(f"{OUT}/paso3_cholesky.png", dpi=120)


# ------------------------------------------------ paso 4: Simpson
def fig_simpson(path=None):
    from scipy.integrate import simpson
    z_max, u_prima = 2.0, 1.0
    g = lambda u: kappa**2 * np.exp(-(u - u_prima)**2 / (2 * ell**2)) / (1 + u)
    exacto = quad(g, 0, z_max)[0]
 
    def simpson_N(N):                     # N puntos (impar) en [0, z]
        u = np.linspace(0, z_max, N)
        return simpson(g(u), x=u), u
 
    fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1.5, 1]},
                           constrained_layout=True)
    uu = np.linspace(0, z_max, 300)
    ax[0].plot(uu, g(uu), color=PU, lw=2)
    ax[0].fill_between(uu, g(uu), color=PK, alpha=.12)
    N = 9
    aprox, u9 = simpson_N(N)
    # Simpson ajusta una parábola por cada 3 puntos consecutivos
    for j in range(0, N - 2, 2):
        uj = u9[j:j + 3]
        coef = np.polyfit(uj, g(uj), 2)
        ut = np.linspace(uj[0], uj[-1], 40)
        ax[0].plot(ut, np.polyval(coef, ut), color=PK, lw=2)
    ax[0].plot(u9, g(u9), "o", color=PK)
    ax[0].set(xlabel="$u$", ylabel=r"$K_w(u,1)/(1+u)$",
              title=fr"Simpson, $N=9$ puntos:  $\Sigma_{{12}}\approx${aprox:.4f}  (exacto {exacto:.4f})")
    limpiar(ax[0])
 
    Ns = np.arange(3, 42, 2)
    err = [abs(simpson_N(n)[0] - exacto) for n in Ns]
    ax[1].semilogy(Ns, np.maximum(err, 1e-16), "o-", color=PK)
    ax[1].set(xlabel="Número de puntos $N$", ylabel="|Error|", title="Convergencia")
    limpiar(ax[1])
    fig.savefig(path or f"{OUT}/paso4_simpson.png", dpi=120)


# ------------------------------------------------ paso 5: log-likelihood
def fig_loglikelihood():
    """Ilustrativo: datos simulados y covarianza diagonal (no son los datos de DESI)."""
    H0, Om, c, rd = 67.7, 0.31, 299792.458, 147.1

    def DV_rd(zv, w0=-1.0, wa=0.0):
        def E(x):
            f = (1 + x)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * x / (1 + x))
            return np.sqrt(Om * (1 + x)**3 + (1 - Om) * f)
        DM = c / H0 * quad(lambda x: 1 / E(x), 0, zv)[0]
        DH = c / (H0 * E(zv))
        return (zv * DM**2 * DH)**(1 / 3) / rd

    zd = np.array([0.3, 0.51, 0.71, 0.93, 1.32, 1.49, 2.33])   # redshifts típicos
    zg = np.linspace(0.2, 2.5, 80)
    T_fid = np.array([DV_rd(x) for x in zd])
    sigma = 0.012 * T_fid
    d = T_fid + rng.normal(0, sigma)                           # datos simulados
    prop = {"A": (-0.9, -0.2), "B": (-0.5, -1.5)}              # (w0, wa)
    T = {k: np.array([DV_rd(x, *p) for x in zd]) for k, p in prop.items()}
    lnL = {k: -0.5 * np.sum(((d - T[k]) / sigma)**2) for k in prop}

    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.8, 1]},
                           constrained_layout=True)
    estilos = {"A": (PK, "-"), "B": (PU, "--")}
    for k, (w0, wa) in prop.items():
        ax[0].plot(zg, [DV_rd(x, w0, wa) / x**(2 / 3) for x in zg],
                   color=estilos[k][0], ls=estilos[k][1], lw=2, label=f"propuesta {k}")
    ax[0].errorbar(zd, d / zd**(2 / 3), sigma / zd**(2 / 3), fmt="o", color=TX,
                   ms=5, capsize=3, label="datos (simulados)")
    ax[0].set(xlabel="$z$", ylabel=r"$D_V/(r_d\,z^{2/3})$", title="T (teoría) vs d")
    ax[0].legend(frameon=False, fontsize=11)
    limpiar(ax[0])

    ax[1].bar(list(lnL), list(lnL.values()), color=[PK, PU])
    ax[1].set(ylabel=r"$\ln\mathcal{L}$", title=r"$\ln\mathcal{L}=-\frac{1}{2}\chi^2$")
    for i, v in enumerate(lnL.values()):
        ax[1].text(i, v / 2, f"{v:.0f}", ha="center", color="white", fontweight="bold")
    limpiar(ax[1])
    fig.text(.5, -.02, "Ilustrativo: datos simulados, covarianza diagonal",
             ha="center", fontsize=10, color=MU)
    fig.savefig(f"{OUT}/paso5_loglikelihood.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    fig_inicializacion()
    fig_covarianza()
    fig_cholesky()
    fig_simpson()
    fig_loglikelihood()
    print(f"Figuras guardadas en ./{OUT}")
