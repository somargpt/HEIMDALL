"""Gráficos de análise (matplotlib, backend headless).

Figuras:
* ``plot_evolution``           — Gini e share da maior cidade por período.
* ``plot_flow_map``            — fluxos dominantes no regime permanente (mapa).
* ``plot_sensitivity_heatmap`` — heatmap da métrica final (ex.: Gini).
* ``plot_baseline_comparison`` — previsto vs. observado (gravitacional/radiation).

Paleta qualitativa segura para daltônicos; ``viridis`` (sequencial perceptual)
nos heatmaps. Estilo consistente entre as figuras via ``_apply_style``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .cities import Cities  # noqa: E402
from .engine import SimulationResult  # noqa: E402

# Paleta qualitativa (Okabe-Ito, segura para daltônicos).
_C = {
    "azul": "#0072B2",
    "laranja": "#E69F00",
    "verde": "#009E73",
    "vermelho": "#D55E00",
    "roxo": "#CC79A7",
    "cinza": "#4D4D4D",
}


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
def plot_evolution(res: SimulationResult, path: str | Path) -> Path:
    """Evolução de Gini e share da maior cidade por período."""
    _apply_style()
    m = res.metrics
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(m["t"], m["gini"], color=_C["azul"], lw=2, label="Gini populacional")
    ax.plot(m["t"], m["share_maior"], color=_C["laranja"], lw=2,
            label="Share da maior cidade")
    ax.set_xlabel("período (t)")
    ax.set_ylabel("índice")
    ax.set_ylim(0, max(0.5, float(m[["gini", "share_maior"]].to_numpy().max()) * 1.15))
    ax.set_title("Concentração populacional ao longo do tempo")
    ax.legend(frameon=False)
    return _save(fig, path)


# ----------------------------------------------------------------------
def plot_flow_map(
    res: SimulationResult,
    cities: Cities,
    path: str | Path,
    top_edges: int = 20,
    dominant_per_origin: bool = True,
) -> Path:
    """Mapa dos fluxos dominantes (matriz O-D acumulada) sobre as coordenadas.

    Tamanho do nó ∝ população final; setas ∝ intensidade do fluxo. Desenha a
    aresta dominante de cada origem e/ou as ``top_edges`` maiores no total.
    """
    _apply_style()
    od = res.od_accumulated.copy()
    np.fill_diagonal(od, 0.0)
    n = cities.n
    Lf = res.panel[res.panel["t"] == res.metrics["t"].max() + 1]
    if len(Lf) == 0:  # fallback: último painel disponível
        Lf = res.panel[res.panel["t"] == res.panel["t"].max()]
    pop = Lf.set_index("cidade").loc[list(cities.names), "L"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 7))
    sizes = 60 + 900 * (pop / pop.max())
    ax.scatter(cities.x, cities.y, s=sizes, color=_C["azul"], alpha=0.6,
               edgecolor="white", zorder=3)
    for i, name in enumerate(cities.names):
        ax.annotate(name.replace("Cidade_", "C"), (cities.x[i], cities.y[i]),
                    fontsize=8, ha="center", va="center", zorder=4)

    # Seleção de arestas.
    edges = set()
    if dominant_per_origin:
        for i in range(n):
            j = int(np.argmax(od[i]))
            if od[i, j] > 0:
                edges.add((i, j))
    if top_edges:
        flat = np.argsort(od, axis=None)[::-1][:top_edges]
        for idx in flat:
            i, j = divmod(int(idx), n)
            if od[i, j] > 0:
                edges.add((i, j))

    fmax = od.max() if od.max() > 0 else 1.0
    for (i, j) in edges:
        lw = 0.5 + 4.0 * (od[i, j] / fmax)
        ax.annotate(
            "", xy=(cities.x[j], cities.y[j]), xytext=(cities.x[i], cities.y[i]),
            arrowprops=dict(arrowstyle="-|>", color=_C["vermelho"], lw=lw,
                            alpha=0.55, shrinkA=8, shrinkB=8),
            zorder=2,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Fluxos dominantes no regime permanente\n(nó ∝ população final, seta ∝ fluxo acumulado)")
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, path)


# ----------------------------------------------------------------------
def plot_sensitivity_heatmap(sweep, path: str | Path) -> Path:
    """Heatmap da métrica final vs. dois parâmetros."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(sweep.grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(sweep.values_x)))
    ax.set_xticklabels([f"{v:g}" for v in sweep.values_x], rotation=45, ha="right")
    ax.set_yticks(range(len(sweep.values_y)))
    ax.set_yticklabels([f"{v:g}" for v in sweep.values_y])
    ax.set_xlabel(sweep.param_x)
    ax.set_ylabel(sweep.param_y)
    ax.set_title(f"{sweep.metric} final vs. {sweep.param_x} × {sweep.param_y}")
    ax.grid(False)
    # Anotações.
    vmin, vmax = sweep.grid.min(), sweep.grid.max()
    for iy in range(sweep.grid.shape[0]):
        for ix in range(sweep.grid.shape[1]):
            val = sweep.grid[iy, ix]
            color = "white" if val < (vmin + vmax) / 2 else "black"
            ax.text(ix, iy, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label=sweep.metric)
    return _save(fig, path)


# ----------------------------------------------------------------------
def plot_baseline_comparison(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Dispersão previsto vs. observado (fora da diagonal) para cada baseline."""
    _apply_style()
    n = observed.shape[0]
    mask = ~np.eye(n, dtype=bool)
    obs = observed[mask].ravel()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    palette = [_C["verde"], _C["roxo"], _C["laranja"], _C["vermelho"]]
    for (name, pred), col in zip(predictions.items(), palette):
        ax.scatter(obs, pred[mask].ravel(), s=18, alpha=0.6, color=col, label=name)
    lim = max(obs.max(), max(p[mask].max() for p in predictions.values())) * 1.05
    ax.plot([0, lim], [0, lim], color=_C["cinza"], ls="--", lw=1, label="identidade")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("fluxo observado (modelo)")
    ax.set_ylabel("fluxo previsto (baseline)")
    ax.set_title("Ajuste dos baselines à matriz O-D do modelo")
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, path)
