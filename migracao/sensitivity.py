"""Varredura de sensibilidade: métrica final vs. dois parâmetros (heatmap).

Foco do enunciado: sensibilidade em ``alpha``, ``kappa`` e ``lam`` sobre o Gini
final. Este módulo faz a varredura 2D genérica; a análise produz os heatmaps
pareados (alpha×kappa, alpha×lam, kappa×lam).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cities import Cities
from .engine import Simulator
from .metrics import gini, hhi, largest_share
from .params import Params

_FINAL_METRICS = {
    "gini": lambda L: gini(L),
    "hhi": lambda L: hhi(L),
    "share_maior": lambda L: largest_share(L),
}


@dataclass
class SweepResult:
    """Grade de sensibilidade 2D."""

    param_x: str
    param_y: str
    values_x: np.ndarray
    values_y: np.ndarray
    grid: np.ndarray          # shape (len(values_y), len(values_x)); linhas=y, colunas=x
    metric: str

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for iy, vy in enumerate(self.values_y):
            for ix, vx in enumerate(self.values_x):
                rows.append({self.param_x: vx, self.param_y: vy,
                             self.metric: self.grid[iy, ix]})
        return pd.DataFrame(rows)


def _final_population(cities: Cities, params: Params, periodos: int) -> np.ndarray:
    res = Simulator(cities, params).run(periodos)
    last = res.panel[res.panel["t"] == periodos]
    # Ordena por cidade para estabilidade; a métrica é invariante à ordem.
    return last["L"].to_numpy(dtype=float)


def sweep_2d(
    cities: Cities,
    base_params: Params,
    param_x: str,
    values_x,
    param_y: str,
    values_y,
    periodos: int = 200,
    metric: str = "gini",
) -> SweepResult:
    """Varre ``param_x`` × ``param_y`` e mede ``metric`` na população final.

    Todos os demais parâmetros vêm de ``base_params``. A ``seed`` de
    ``base_params`` é mantida (reprodutível).
    """
    if metric not in _FINAL_METRICS:
        raise KeyError(f"Métrica '{metric}' desconhecida. Use {list(_FINAL_METRICS)}.")
    fn = _FINAL_METRICS[metric]
    vx = np.asarray(list(values_x), dtype=float)
    vy = np.asarray(list(values_y), dtype=float)
    grid = np.empty((vy.size, vx.size), dtype=float)
    for iy, y in enumerate(vy):
        for ix, x in enumerate(vx):
            params = base_params.with_(**{param_x: float(x), param_y: float(y)})
            grid[iy, ix] = fn(_final_population(cities, params, periodos))
    return SweepResult(param_x=param_x, param_y=param_y, values_x=vx, values_y=vy,
                       grid=grid, metric=metric)
