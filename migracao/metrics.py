"""Métricas de concentração e de migração.

Funções puras sobre arrays/DataFrames — sem estado. Usadas pelo motor (painel
por período) e pela análise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gini(values: np.ndarray) -> float:
    """Coeficiente de Gini de uma distribuição não-negativa (0 = igual, ~1 = concentrado).

    Definição via diferença média absoluta relativa. Robusta a soma zero.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    if np.any(v < 0):
        raise ValueError("Gini exige valores não-negativos.")
    total = v.sum()
    if total <= 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    # Fórmula do Gini ordenado: (2*sum(i*v_i))/(n*sum(v)) - (n+1)/n
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * v)) / (n * total) - (n + 1.0) / n)


def largest_share(pop: np.ndarray) -> float:
    """Fração da população na maior cidade."""
    p = np.asarray(pop, dtype=float)
    total = p.sum()
    return float(p.max() / total) if total > 0 else 0.0


def hhi(pop: np.ndarray) -> float:
    """Índice Herfindahl-Hirschman das *shares* populacionais (em [1/n, 1])."""
    p = np.asarray(pop, dtype=float)
    total = p.sum()
    if total <= 0:
        return 0.0
    s = p / total
    return float(np.sum(s * s))


def gross_migration_rate(flow_offdiag: np.ndarray, total_pop: float) -> float:
    """Taxa bruta de migração = soma dos fluxos (fora da diagonal) / população total."""
    if total_pop <= 0:
        return 0.0
    return float(np.asarray(flow_offdiag, dtype=float).sum() / total_pop)


def mean_flow_distance(flow: np.ndarray, distance: np.ndarray) -> float:
    """Distância média percorrida, ponderada pelos fluxos (fora da diagonal).

    ``flow`` e ``distance`` são ``(n, n)``. A diagonal é ignorada. Retorna 0.0 se
    não houver fluxo.
    """
    f = np.asarray(flow, dtype=float).copy()
    np.fill_diagonal(f, 0.0)
    total = f.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(f * distance) / total)


def accumulate_od(flows_long: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Matriz origem-destino acumulada (soma dos fluxos sobre todos os períodos).

    Recebe o DataFrame longo (t, origem, destino, fluxo) e retorna uma matriz
    ``n x n`` (DataFrame) indexada por nome de cidade.
    """
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    mat = np.zeros((n, n), dtype=float)
    if len(flows_long) > 0:
        oi = flows_long["origem"].map(idx).to_numpy()
        di = flows_long["destino"].map(idx).to_numpy()
        np.add.at(mat, (oi, di), flows_long["fluxo"].to_numpy(dtype=float))
    return pd.DataFrame(mat, index=names, columns=names)


def cpc(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Common Part of Commuters (índice de Sørensen) entre duas matrizes O-D.

    ``CPC = 2 * sum(min(obs, pred)) / (sum(obs) + sum(pred))`` em [0, 1].
    1 = ajuste perfeito; 0 = sem sobreposição.
    """
    o = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    denom = o.sum() + p.sum()
    if denom <= 0:
        return 0.0
    return float(2.0 * np.minimum(o, p).sum() / denom)
