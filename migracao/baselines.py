"""Baselines para comparação de ajuste sobre uma matriz origem-destino.

Dois modelos clássicos de interação espacial, comparados ao modelo mecanicista
usando a MESMA matriz O-D:

* **Gravitacional**: ``F_ij ∝ L_i^p L_j^q / d_ij^b`` (por default p=q=1, conforme o
  enunciado; expoentes livres opcionais). Ajustado por mínimos quadrados sobre o
  log dos fluxos positivos.
* **Radiation model** (Simini et al., 2012): livre de parâmetros —
  ``T_ij = T_i · m_i n_j / [(m_i + s_ij)(m_i + n_j + s_ij)]``, com ``s_ij`` a
  população no círculo de raio ``d_ij`` centrado em ``i`` (excluindo origem e
  destino) e ``T_i`` o total de saídas observado da origem.

Ambos são normalizados por origem para reproduzir as saídas observadas ``O_i``
(versão *production-constrained*), de modo que a comparação (CPC, correlação)
avalie a **estrutura espacial**, não os totais.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import cpc


def _offdiag_mask(n: int) -> np.ndarray:
    m = ~np.eye(n, dtype=bool)
    return m


def _row_normalize_to_outflows(pred: np.ndarray, outflows: np.ndarray) -> np.ndarray:
    """Escala cada linha de ``pred`` para somar ``outflows[i]`` (diagonal zerada)."""
    p = pred.copy()
    np.fill_diagonal(p, 0.0)
    row = p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(row > 0, outflows[:, None] / row, 0.0)
    return p * scale


# ----------------------------------------------------------------------
@dataclass
class GravityResult:
    """Resultado do ajuste gravitacional."""

    K: float                 # constante (exp do intercepto)
    b: float                 # expoente de distância
    p: float                 # expoente de L_i (origem)
    q: float                 # expoente de L_j (destino)
    predicted: np.ndarray    # matriz O-D prevista (normalizada às saídas)
    constrained: bool

    def summary(self) -> dict:
        return {"K": self.K, "b": self.b, "p": self.p, "q": self.q,
                "constrained": self.constrained}


def gravity_fit(
    od_observed: np.ndarray,
    L: np.ndarray,
    distance: np.ndarray,
    constrained: bool = True,
) -> GravityResult:
    """Ajusta o modelo gravitacional a ``od_observed`` por OLS no log dos fluxos.

    ``constrained=True`` (default) fixa p=q=1 (forma ``L_i L_j / d_ij^b`` do
    enunciado) e estima apenas ``K`` e ``b``. ``constrained=False`` estima também
    os expoentes de população.
    """
    n = L.shape[0]
    od = np.asarray(od_observed, dtype=float)
    mask = _offdiag_mask(n) & (od > 0) & (distance > 0)
    i_idx, j_idx = np.where(mask)
    if i_idx.size < 3:
        raise ValueError("Fluxos positivos insuficientes para ajustar o gravitacional.")

    y = np.log(od[i_idx, j_idx])
    ln_di = np.log(L[i_idx])
    ln_dj = np.log(L[j_idx])
    ln_d = np.log(distance[i_idx, j_idx])
    ones = np.ones_like(y)

    if constrained:
        # ln F = c - b ln d + (ln L_i + ln L_j) offset  =>  regressa ln F - offset.
        offset = ln_di + ln_dj
        X = np.column_stack([ones, -ln_d])
        coef, *_ = np.linalg.lstsq(X, y - offset, rcond=None)
        c, b = coef
        p = q = 1.0
    else:
        X = np.column_stack([ones, ln_di, ln_dj, -ln_d])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        c, p, q, b = coef

    K = float(np.exp(c))
    # Predição bruta em todos os pares fora da diagonal.
    Li = L[:, None]
    Lj = L[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = K * (Li ** p) * (Lj ** q) * np.where(distance > 0, distance ** (-b), 0.0)
    np.fill_diagonal(raw, 0.0)

    outflows = np.asarray(od, dtype=float)
    np.fill_diagonal(outflows, 0.0)
    outflows = outflows.sum(axis=1)
    predicted = _row_normalize_to_outflows(raw, outflows)
    return GravityResult(K=K, b=float(b), p=float(p), q=float(q),
                         predicted=predicted, constrained=constrained)


# ----------------------------------------------------------------------
def _intervening_population(L: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """``s_ij`` = população no círculo de raio ``d_ij`` em torno de ``i``,
    excluindo origem ``i`` e destino ``j`` (oportunidades interpostas)."""
    n = L.shape[0]
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        di = distance[i]                     # (n,)
        # closer[j,k] = d_ik < d_ij  => população estritamente mais perto que j.
        closer = di[None, :] < di[:, None]   # (j, k)
        S[i] = closer @ L - L[i]             # exclui a própria origem i
    np.fill_diagonal(S, 0.0)
    S[S < 0] = 0.0
    return S


def radiation_predict(
    L: np.ndarray,
    distance: np.ndarray,
    outflows: np.ndarray,
) -> np.ndarray:
    """Radiation model (Simini et al., 2012), normalizado por origem às saídas.

    ``outflows[i]`` = total de saídas observado da origem ``i`` (define a escala).
    """
    n = L.shape[0]
    m = np.asarray(L, dtype=float)
    s = _intervening_population(L, distance)
    mi = m[:, None]
    nj = m[None, :]
    denom = (mi + s) * (mi + nj + s)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(denom > 0, (mi * nj) / denom, 0.0)
    np.fill_diagonal(P, 0.0)
    return _row_normalize_to_outflows(P, np.asarray(outflows, dtype=float))


# ----------------------------------------------------------------------
def _pearson_offdiag(a: np.ndarray, b: np.ndarray) -> float:
    n = a.shape[0]
    mask = _offdiag_mask(n)
    x = a[mask].ravel()
    y = b[mask].ravel()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compare_od(observed: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Tabela de ajuste (CPC e correlação de Pearson fora da diagonal) por modelo."""
    rows = []
    for name, pred in predictions.items():
        rows.append({
            "modelo": name,
            "CPC": cpc(np.where(~np.eye(observed.shape[0], dtype=bool), observed, 0.0),
                       np.where(~np.eye(pred.shape[0], dtype=bool), pred, 0.0)),
            "pearson_r": _pearson_offdiag(observed, pred),
        })
    return pd.DataFrame(rows).sort_values("CPC", ascending=False).reset_index(drop=True)


def compare_against_baselines(
    od_observed: np.ndarray,
    L: np.ndarray,
    distance: np.ndarray,
    constrained_gravity: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], GravityResult]:
    """Ajusta gravitacional e radiation a ``od_observed`` e devolve a comparação.

    Retorna (tabela_de_ajuste, dict_de_previsoes, resultado_gravitacional).
    A "matriz origem-destino" de referência é a mesma para os dois baselines.
    """
    od = np.asarray(od_observed, dtype=float)
    n = L.shape[0]
    outflows = np.where(~np.eye(n, dtype=bool), od, 0.0).sum(axis=1)

    grav = gravity_fit(od, L, distance, constrained=constrained_gravity)
    rad = radiation_predict(L, distance, outflows)
    preds = {"gravitacional": grav.predicted, "radiation": rad}
    table = compare_od(od, preds)
    return table, preds, grav
