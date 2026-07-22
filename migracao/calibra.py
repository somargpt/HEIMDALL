"""Calibração por máxima verossimilhança de ``kappa``, ``lam`` e ``theta``.

Estima os parâmetros de escolha a partir de uma **matriz origem-destino
observada** (formato do Censo IBGE — migração de data fixa: contagens de
migrantes de ``i`` para ``j`` no intervalo). Reporta **erros padrão** (via
informação de Fisher) e **ajuste fora da amostra** por *holdout* de pares O-D.

Identificação
-------------
Sobre a matriz de **migrantes** (fora da diagonal), a verossimilhança natural é a
do **logit condicional de destino** (condicional a migrar). A utilidade de mudar
de ``i`` para ``j`` é ``U_ij = V_j − kappa·d_ij − F + lam·ln(1+N_ij)``; como ``F``
é constante entre destinos, ele **cancela** na normalização por origem — logo a
O-D de migrantes identifica ``theta``, ``kappa`` e ``lam`` (exatamente os três
pedidos), **não** ``F``.

O índice linear é ``eta_ij = cV·V_j + cD·d_ij + cN·ln(1+N_ij)`` com
``cV = theta``, ``cD = −theta·kappa``, ``cN = theta·lam``. A log-verossimilhança
é **côncava** em ``(cV, cD, cN)`` (logit condicional), então o MLE é único; os
parâmetros estruturais saem por ``theta=cV``, ``kappa=−cD/cV``, ``lam=cN/cV``
(erros padrão pelo método delta).

Requer ``N_ij`` (estoque de diáspora) como regressor para identificar ``lam``. Se
``N`` não for fornecido, estima-se apenas ``theta`` e ``kappa`` (com aviso) — em
linha com "não inventar dados".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from .cities import Cities
from .metrics import cpc
from .params import Params

_LOG_FLOOR = 1e-12


# ----------------------------------------------------------------------
def place_value_static(cities: Cities, params: Params | None = None,
                       L0: float | None = None) -> np.ndarray:
    """Valor do lugar ``V_j`` numa seção transversal (desemprego = u0, sem entrada).

    Usa as mesmas equações de preço do motor, com a população observada.
    """
    p = params or Params()
    L = cities.L_init
    ref = float(L0) if L0 is not None else (float(p.L0) if p.L0 is not None else float(np.mean(L)))
    ratio = np.maximum(L / ref, 1e-9)
    w = cities.A * np.power(ratio, -p.alpha)
    h = p.h0 * np.power(ratio, p.beta)
    u = np.clip(p.u0, 0.0, 0.45)
    inside = np.maximum(w * (1.0 - u), _LOG_FLOOR)
    return np.log(inside) - h + cities.amenity


# ----------------------------------------------------------------------
def _design(V: np.ndarray, distance: np.ndarray, N: np.ndarray | None):
    """Matriz de regressores ``X`` (n, n, p) e nomes das colunas.

    Colunas: ``V_j`` (destino), ``d_ij`` (distância), e ``ln(1+N_ij)`` se ``N`` dado.
    """
    n = V.shape[0]
    cols = [np.broadcast_to(V[None, :], (n, n)), distance]
    names = ["V", "d"]
    if N is not None:
        cols.append(np.log1p(N))
        names.append("lnN")
    X = np.stack(cols, axis=-1).astype(float)  # (n, n, p)
    return X, names


def _softmax_rows(eta: np.ndarray, include: np.ndarray):
    """Softmax por linha sobre alternativas incluídas; retorna (P, logsumexp)."""
    masked = np.where(include, eta, -np.inf)
    m = masked.max(axis=1, keepdims=True)
    Z = np.exp(masked - m)  # excluídas -> 0
    denom = Z.sum(axis=1, keepdims=True)
    P = Z / denom
    logden = (m + np.log(denom)).ravel()
    return P, logden


def _neg_ll_grad(beta, X, counts, include, Ni, scale=1.0):
    eta = X @ beta                              # (n, n), finito
    P, logden = _softmax_rows(eta, include)
    # LL = sum_ij counts*eta - sum_i Ni*logden  (counts=0 fora do conjunto).
    # ``scale`` normaliza a objetivo (ex.: 1/total) para estabilizar a otimização
    # sem mudar o argmax; a informação de Fisher (SE) usa as contagens reais.
    ll = float(np.sum(counts * eta) - np.sum(Ni * logden))
    resid = counts - Ni[:, None] * P            # (n, n)
    grad = np.einsum("ij,ijp->p", resid, X)     # (p,)
    return -ll * scale, -grad * scale


def _fisher_information(beta, X, include, Ni):
    """Informação de Fisher observada (p x p) no ótimo (LL côncava)."""
    eta = X @ beta
    P, _ = _softmax_rows(eta, include)
    Xbar = np.einsum("ij,ijp->ip", P, X)                    # (n, p)
    second = np.einsum("ij,ijp,ijq->ipq", P, X, X)          # (n, p, p)
    cov_i = second - np.einsum("ip,iq->ipq", Xbar, Xbar)    # (n, p, p)
    return np.einsum("i,ipq->pq", Ni, cov_i)                # (p, p)


# ----------------------------------------------------------------------
@dataclass
class CalibrationResult:
    theta: float
    kappa: float
    lam: float
    se: dict = field(default_factory=dict)          # SE de theta/kappa/lam
    beta: np.ndarray = None                          # coefs lineares (cV, cD, [cN])
    cov_beta: np.ndarray = None
    loglik: float = float("nan")
    n_migrantes: float = 0.0
    converged: bool = False
    identified_lam: bool = True
    col_names: list = field(default_factory=list)

    def summary_frame(self) -> pd.DataFrame:
        rows = [
            {"param": "theta", "estimativa": self.theta, "erro_padrao": self.se.get("theta", float("nan"))},
            {"param": "kappa", "estimativa": self.kappa, "erro_padrao": self.se.get("kappa", float("nan"))},
            {"param": "lam", "estimativa": self.lam, "erro_padrao": self.se.get("lam", float("nan"))},
        ]
        df = pd.DataFrame(rows)
        df["z"] = df["estimativa"] / df["erro_padrao"]
        return df


def _structural_from_beta(beta, cov_beta, names):
    """Converte coefs lineares em (theta, kappa, lam) + SE pelo método delta."""
    cV = beta[0]
    cD = beta[1]
    theta = cV
    kappa = -cD / cV
    has_N = "lnN" in names
    cN = beta[2] if has_N else 0.0
    lam = (cN / cV) if has_N else 0.0

    p = beta.shape[0]
    # Jacobiano g(beta) = [theta, kappa, lam].
    J = np.zeros((3, p))
    J[0, 0] = 1.0                       # d theta / d cV
    J[1, 0] = cD / cV**2                # d kappa / d cV
    J[1, 1] = -1.0 / cV                 # d kappa / d cD
    if has_N:
        J[2, 0] = -cN / cV**2           # d lam / d cV
        J[2, 2] = 1.0 / cV              # d lam / d cN
    cov_struct = J @ cov_beta @ J.T
    se = {
        "theta": float(np.sqrt(max(cov_struct[0, 0], 0.0))),
        "kappa": float(np.sqrt(max(cov_struct[1, 1], 0.0))),
        "lam": float(np.sqrt(max(cov_struct[2, 2], 0.0))) if has_N else float("nan"),
    }
    return theta, kappa, lam, se


def calibrate(
    od_observed: np.ndarray,
    V: np.ndarray,
    distance: np.ndarray,
    N: np.ndarray | None = None,
    include: np.ndarray | None = None,
) -> CalibrationResult:
    """Estima (theta, kappa, lam) por MLE do logit condicional de destino.

    Parâmetros
        od_observed : contagens de migrantes ``i->j`` (n x n; diagonal ignorada).
        V           : valor do lugar por destino (n,). Ver ``place_value_static``.
        distance    : matriz de distâncias (n x n), mesma escala do modelo.
        N           : estoque de diáspora ``N_ij`` (n x n). Sem ``N``, ``lam`` não
                      é identificado (estima theta, kappa).
        include     : máscara booleana do conjunto de escolha (n x n). Default =
                      todos os pares fora da diagonal.
    """
    n = V.shape[0]
    counts = np.asarray(od_observed, dtype=float).copy()
    np.fill_diagonal(counts, 0.0)
    if include is None:
        include = ~np.eye(n, dtype=bool)
    else:
        include = include & ~np.eye(n, dtype=bool)
    counts = np.where(include, counts, 0.0)
    Ni = counts.sum(axis=1)

    X, names = _design(V, distance, N)
    p = X.shape[-1]
    beta0 = np.zeros(p)

    scale = 1.0 / max(1.0, float(Ni.sum()))
    res = optimize.minimize(
        _neg_ll_grad, beta0, args=(X, counts, include, Ni, scale),
        jac=True, method="BFGS", options={"gtol": 1e-8},
    )
    beta = res.x
    info = _fisher_information(beta, X, include, Ni)
    try:
        cov_beta = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov_beta = np.linalg.pinv(info)

    theta, kappa, lam, se = _structural_from_beta(beta, cov_beta, names)
    loglik = -float(res.fun) / scale  # desfaz a normalização da objetivo
    return CalibrationResult(
        theta=float(theta), kappa=float(kappa), lam=float(lam), se=se,
        beta=beta, cov_beta=cov_beta, loglik=loglik,
        n_migrantes=float(Ni.sum()), converged=bool(res.success),
        identified_lam=(N is not None), col_names=names,
    )


# ----------------------------------------------------------------------
def _predict_flows(cal: CalibrationResult, X, include, Ni_obs):
    """Fluxos esperados previstos = Ni_obs * P (sobre o conjunto completo)."""
    eta = X @ cal.beta
    P, _ = _softmax_rows(eta, include)
    return Ni_obs[:, None] * P


def calibrate_holdout(
    od_observed: np.ndarray,
    V: np.ndarray,
    distance: np.ndarray,
    N: np.ndarray | None = None,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict:
    """Ajuste fora da amostra por *holdout* de pares O-D.

    Remove aleatoriamente uma fração ``test_frac`` dos pares (i,j) do conjunto de
    escolha de treino (o logit condicional é consistente sob subconjuntos —
    propriedade IIA), reestima e mede o ajuste nos pares retidos:
    CPC, correlação e log-verossimilhança preditiva média.
    """
    n = V.shape[0]
    rng = np.random.default_rng(seed)
    off = ~np.eye(n, dtype=bool)
    idx = np.argwhere(off)
    n_test = max(1, int(round(test_frac * len(idx))))
    test_sel = rng.choice(len(idx), size=n_test, replace=False)
    test_mask = np.zeros((n, n), dtype=bool)
    for t in test_sel:
        i, j = idx[t]
        test_mask[i, j] = True
    train_include = off & ~test_mask

    cal = calibrate(od_observed, V, distance, N=N, include=train_include)

    # Predição sobre o conjunto COMPLETO, usando os totais observados por origem.
    counts = np.where(off, np.asarray(od_observed, float), 0.0)
    Ni_obs = counts.sum(axis=1)
    X, _ = _design(V, distance, N)
    pred_full = _predict_flows(cal, X, off, Ni_obs)

    obs_test = np.where(test_mask, counts, 0.0)
    pred_test = np.where(test_mask, pred_full, 0.0)

    # Métricas fora da amostra nos pares retidos.
    cpc_oos = cpc(obs_test, pred_test)
    o = counts[test_mask]
    pr = pred_full[test_mask]
    corr = float(np.corrcoef(o, pr)[0, 1]) if o.std() > 0 and pr.std() > 0 else 0.0
    # LL preditiva média por migrante retido.
    P_full, _ = _softmax_rows(X @ cal.beta, off)
    lp = np.log(np.clip(P_full[test_mask], 1e-300, None))
    total_test = o.sum()
    pred_ll = float(np.sum(o * lp) / total_test) if total_test > 0 else float("nan")

    return {
        "calibration": cal,
        "n_test_pairs": int(n_test),
        "cpc_oos": cpc_oos,
        "corr_oos": corr,
        "pred_loglik_medio": pred_ll,
    }


# ----------------------------------------------------------------------
def load_od_matrix(path) -> tuple[np.ndarray, list[str]]:
    """Carrega uma matriz O-D quadrada (linhas=origem, colunas=destino).

    Mesmo formato de ``od_matrix.csv`` produzido pela CLI. Para tabelas reais do
    IBGE (migração de data fixa), reorganize os dados nessa forma quadrada, com
    origens e destinos nas mesmas categorias e na mesma ordem.
    """
    df = pd.read_csv(path, index_col=0)
    names = [str(c) for c in df.columns]
    return df.to_numpy(dtype=float), names
