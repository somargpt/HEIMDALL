"""Teste 1 — Conservação de população.

Sem natalidade/mortalidade: a soma de L deve ser constante a cada período,
tolerância 1e-9. Verificado em múltiplas configurações (caso completo,
conjunto amostrado, sem liquidez, com choque de rede).
"""

from __future__ import annotations

import numpy as np

from migracao import Params, Simulator


def _max_dev(res, total0):
    sums = res.panel.groupby("t")["L"].sum().to_numpy()
    return float(np.max(np.abs(sums - total0)))


def test_conservacao_caso_completo(cities):
    total0 = float(cities.L_init.sum())
    res = Simulator(cities, Params(seed=1)).run(200)
    assert _max_dev(res, total0) < 1e-9


def test_conservacao_amostrado(cities):
    total0 = float(cities.L_init.sum())
    res = Simulator(cities, Params(seed=3, k=5)).run(200)
    assert _max_dev(res, total0) < 1e-9


def test_conservacao_sem_liquidez(cities):
    total0 = float(cities.L_init.sum())
    res = Simulator(cities, Params(seed=5, use_liquidity=False)).run(150)
    assert _max_dev(res, total0) < 1e-9


def test_conservacao_com_choque_de_rede(cities):
    total0 = float(cities.L_init.sum())
    n = cities.n
    N0 = np.zeros((n, n))
    N0[:, 4] = 500.0
    np.fill_diagonal(N0, 0.0)
    res = Simulator(cities, Params(seed=7, lam=1.0)).run(120, initial_N=N0)
    assert _max_dev(res, total0) < 1e-9


def test_conservacao_regime_alto_alpha(cities):
    # Regime de forte congestão (dinâmica intensa) ainda conserva população.
    total0 = float(cities.L_init.sum())
    res = Simulator(cities, Params(seed=2, alpha=1.2, m0=0.2, theta=4.0)).run(180)
    assert _max_dev(res, total0) < 1e-9
