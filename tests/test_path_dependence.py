"""Teste 5 — Path dependence (dependência da trajetória).

Com ``lam`` alto (efeito de rede forte), dois runs com choque inicial diferente
convergem para matrizes origem-destino distintas. Com ``lam`` baixo, os mesmos
choques se dissipam e as matrizes O-D ficam quase idênticas.

O choque é um estoque inicial de diáspora ``N_ij`` injetado na *coluna* de uma
cidade-alvo (todos passam a "conhecer" alguém lá), auto-reforçado quando lam é alto.
"""

from __future__ import annotations

import numpy as np

from migracao import Params, Simulator


def _od_with_shock(cities, target, lam, periodos=120, mag=800.0, seed=1):
    n = cities.n
    N0 = np.zeros((n, n))
    N0[:, target] = mag
    np.fill_diagonal(N0, 0.0)
    res = Simulator(cities, Params(seed=seed, lam=lam, alpha=0.15, k=None)).run(
        periodos, initial_N=N0
    )
    return res.od_accumulated


def _cos_distance(A, B):
    a, b = A.ravel(), B.ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return 1.0 - float(a @ b / denom)


def test_lam_alto_gera_od_divergentes(cities):
    A = _od_with_shock(cities, target=5, lam=1.2)
    B = _od_with_shock(cities, target=9, lam=1.2)
    assert _cos_distance(A, B) > 0.5


def test_lam_baixo_dissipa_o_choque(cities):
    A = _od_with_shock(cities, target=5, lam=0.05)
    B = _od_with_shock(cities, target=9, lam=0.05)
    assert _cos_distance(A, B) < 0.15


def test_path_dependence_cresce_com_lam(cities):
    d_baixo = _cos_distance(
        _od_with_shock(cities, 5, lam=0.05), _od_with_shock(cities, 9, lam=0.05)
    )
    d_alto = _cos_distance(
        _od_with_shock(cities, 5, lam=1.2), _od_with_shock(cities, 9, lam=1.2)
    )
    # A divergência das matrizes O-D é muito maior sob lam alto.
    assert d_alto > d_baixo + 0.3
