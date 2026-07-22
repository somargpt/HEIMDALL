"""Teste 6 — Logit multinomial.

* As probabilidades somam 1 por origem (caso completo e conjunto amostrado).
* Nenhum NaN mesmo com salário próximo de zero (A minúsculo => ln(w(1-u)) no piso).
"""

from __future__ import annotations

import numpy as np

from migracao import Params, Simulator, load_cities
from migracao.cities import Cities


def test_probabilidades_somam_um_caso_completo(cities):
    sim = Simulator(cities, Params(seed=1, k=None))
    rng = np.random.default_rng(0)
    for _ in range(20):
        L = rng.uniform(1.0, 500.0, size=cities.n)
        entry = rng.uniform(0.0, 0.3, size=cities.n)
        N = rng.uniform(0.0, 1e4, size=(cities.n, cities.n))
        P = sim.choice_probabilities(L, entry, N, rng=rng)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert not np.isnan(P).any()


def test_probabilidades_somam_um_amostrado(cities):
    sim = Simulator(cities, Params(seed=2, k=4))
    rng = np.random.default_rng(1)
    for _ in range(20):
        L = rng.uniform(1.0, 500.0, size=cities.n)
        P = sim.choice_probabilities(L, rng=rng)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert not np.isnan(P).any()
        # Cada origem tem no máximo k destinos + a opção de ficar => k+1 positivos.
        assert (P > 0).sum(axis=1).max() <= sim.params.k + 1


def test_sem_nan_com_salario_proximo_de_zero(cities):
    # Produtividade minúscula => salário ~ 0.
    tiny = Cities(
        names=cities.names,
        x=cities.x, y=cities.y,
        A=np.full(cities.n, 1e-12),
        amenity=cities.amenity,
        L_init=cities.L_init,
    )
    sim = Simulator(tiny, Params(seed=1))
    V, w, h, u = sim._place_value(tiny.L_init, np.zeros(tiny.n))
    assert np.isfinite(V).all()
    P = sim.choice_probabilities(tiny.L_init)
    assert not np.isnan(P).any()
    assert np.allclose(P.sum(axis=1), 1.0)


def test_simulacao_completa_sem_nan_com_salario_baixo(cities):
    tiny = Cities(
        names=cities.names, x=cities.x, y=cities.y,
        A=np.full(cities.n, 1e-9), amenity=cities.amenity, L_init=cities.L_init,
    )
    res = Simulator(tiny, Params(seed=1)).run(50)
    assert not res.panel[["L", "w", "h", "u"]].isna().to_numpy().any()
    assert not res.flows["fluxo"].isna().to_numpy().any()


def test_cidade_vazia_nao_gera_nan(cities):
    # Cidade com população inicial zero: salário não deve virar infinito/NaN.
    L0 = cities.L_init.copy()
    L0[3] = 0.0
    res = Simulator(cities, Params(seed=1)).run(40, initial_L=L0)
    assert not res.panel[["L", "w", "h", "u"]].isna().to_numpy().any()
    # Conservação continua valendo (a soma inicial inclui a cidade vazia).
    sums = res.panel.groupby("t")["L"].sum().to_numpy()
    assert np.max(np.abs(sums - L0.sum())) < 1e-9


def test_theta_maior_concentra_probabilidade(cities):
    # Sanidade do logit: theta maior => escolha mais determinística (entropia menor).
    rng = np.random.default_rng(0)
    L = cities.L_init.copy()
    p_lo = Simulator(cities, Params(seed=1, theta=0.5, k=None)).choice_probabilities(L, rng=np.random.default_rng(0))
    p_hi = Simulator(cities, Params(seed=1, theta=6.0, k=None)).choice_probabilities(L, rng=np.random.default_rng(0))

    def entropy(P):
        return -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=1).mean()

    assert entropy(p_hi) < entropy(p_lo)
