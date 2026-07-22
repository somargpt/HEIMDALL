"""Testes dos baselines (gravitacional e radiation) e da comparação de ajuste."""

from __future__ import annotations

import numpy as np
import pytest

from migracao import Params, Simulator, load_cities
from migracao.baselines import (
    compare_against_baselines,
    gravity_fit,
    radiation_predict,
)


@pytest.fixture
def model_od(cities):
    res = Simulator(cities, Params(seed=1)).run(150)
    return res.od_accumulated, cities.distance_matrix(normalize=False)


def test_gravity_recupera_b_de_dados_sinteticos(cities):
    # Gera uma O-D puramente gravitacional com b conhecido e verifica a recuperação.
    L = cities.L_init
    d = cities.distance_matrix(normalize=False)
    n = cities.n
    b_true = 1.7
    with np.errstate(divide="ignore"):
        od = np.where(d > 0, np.outer(L, L) * d ** (-b_true), 0.0)
    np.fill_diagonal(od, 0.0)
    fit = gravity_fit(od, L, d, constrained=True)
    assert fit.b == pytest.approx(b_true, abs=0.05)


def test_gravity_predicoes_batem_saidas(model_od, cities):
    od, d = model_od
    fit = gravity_fit(od, cities.L_init, d)
    outflows = np.where(~np.eye(cities.n, dtype=bool), od, 0.0).sum(axis=1)
    assert np.allclose(fit.predicted.sum(axis=1), outflows)
    assert np.all(fit.predicted >= 0)
    assert np.allclose(np.diag(fit.predicted), 0.0)


def test_radiation_bem_formado(model_od, cities):
    od, d = model_od
    outflows = np.where(~np.eye(cities.n, dtype=bool), od, 0.0).sum(axis=1)
    pred = radiation_predict(cities.L_init, d, outflows)
    assert np.all(pred >= 0)
    assert np.allclose(np.diag(pred), 0.0)
    assert np.allclose(pred.sum(axis=1), outflows)


def test_comparacao_retorna_ambos_modelos(model_od, cities):
    od, d = model_od
    table, preds, grav = compare_against_baselines(od, cities.L_init, d)
    assert set(preds) == {"gravitacional", "radiation"}
    assert set(table["modelo"]) == {"gravitacional", "radiation"}
    # CPC de identidade = 1.
    from migracao.metrics import cpc
    off = np.where(~np.eye(cities.n, dtype=bool), od, 0.0)
    assert cpc(off, off) == pytest.approx(1.0)


def test_gravity_ajusta_melhor_que_radiation_no_modelo(model_od, cities):
    # O modelo tem decaimento de distância explícito => o gravitacional deve
    # reproduzir sua O-D melhor que o radiation (livre de parâmetros).
    od, d = model_od
    table, _, _ = compare_against_baselines(od, cities.L_init, d)
    cpc_grav = float(table.set_index("modelo").loc["gravitacional", "CPC"])
    cpc_rad = float(table.set_index("modelo").loc["radiation", "CPC"])
    assert cpc_grav > cpc_rad
