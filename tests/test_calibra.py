"""Testes da calibração (MLE de theta/kappa/lam) e do ajuste fora da amostra.

Os dados de teste são gerados PELO logit condicional de destino (especificação
correta) — é sobre esse processo que a calibração deve recuperar os parâmetros.
Calibrar sobre a saída do modelo dinâmico é uma má-especificação (documentada).
"""

from __future__ import annotations

import numpy as np
import pytest

from migracao import Params, Simulator, load_cities
from migracao.calibra import (
    calibrate,
    calibrate_holdout,
    load_od_matrix,
    place_value_static,
)


def _static_probs(V, d, N, theta, kappa, lam):
    n = V.shape[0]
    off = ~np.eye(n, dtype=bool)
    eta = np.where(off, theta * (V[None, :] - kappa * d + lam * np.log1p(N)), -np.inf)
    m = eta.max(1, keepdims=True)
    P = np.exp(eta - m)
    return P / P.sum(1, keepdims=True)


@pytest.fixture
def setup(cities):
    V = place_value_static(cities, Params())
    d = cities.distance_matrix(normalize=True)
    N = Simulator(cities, Params(seed=3)).run(60).od_accumulated
    return cities, V, d, N


def test_place_value_static_bate_com_o_motor(cities):
    V = place_value_static(cities, Params())
    sim = Simulator(cities, Params())
    V_engine, *_ = sim._place_value(cities.L_init, np.zeros(cities.n))
    assert np.allclose(V, V_engine)


def test_recuperacao_exata_sem_ruido(setup):
    cities, V, d, N = setup
    tt, kt, lt = 2.0, 0.8, 0.4
    P = _static_probs(V, d, N, tt, kt, lt)
    counts = 200000.0 * P  # contagens esperadas (sem ruído)
    cal = calibrate(counts, V, d, N=N)
    assert cal.converged
    assert cal.theta == pytest.approx(tt, abs=1e-3)
    assert cal.kappa == pytest.approx(kt, abs=1e-3)
    assert cal.lam == pytest.approx(lt, abs=1e-3)


def test_erros_padrao_cobrem_a_verdade(setup):
    cities, V, d, N = setup
    tt, kt, lt = 2.0, 0.8, 0.4
    P = _static_probs(V, d, N, tt, kt, lt)
    rng = np.random.default_rng(1)
    counts = np.vstack([rng.multinomial(3000, P[i]) for i in range(cities.n)]).astype(float)
    cal = calibrate(counts, V, d, N=N)
    assert abs(cal.theta - tt) < 4 * cal.se["theta"]
    assert abs(cal.kappa - kt) < 4 * cal.se["kappa"]
    assert abs(cal.lam - lt) < 4 * cal.se["lam"]
    # Erros padrão positivos e finitos.
    assert all(np.isfinite(v) and v > 0 for v in cal.se.values())


def test_holdout_ajuste_fora_da_amostra(setup):
    cities, V, d, N = setup
    P = _static_probs(V, d, N, 2.0, 0.8, 0.4)
    rng = np.random.default_rng(2)
    counts = np.vstack([rng.multinomial(5000, P[i]) for i in range(cities.n)]).astype(float)
    res = calibrate_holdout(counts, V, d, N=N, test_frac=0.25, seed=0)
    n = cities.n
    assert res["n_test_pairs"] == round(0.25 * n * (n - 1))
    # Boa especificação => ajuste alto nos pares retidos.
    assert res["cpc_oos"] > 0.9
    assert res["corr_oos"] > 0.9


def test_sem_N_lam_nao_identificado(setup):
    cities, V, d, N = setup
    P = _static_probs(V, d, N, 2.0, 0.8, 0.4)
    counts = 50000.0 * P
    cal = calibrate(counts, V, d, N=None)
    assert cal.identified_lam is False
    assert cal.lam == 0.0
    assert np.isnan(cal.se["lam"])
    assert np.isfinite(cal.theta) and np.isfinite(cal.kappa)


def test_load_od_matrix_roundtrip(tmp_path, cities):
    import pandas as pd
    od = Simulator(cities, Params(seed=1)).run(30).od_accumulated
    p = tmp_path / "od.csv"
    pd.DataFrame(od, index=cities.names, columns=cities.names).to_csv(p)
    mat, names = load_od_matrix(p)
    assert names == list(cities.names)
    assert np.allclose(mat, od)


def test_cli_calibrate_demo(tmp_path):
    from migracao.cli import main
    import json
    out = tmp_path / "cal"
    assert main(["calibrate", "--demo", "--out", str(out)]) == 0
    assert (out / "calibracao.csv").exists()
    data = json.load(open(out / "calibracao.json"))
    # A demo recupera aproximadamente a verdade (theta=2, kappa=0.8, lam=0.4).
    assert data["theta"] == pytest.approx(2.0, abs=0.2)
    assert data["kappa"] == pytest.approx(0.8, abs=0.15)
    assert data["lam"] == pytest.approx(0.4, abs=0.15)
