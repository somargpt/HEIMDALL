"""Testes auxiliares: métricas puras e camada de dados (loader/CSV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from migracao import load_cities, synthetic_cities, write_synthetic_csv
from migracao.cities import Cities
from migracao.metrics import cpc, gini, hhi, largest_share, mean_flow_distance


# --------------------------- métricas ---------------------------
def test_gini_igualdade_e_concentracao():
    assert gini(np.ones(10)) == pytest.approx(0.0, abs=1e-12)
    # Toda a massa numa unidade => Gini -> (n-1)/n.
    v = np.zeros(10); v[0] = 100.0
    assert gini(v) == pytest.approx(9 / 10, abs=1e-9)


def test_gini_rejeita_negativos():
    with pytest.raises(ValueError):
        gini(np.array([-1.0, 2.0]))


def test_hhi_bounds():
    assert hhi(np.ones(5)) == pytest.approx(1 / 5)
    v = np.zeros(5); v[0] = 1.0
    assert hhi(v) == pytest.approx(1.0)


def test_largest_share():
    assert largest_share(np.array([1.0, 3.0, 6.0])) == pytest.approx(0.6)
    assert largest_share(np.zeros(3)) == 0.0


def test_mean_flow_distance_ignora_diagonal():
    flow = np.array([[100.0, 2.0], [4.0, 0.0]])
    dist = np.array([[0.0, 10.0], [20.0, 0.0]])
    # (2*10 + 4*20) / (2+4) = 100/6
    assert mean_flow_distance(flow, dist) == pytest.approx(100 / 6)


def test_cpc_identidade_e_disjunto():
    A = np.array([[0.0, 5.0], [3.0, 0.0]])
    assert cpc(A, A) == pytest.approx(1.0)
    B = np.array([[0.0, 0.0], [0.0, 8.0]])
    assert cpc(A, B) == pytest.approx(0.0)


# --------------------------- dados ---------------------------
def test_dataset_sintetico_shape():
    c = synthetic_cities()
    assert c.n == 14
    assert (c.A > 0).all()
    d = c.distance_matrix(normalize=True)
    assert d.max() == pytest.approx(1.0)
    assert np.allclose(np.diag(d), 0.0)


def test_csv_roundtrip(tmp_path):
    path = write_synthetic_csv(tmp_path / "cid.csv")
    c = load_cities(path)
    ref = synthetic_cities()
    assert c.names == ref.names
    assert np.allclose(c.A, ref.A)
    assert np.allclose(c.L_init, ref.L_init)


def test_csv_faltando_coluna_gera_erro(tmp_path):
    df = pd.DataFrame({"nome": ["a"], "x": [0.0]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_cities(p)


def test_cities_valida_A_positivo():
    with pytest.raises(ValueError):
        Cities(names=["a"], x=[0], y=[0], A=[0.0], amenity=[0.0], L_init=[10.0])


def test_cities_rejeita_nao_finitos():
    # NaN numa célula (ex.: célula em branco no CSV) deve ser rejeitado no loader.
    with pytest.raises(ValueError):
        Cities(names=["a", "b"], x=[0.0, 1.0], y=[0.0, 1.0],
               A=[np.nan, 1.0], amenity=[0.0, 0.0], L_init=[10.0, 20.0])
    with pytest.raises(ValueError):
        Cities(names=["a", "b"], x=[0.0, np.inf], y=[0.0, 1.0],
               A=[1.0, 1.0], amenity=[0.0, 0.0], L_init=[10.0, 20.0])


def test_cidades_coincidentes_podem_ser_sorteadas():
    # Duas cidades em coordenadas idênticas: com peso 1/d e piso de distância, a
    # cidade coincidente (d=0) deve ter peso MÁXIMO e ser sempre sorteada (k<n-1).
    from migracao import Params, Simulator
    c = Cities(
        names=["A", "B", "C", "D"],
        x=[0.0, 0.0, 50.0, 90.0],   # A e B coincidem
        y=[0.0, 0.0, 50.0, 10.0],
        A=[1.0, 1.0, 1.0, 1.0],
        amenity=[0.0, 0.0, 0.0, 0.0],
        L_init=[100.0, 100.0, 100.0, 100.0],
    )
    sim = Simulator(c, Params(seed=1, k=1))  # amostra 1 destino por origem
    rng = np.random.default_rng(0)
    incluiu_B = 0
    trials = 2000
    for _ in range(trials):
        mask = sim._choice_mask(rng)
        incluiu_B += bool(mask[0, 1])  # origem A (0) sorteou a gêmea B (1)?
    # B (coincidente com A) deve ser sempre incluída no conjunto de escolha de A.
    assert incluiu_B == trials
