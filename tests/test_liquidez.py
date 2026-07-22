"""Teste 4 — Restrição de liquidez.

Desligar a restrição de liquidez deve aumentar a taxa de saída das cidades de
menor salário (que, com a restrição, ficam "presas" por não conseguirem pagar F).
"""

from __future__ import annotations

import numpy as np

from migracao import Params, Simulator


def _low_wage_indices(cities, params):
    sim = Simulator(cities, params)
    w0 = cities.A * (cities.L_init / sim.L0) ** (-params.alpha)
    return np.where(w0 < params.w_F)[0]


def _cumulative_outflow_low(cities, use_liquidity, periodos=60):
    p = Params(seed=1, use_liquidity=use_liquidity)
    low = _low_wage_indices(cities, p)
    res = Simulator(cities, p).run(periodos)
    return float(res.od_accumulated[low, :].sum()), low


def test_existem_cidades_de_baixo_salario(cities):
    low = _low_wage_indices(cities, Params())
    assert low.size > 0, "dataset sintético deveria ter cidades com w < w_F"


def test_desligar_liquidez_aumenta_saida_de_baixo_salario(cities):
    saida_on, low_on = _cumulative_outflow_low(cities, use_liquidity=True)
    saida_off, low_off = _cumulative_outflow_low(cities, use_liquidity=False)
    assert np.array_equal(low_on, low_off)
    # Sem a restrição, a saída acumulada das cidades pobres é claramente maior.
    assert saida_off > saida_on
    assert saida_off > 1.2 * saida_on


def test_liquidez_nao_afeta_cidades_ricas_tanto_quanto(cities):
    # Sanidade: o efeito da liquidez é concentrado nas cidades de baixo salário.
    p_on = Params(seed=1, use_liquidity=True)
    p_off = Params(seed=1, use_liquidity=False)
    low = _low_wage_indices(cities, p_on)
    high = np.setdiff1d(np.arange(cities.n), low)
    on = Simulator(cities, p_on).run(60).od_accumulated
    off = Simulator(cities, p_off).run(60).od_accumulated
    ratio_low = off[low, :].sum() / on[low, :].sum()
    ratio_high = off[high, :].sum() / on[high, :].sum()
    # A liquidez destrava proporcionalmente muito mais as cidades pobres.
    assert ratio_low > ratio_high
