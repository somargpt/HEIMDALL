"""Teste 3 — Congestão (alpha).

Experimento controlado: variando SOMENTE ``alpha`` (com beta=0, gamma=0 em ambos
os runs, para que a congestão salarial seja o único canal de congestão ativo):

* ``alpha = 0`` (sem congestão salarial): não há força de realimentação negativa;
  a população faz *runaway* rumo ao "canto" — colapsa para pouquíssimas cidades
  (concentração extrema). É o regime "cresce sem estabilizar [num equilíbrio
  interior]".
* ``alpha`` alto: forte realimentação negativa (salário cai com a população) =>
  o sistema converge para um equilíbrio INTERIOR, disperso e estável
  (variação de L abaixo de epsilon nos últimos 20 períodos).

Nota de honestidade (ver README): num modelo de população finita com termo de
rede log-saturante, o regime alpha=0 acaba *congelando* no canto após um
horizonte longo — portanto NÃO testamos "nunca converge" (isso seria um artefato
do horizonte). Testamos a propriedade robusta e invariante ao horizonte que de
fato distingue os regimes: o NÍVEL de concentração de equilíbrio — canto (poucas
cidades, HHI alto) vs. interior (todas as cidades vivas, HHI baixo).
"""

from __future__ import annotations

import numpy as np

from migracao import Params, Simulator

# Parâmetros comuns aos dois runs (só alpha varia).
_BASE = dict(seed=1, beta=0.0, gamma=0.0, lam=0.6, m0=0.05, theta=2.0,
             k=None, use_liquidity=False)
_T = 160
_W = 20


def _run(cities, alpha):
    return Simulator(cities, Params(alpha=alpha, **_BASE)).run(_T)


def _final_shares(res, L_pivot):
    L = L_pivot(res)[-1]
    return np.sort(L / L.sum())[::-1]


def _max_dl_over_total(res, L_pivot, w=_W):
    L = L_pivot(res)
    return float(np.abs(np.diff(L[-w - 1:], axis=0)).max() / L[0].sum())


def test_alpha_zero_colapsa_para_o_canto(cities, L_pivot):
    """Sem congestão: concentração extrema (canto) — poucas cidades sobrevivem."""
    res = _run(cities, 0.0)
    hhi = res.metrics["hhi"].to_numpy()
    shares = _final_shares(res, L_pivot)
    # Concentração cresceu muito (nível robusto, não transiente).
    assert hhi[-1] > 0.20
    assert hhi[-1] - hhi[0] > 0.10
    # A população colapsou para poucas cidades.
    assert shares[:2].sum() > 0.55           # top-2 concentra a maioria
    assert int((shares > 0.01).sum()) <= 8   # muitas cidades esvaziaram


def test_alpha_alto_converge_no_interior(cities, L_pivot):
    """Congestão forte: equilíbrio interior, disperso e convergente."""
    res = _run(cities, 1.2)
    hhi = res.metrics["hhi"].to_numpy()
    shares = _final_shares(res, L_pivot)
    # Converge: variação de L abaixo de epsilon nos últimos 20 períodos.
    assert _max_dl_over_total(res, L_pivot) < 1.5e-4
    # Equilíbrio INTERIOR: concentração baixa, todas as cidades vivas.
    assert hhi[-1] < 0.10
    assert int((shares > 0.01).sum()) >= 12


def test_congestao_separa_canto_de_interior(cities, L_pivot):
    """A concentração final sem congestão é muito maior que com congestão forte."""
    hhi0 = _run(cities, 0.0).metrics["hhi"].iloc[-1]
    hhiH = _run(cities, 1.2).metrics["hhi"].iloc[-1]
    assert hhi0 > 3.0 * hhiH
