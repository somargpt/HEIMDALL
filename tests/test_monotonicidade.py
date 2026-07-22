"""Teste 2 — Monotonicidade em kappa.

Aumentar ``kappa`` (desutilidade de distância) deve reduzir a distância média
percorrida pelos fluxos. Usa o caso completo (k=None) para eliminar ruído de
amostragem.
"""

from __future__ import annotations

import numpy as np

from migracao import Params, Simulator
from migracao.metrics import mean_flow_distance


def _mean_distance_for_kappa(cities, kappa, periodos=120):
    res = Simulator(cities, Params(seed=1, kappa=kappa, k=None)).run(periodos)
    # Distância média ponderada pela matriz O-D acumulada (distância normalizada).
    return mean_flow_distance(res.od_accumulated, res.distance)


def test_distancia_media_decresce_com_kappa(cities):
    kappas = [0.1, 0.5, 1.0, 2.0, 4.0]
    dists = [_mean_distance_for_kappa(cities, k) for k in kappas]
    # Estritamente decrescente.
    for a, b in zip(dists, dists[1:]):
        assert b < a, f"distância não decresceu: {dists}"


def test_extremos_de_kappa(cities):
    baixo = _mean_distance_for_kappa(cities, 0.1)
    alto = _mean_distance_for_kappa(cities, 5.0)
    assert alto < baixo
    # Diferença material (não apenas ruído numérico).
    assert (baixo - alto) / baixo > 0.3
