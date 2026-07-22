"""Modelo computacional de migração interna dirigida por diferencial de riqueza.

API pública principal:

    >>> from migracao import Params, Simulator, load_cities, simulate
    >>> cities = load_cities()            # dataset sintético de 14 cidades
    >>> res = simulate(cities, Params(), periodos=200)
    >>> res.panel.head()
"""

from __future__ import annotations

from .cities import Cities, load_cities, synthetic_cities, write_synthetic_csv
from .engine import SimulationResult, Simulator, simulate
from .params import Params

__all__ = [
    "Params",
    "Cities",
    "Simulator",
    "SimulationResult",
    "simulate",
    "load_cities",
    "synthetic_cities",
    "write_synthetic_csv",
]

__version__ = "0.1.0"
