"""Fixtures compartilhadas pelos testes."""

from __future__ import annotations

import numpy as np
import pytest

from migracao import Params, Simulator, load_cities


@pytest.fixture
def cities():
    """Dataset sintético de 14 cidades (determinístico)."""
    return load_cities()


@pytest.fixture
def L_pivot():
    """Retorna função que transforma o painel em matriz (T+1, n) de populações."""
    def _pivot(res):
        return res.panel.pivot(index="t", columns="cidade", values="L").to_numpy()
    return _pivot
