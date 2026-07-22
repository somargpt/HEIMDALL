"""Dados de cidades: holder tipado, loader de CSV e dataset sintético.

Formato de entrada (CSV): colunas ``nome, x, y, A, amenidade, populacao_inicial``.
O loader aceita dados reais no mesmo formato; se nada for fornecido, usa-se o
dataset sintético de 14 cidades (claramente rotulado como sintético).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Colunas obrigatórias no CSV de cidades.
REQUIRED_COLUMNS = ["nome", "x", "y", "A", "amenidade", "populacao_inicial"]


@dataclass
class Cities:
    """Conjunto de cidades com atributos vetorizados.

    Todos os arrays têm shape ``(n,)`` e estão alinhados por índice.
    """

    names: list[str]
    x: np.ndarray
    y: np.ndarray
    A: np.ndarray          # produtividade
    amenity: np.ndarray    # amenidade
    L_init: np.ndarray     # população inicial

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.A = np.asarray(self.A, dtype=float)
        self.amenity = np.asarray(self.amenity, dtype=float)
        self.L_init = np.asarray(self.L_init, dtype=float)
        n = len(self.names)
        for name, arr in [
            ("x", self.x), ("y", self.y), ("A", self.A),
            ("amenidade", self.amenity), ("populacao_inicial", self.L_init),
        ]:
            if arr.shape != (n,):
                raise ValueError(f"Coluna '{name}' com tamanho {arr.shape}, esperado ({n},).")
            # NaN/inf passam despercebidos por comparações (np.nan <= 0 é False) e
            # só quebrariam no meio do run: rejeitamos aqui, com mensagem clara.
            if not np.isfinite(arr).all():
                raise ValueError(f"Coluna '{name}' contém valores não-finitos (NaN/inf).")
        if np.any(self.A <= 0):
            raise ValueError("Produtividade A deve ser estritamente positiva.")
        if np.any(self.L_init < 0):
            raise ValueError("População inicial não pode ser negativa.")

    @property
    def n(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------------
    def distance_matrix(self, normalize: bool = True) -> np.ndarray:
        """Matriz de distâncias euclidianas ``(n, n)``.

        Se ``normalize=True``, divide pela maior distância par-a-par, de modo que
        ``d_ij in [0, 1]``. A diagonal é sempre 0.
        """
        dx = self.x[:, None] - self.x[None, :]
        dy = self.y[:, None] - self.y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        if normalize:
            dmax = d.max()
            if dmax > 0:
                d = d / dmax
        return d

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "nome": self.names,
            "x": self.x,
            "y": self.y,
            "A": self.A,
            "amenidade": self.amenity,
            "populacao_inicial": self.L_init,
        })

    # ------------------------------------------------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Cities":
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV de cidades sem as colunas obrigatórias: {missing}. "
                f"Esperado: {REQUIRED_COLUMNS}."
            )
        return cls(
            names=[str(v) for v in df["nome"].tolist()],
            x=df["x"].to_numpy(dtype=float),
            y=df["y"].to_numpy(dtype=float),
            A=df["A"].to_numpy(dtype=float),
            amenity=df["amenidade"].to_numpy(dtype=float),
            L_init=df["populacao_inicial"].to_numpy(dtype=float),
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "Cities":
        df = pd.read_csv(path)
        return cls.from_dataframe(df)


# ----------------------------------------------------------------------
def synthetic_cities(seed: int = 7) -> Cities:
    """Dataset SINTÉTICO de 14 cidades (não são dados reais).

    Geometria e atributos são gerados de forma determinística a partir de
    ``seed``. Há uma cidade "primaz" (alta produtividade) para tornar a dinâmica
    interessante, além de um espectro de cidades médias e pequenas.

    Populações estão em "milhares de habitantes".
    """
    rng = np.random.default_rng(seed)
    n = 14
    names = [f"Cidade_{i + 1:02d}" for i in range(n)]

    # Coordenadas espalhadas num plano 100 x 100.
    x = rng.uniform(0, 100, size=n)
    y = rng.uniform(0, 100, size=n)

    # Produtividade log-normal em torno de ~1, com uma cidade primaz destacada.
    A = np.exp(rng.normal(0.0, 0.18, size=n))
    A[0] = A.max() * 1.35  # cidade primaz claramente mais produtiva

    # Amenidades centradas em 0 (podem ser negativas ou positivas).
    amenity = rng.normal(0.0, 0.20, size=n)

    # População inicial: primaz maior; demais entre pequenas e médias.
    L_init = rng.uniform(40.0, 180.0, size=n)
    L_init[0] = 320.0

    # Garante coordenadas distintas (evita distância zero fora da diagonal).
    x = np.round(x, 3)
    y = np.round(y, 3)

    return Cities(
        names=names,
        x=x,
        y=y,
        A=np.round(A, 4),
        amenity=np.round(amenity, 4),
        L_init=np.round(L_init, 2),
    )


def write_synthetic_csv(path: str | Path, seed: int = 7) -> Path:
    """Escreve o dataset sintético em CSV e retorna o caminho."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_cities(seed=seed).to_dataframe().to_csv(path, index=False)
    return path


def load_cities(path: str | Path | None = None, seed: int = 7) -> Cities:
    """Carrega cidades de ``path`` (CSV) ou retorna o dataset sintético.

    Loader único usado pela CLI e pelos testes: aceita dados reais no formato
    esperado ou cai no dataset sintético quando ``path is None``.
    """
    if path is None:
        return synthetic_cities(seed=seed)
    return Cities.from_csv(path)
