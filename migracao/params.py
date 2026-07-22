"""Parâmetros do modelo de migração interna.

Todos os parâmetros comportamentais do modelo vivem na dataclass :class:`Params`,
com valores *default* declarados e documentados. Um único ``seed`` controla toda
a estocasticidade (amostragem de conjuntos de escolha).

Convenções de unidades
-----------------------
* Utilidade é adimensional (soma de log-salário, aluguel e amenidade).
* População/fluxos estão numa unidade consistente arbitrária. O dataset
  sintético usa "milhares de habitantes", de modo que ``ln(1 + N_ij)`` fique
  numa faixa numérica confortável. Como o termo de rede usa ``N_ij`` em nível,
  o modelo é sensível à escala absoluta da população — ver README.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class Params:
    """Parâmetros do motor de simulação.

    Atributos (todos com default documentado):

    Congestão e preços
        alpha : elasticidade de congestão do salário. ``w_j = A_j (L_j/L0)^(-alpha)``.
                ``alpha > 0`` => salário cai quando a população cresce (deseconomia
                de aglomeração). ``alpha = 0`` => sem congestão salarial.
        beta  : elasticidade do aluguel à população. ``h_j = h0 (L_j/L0)^beta``.
        h0    : nível-base de aluguel (em unidades de utilidade).
        u0    : desemprego de base.
        gamma : sensibilidade do desemprego esperado à taxa de entrada recente.
                ``u_j = clip(u0 + gamma * taxa_entrada_j, 0, 0.45)``.

    Escolha e mobilidade
        kappa : desutilidade de distância por unidade de distância normalizada.
        lam   : força do efeito de rede (diáspora): ``lam * ln(1 + N_ij)``.
        theta : escala do logit (temperatura inversa). Maior => escolhas mais
                determinísticas.
        F     : custo fixo de migração (unidades de utilidade).
        m0    : taxa-base de mobilidade — fração da população que entra no
                conjunto de escolha a cada período.

    Restrição de liquidez
        w_F          : salário de referência (piso) usado na sigmoide de liquidez.
        k_liq        : inclinação da sigmoide de liquidez.
        use_liquidity: liga/desliga a restrição de liquidez (flag).

    Amostragem do conjunto de escolha
        k : número de destinos amostrados por origem (sorteio ponderado por
            ``1/d_ij``). ``None`` recupera o caso completo (todos os ``n-1``
            destinos). ``k = n-1`` também recupera o caso completo.

    Normalização e reprodutibilidade
        L0   : população de referência das fórmulas de salário/aluguel.
               ``None`` => usa a média das populações iniciais.
        seed : semente-mestra que controla TODA a estocasticidade.
    """

    # --- Congestão / preços ---
    alpha: float = 0.15
    beta: float = 0.30
    h0: float = 0.30
    u0: float = 0.05
    gamma: float = 0.50

    # --- Escolha / mobilidade ---
    kappa: float = 0.50
    lam: float = 0.30
    theta: float = 2.0
    F: float = 0.20
    m0: float = 0.10

    # --- Liquidez ---
    w_F: float = 1.0
    k_liq: float = 3.0
    use_liquidity: bool = True

    # --- Amostragem do conjunto de escolha ---
    k: int | None = None

    # --- Normalização / reprodutibilidade ---
    L0: float | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        # Validações leves — evitam configurações sem sentido físico.
        if not (0.0 <= self.m0 <= 1.0):
            raise ValueError("m0 deve estar em [0, 1] (fração da população).")
        if self.theta <= 0:
            raise ValueError("theta (escala do logit) deve ser > 0.")
        if self.k is not None and self.k < 1:
            raise ValueError("k deve ser >= 1 ou None (caso completo).")
        if self.L0 is not None and self.L0 <= 0:
            raise ValueError("L0 deve ser > 0.")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Retorna os parâmetros como dicionário (útil para logging/JSON)."""
        return asdict(self)

    def with_(self, **overrides: Any) -> "Params":
        """Cria uma cópia com alguns campos sobrescritos (imutável)."""
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise KeyError(f"Parâmetros desconhecidos: {sorted(unknown)}")
        return replace(self, **overrides)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Params":
        """Constrói a partir de um dicionário, ignorando chaves extras."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
