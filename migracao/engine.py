"""Motor da simulação de migração interna dirigida por diferencial de riqueza.

Modelo de escolha discreta (logit multinomial) sobre um grafo de cidades. A cada
período, para cada origem, o conjunto de escolha é ``{ficar} ∪ destinos``, onde os
destinos podem ser amostrados por informação imperfeita (sorteio ponderado por
``1/d_ij``). A matriz de utilidade ``n x n`` é construída de forma totalmente
vetorizada (sem laço duplo em Python).

Sequência de um período ``t`` (estado entra com ``L`` e ``taxa_entrada`` do período
anterior; ``N`` acumula a história):

1. ``w_j = A_j (L_j/L0)^(-alpha)``
2. ``h_j = h0 (L_j/L0)^beta``
3. ``u_j = clip(u0 + gamma * taxa_entrada_j, 0, 0.45)``   (taxa do período anterior)
4. ``V_j = ln(w_j (1 - u_j)) - h_j + a_j``
5. ``U_ij = V_j - kappa d_ij - F + lam ln(1 + N_ij)`` ; ficar em ``i`` tem ``V_i``
6. amostra do conjunto de escolha (ou caso completo)
7. logit com escala ``theta`` sobre ``{ficar} ∪ destinos`` -> ``P_ij``
8. liquidez ``s_i = sigmoid((w_i - w_F) k_liq)`` (desligável)
9. ``Fluxo_ij = L_i m0 s_i P_ij`` -> atualiza ``L`` (conserva população), ``N`` e a
   taxa de entrada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from .cities import Cities
from .metrics import (
    gini,
    gross_migration_rate,
    hhi,
    largest_share,
    mean_flow_distance,
)
from .params import Params

# Piso numérico dentro do log de salário (evita -inf/NaN com w ~ 0).
_LOG_FLOOR = 1e-12
# Piso da razão L/L0 nas equações de preço: evita salário infinito (divisão por
# zero em (L/L0)^(-alpha)) numa cidade que ficou completamente vazia. É um piso de
# segurança negligível (uma cidade esvaziada tem outflow ~0 de qualquer modo).
_RATIO_FLOOR = 1e-9
# Piso de distância no PESO de amostragem (1/d): cidades coincidentes (d=0 fora da
# diagonal) recebem peso máximo (sempre sorteadas), não zero.
_DIST_FLOOR = 1e-9
# Teto de desemprego esperado (do enunciado).
_U_MAX = 0.45


@dataclass
class SimulationResult:
    """Saída de uma simulação.

    Atributos
        panel   : DataFrame longo (t, cidade, L, w, h, u) para t = 0..T.
        flows   : DataFrame longo (t, origem, destino, fluxo) para t = 0..T-1,
                  apenas fluxos fora da diagonal (migração efetiva) e > 0.
        metrics : DataFrame por período (t, gini, share_maior, hhi,
                  taxa_migracao, distancia_media).
        od_accumulated : matriz O-D acumulada ``n x n`` (np.ndarray).
        names   : nomes das cidades (ordem dos índices).
        distance: matriz de distâncias normalizadas usada na simulação.
        L0      : população de referência efetivamente usada.
        params  : parâmetros usados.
    """

    panel: pd.DataFrame
    flows: pd.DataFrame
    metrics: pd.DataFrame
    od_accumulated: np.ndarray
    names: list[str]
    distance: np.ndarray
    L0: float
    params: Params

    def od_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.od_accumulated, index=self.names, columns=self.names)


class Simulator:
    """Motor vetorizado. Construído com ``Cities`` + ``Params``; ``run`` roda T períodos."""

    def __init__(self, cities: Cities, params: Params | None = None):
        self.cities = cities
        self.params = params or Params()
        self.n = cities.n
        # Distância normalizada (usada na utilidade) e física (usada na métrica).
        self.d_norm = cities.distance_matrix(normalize=True)
        self.d_phys = cities.distance_matrix(normalize=False)
        # Pesos de amostragem do conjunto de escolha: 1/d (fora da diagonal).
        # Piso de distância => cidades coincidentes (d=0) recebem peso máximo em vez
        # de zero, de modo que possam ser sorteadas quando k < n-1.
        w = 1.0 / np.maximum(self.d_norm, _DIST_FLOOR)
        np.fill_diagonal(w, 0.0)
        self._sample_weights = w
        # População de referência L0: default = média das populações iniciais.
        self.L0 = float(self.params.L0) if self.params.L0 is not None else float(np.mean(cities.L_init))
        if self.L0 <= 0:
            raise ValueError("L0 (média das populações iniciais) deve ser > 0.")

    # ------------------------------------------------------------------
    # Blocos vetorizados
    # ------------------------------------------------------------------
    def _place_value(self, L: np.ndarray, entry_rate: np.ndarray):
        """Retorna (V, w, h, u) para todas as cidades, dado o estado ``L``."""
        p = self.params
        ratio = np.maximum(L / self.L0, _RATIO_FLOOR)
        w = self.cities.A * np.power(ratio, -p.alpha)
        h = p.h0 * np.power(ratio, p.beta)
        u = np.clip(p.u0 + p.gamma * entry_rate, 0.0, _U_MAX)
        # Piso no argumento do log => sem NaN mesmo com w ~ 0.
        inside = np.maximum(w * (1.0 - u), _LOG_FLOOR)
        V = np.log(inside) - h + self.cities.amenity
        return V, w, h, u

    def _choice_mask(self, rng: np.random.Generator) -> np.ndarray:
        """Máscara booleana ``(n, n)`` do conjunto de escolha por origem.

        A diagonal (ficar) é sempre ``True``. Para ``k`` destinos, amostra sem
        reposição ponderado por ``1/d_ij`` via chaves de Efraimidis-Spirakis
        (vetorizado, top-k por linha). ``k = None`` ou ``k >= n-1`` => caso completo.
        """
        n = self.n
        k = self.params.k
        mask = np.zeros((n, n), dtype=bool)
        np.fill_diagonal(mask, True)  # ficar sempre disponível
        if k is None or k >= n - 1:
            mask[:] = True  # caso completo: todos os destinos
            return mask
        # Chaves log de Efraimidis-Spirakis: key = log(U)/w ; maiores => escolhidos.
        u = rng.random((n, n))
        with np.errstate(divide="ignore", invalid="ignore"):
            keys = np.log(u) / self._sample_weights  # w=0 na diagonal => +/-inf
        np.fill_diagonal(keys, -np.inf)  # nunca escolhe a própria origem como destino
        # top-k colunas por linha (maiores chaves).
        topk = np.argpartition(keys, -k, axis=1)[:, -k:]
        rows = np.repeat(np.arange(n), k)
        mask[rows, topk.ravel()] = True
        np.fill_diagonal(mask, True)
        return mask

    def _probabilities(self, V, w, N, mask):
        """Matriz de probabilidades ``P_ij`` (linhas somam 1) via logit multinomial.

        Constrói a matriz de utilidade ``n x n`` vetorizada, aplica a máscara do
        conjunto de escolha e faz softmax estável (subtrai o máximo da linha) com
        escala ``theta``.
        """
        p = self.params
        # U[i,j] = V[j] - kappa d_ij - F + lam ln(1+N_ij) ; diagonal sobrescrita = V[i].
        U = V[None, :] - p.kappa * self.d_norm - p.F + p.lam * np.log1p(N)
        np.fill_diagonal(U, V)  # ficar em i: utilidade V_i (sem custo, sem rede)
        Z = p.theta * U
        Z = np.where(mask, Z, -np.inf)
        Zmax = Z.max(axis=1, keepdims=True)  # finito: diagonal sempre no conjunto
        E = np.exp(Z - Zmax)  # -inf -> 0
        P = E / E.sum(axis=1, keepdims=True)
        return P

    def _liquidity(self, w: np.ndarray) -> np.ndarray:
        """Fração da população capaz de pagar F: ``s_i = sigmoid((w_i - w_F) k_liq)``."""
        p = self.params
        if not p.use_liquidity:
            return np.ones(self.n, dtype=float)
        # expit é numericamente estável (não transborda para w << w_F).
        return expit((w - p.w_F) * p.k_liq)

    # ------------------------------------------------------------------
    def choice_probabilities(
        self,
        L: np.ndarray,
        entry_rate: np.ndarray | None = None,
        N: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Matriz ``P_ij`` (linhas somam 1) para um estado arbitrário.

        Método público: útil para testes e para a calibração (avaliar o logit num
        estado observado). Se ``rng`` não for dado, usa a semente dos parâmetros.
        """
        L = np.asarray(L, dtype=float)
        if entry_rate is None:
            entry_rate = np.zeros(self.n, dtype=float)
        if N is None:
            N = np.zeros((self.n, self.n), dtype=float)
        if rng is None:
            rng = np.random.default_rng(self.params.seed)
        V, w, _h, _u = self._place_value(L, entry_rate)
        mask = self._choice_mask(rng)
        return self._probabilities(V, w, N, mask)

    # ------------------------------------------------------------------
    def run(
        self,
        periodos: int,
        initial_N: np.ndarray | None = None,
        initial_L: np.ndarray | None = None,
    ) -> SimulationResult:
        """Roda ``periodos`` períodos e retorna :class:`SimulationResult`.

        Parâmetros
            periodos   : número de períodos T.
            initial_N  : estoque inicial de diáspora ``N_ij`` (n x n). Default = 0.
                         Serve para introduzir "choques" de rede (path dependence).
            initial_L  : população inicial alternativa (n,). Default = cities.L_init.
        """
        if periodos < 1:
            raise ValueError("periodos deve ser >= 1.")
        n = self.n
        names = self.cities.names
        rng = np.random.default_rng(self.params.seed)

        L = (self.cities.L_init.copy() if initial_L is None
             else np.asarray(initial_L, dtype=float).copy())
        if L.shape != (n,):
            raise ValueError("initial_L deve ter shape (n,).")
        N = (np.zeros((n, n), dtype=float) if initial_N is None
             else np.asarray(initial_N, dtype=float).copy())
        if N.shape != (n, n):
            raise ValueError("initial_N deve ter shape (n, n).")
        entry_rate = np.zeros(n, dtype=float)  # taxa de entrada do período anterior

        total0 = float(L.sum())
        od_acc = np.zeros((n, n), dtype=float)

        panel_rows: list[dict] = []
        metric_rows: list[dict] = []
        # Buffers para o DataFrame de fluxos (construído de uma vez no final).
        f_t: list[np.ndarray] = []
        f_i: list[np.ndarray] = []
        f_j: list[np.ndarray] = []
        f_val: list[np.ndarray] = []

        def record_panel(t: int, L_, w_, h_, u_):
            for j in range(n):
                panel_rows.append({
                    "t": t, "cidade": names[j],
                    "L": float(L_[j]), "w": float(w_[j]),
                    "h": float(h_[j]), "u": float(u_[j]),
                })

        for t in range(periodos):
            V, w, h, u = self._place_value(L, entry_rate)
            record_panel(t, L, w, h, u)

            mask = self._choice_mask(rng)
            P = self._probabilities(V, w, N, mask)
            s = self._liquidity(w)

            # Pool de mobilidade e fluxos.
            M = L * self.params.m0 * s               # (n,)
            Flow = M[:, None] * P                    # (n, n)
            np.fill_diagonal(Flow, 0.0)              # só migração efetiva (fora da diagonal)

            outflow = Flow.sum(axis=1)               # sai de i
            inflow = Flow.sum(axis=0)                # chega em j

            # Métricas do período (usam o estado que gerou os fluxos).
            metric_rows.append({
                "t": t,
                "gini": gini(L),
                "share_maior": largest_share(L),
                "hhi": hhi(L),
                "taxa_migracao": gross_migration_rate(Flow, float(L.sum())),
                "distancia_media": mean_flow_distance(Flow, self.d_phys),
            })

            # Registra fluxos > 0 (fora da diagonal).
            ii, jj = np.nonzero(Flow)
            if ii.size:
                f_t.append(np.full(ii.size, t))
                f_i.append(ii)
                f_j.append(jj)
                f_val.append(Flow[ii, jj])
            od_acc += Flow

            # Atualização de estado (conserva população por construção).
            new_entry_rate = np.divide(
                inflow, L, out=np.zeros_like(L), where=L > 0
            )
            L = L - outflow + inflow
            N = N + Flow
            entry_rate = new_entry_rate

        # Estado final (t = T): painel sem fluxos.
        V, w, h, u = self._place_value(L, entry_rate)
        record_panel(periodos, L, w, h, u)

        panel = pd.DataFrame(panel_rows, columns=["t", "cidade", "L", "w", "h", "u"])
        metrics = pd.DataFrame(
            metric_rows,
            columns=["t", "gini", "share_maior", "hhi", "taxa_migracao", "distancia_media"],
        )
        names_arr = np.asarray(names, dtype=object)
        if f_t:
            t_all = np.concatenate(f_t)
            i_all = np.concatenate(f_i)
            j_all = np.concatenate(f_j)
            v_all = np.concatenate(f_val)
            flows = pd.DataFrame({
                "t": t_all,
                "origem": names_arr[i_all],
                "destino": names_arr[j_all],
                "fluxo": v_all,
            })
        else:
            flows = pd.DataFrame(columns=["t", "origem", "destino", "fluxo"])

        # Checagem barata de conservação (defensiva, além do teste dedicado).
        assert abs(float(L.sum()) - total0) < 1e-6 * max(1.0, total0), "violação de conservação"

        return SimulationResult(
            panel=panel,
            flows=flows,
            metrics=metrics,
            od_accumulated=od_acc,
            names=list(names),
            distance=self.d_norm,
            L0=self.L0,
            params=self.params,
        )


def simulate(
    cities: Cities,
    params: Params | None = None,
    periodos: int = 200,
    **run_kwargs,
) -> SimulationResult:
    """Atalho funcional: constrói o ``Simulator`` e roda ``periodos`` períodos."""
    return Simulator(cities, params).run(periodos, **run_kwargs)
