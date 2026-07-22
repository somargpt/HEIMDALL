# migracao — Migração interna dirigida por diferencial de riqueza

Modelo computacional de **migração interna** entre cidades, formulado como
**escolha discreta (logit multinomial)** sobre um grafo de cidades. A cada
período, os habitantes de cada cidade escolhem entre *ficar* ou *migrar* para um
conjunto de destinos, comparando salário, aluguel, desemprego esperado,
amenidades, distância e o efeito de rede (diáspora). Congestão (salário que cai
com a população) e restrição de liquidez fecham os laços de realimentação.

Sem frameworks pesados: apenas `numpy`, `pandas`, `scipy`, `matplotlib`.
Testes com `pytest`.

> **Aviso sobre dados.** Este repositório **não contém dados reais nem constantes
> calibradas**. O dataset de 14 cidades em `data/cidades.csv` é **sintético**
> (gerado deterministicamente em `migracao/cities.py`). Os valores *default* dos
> parâmetros são escolhas didáticas plausíveis, **não** estimativas empíricas.

---

## O modelo

Cada cidade `j` tem produtividade `A_j`, população `L_j`, amenidade `a_j` e
coordenadas `(x_j, y_j)`. A cada período `t`:

| # | Quantidade | Fórmula |
|---|------------|---------|
| 1 | Salário | `w_j = A_j · (L_j/L0)^(-alpha)` |
| 2 | Aluguel | `h_j = h0 · (L_j/L0)^beta` |
| 3 | Desemprego esperado | `u_j = clip(u0 + gamma · taxa_entrada_recente_j, 0, 0.45)` |
| 4 | Valor do lugar | `V_j = ln(w_j·(1−u_j)) − h_j + a_j` |
| 5 | Utilidade do par | `U_ij = V_j − kappa·d_ij − F + lam·ln(1+N_ij)` |
| 6 | Ficar em `i` | `U_ii = V_i` (sem custo, sem rede) |
| 7 | Probabilidade | logit `P_ij = softmax_j(theta·U_ij)` sobre `{ficar} ∪ destinos` |
| 8 | Liquidez | `s_i = sigmoid((w_i − w_F)·k_liq)` (desligável) |
| 9 | Fluxo | `Fluxo_ij = L_i·m0·s_i·P_ij` → atualiza `L`, `N`, taxa de entrada |

- **`d_ij`**: distância euclidiana **normalizada** pela maior distância par-a-par
  (fica em `[0,1]`); a métrica de "distância média percorrida" usa a distância
  **física**.
- **`N_ij`**: estoque **acumulado** de migrantes de `i` em `j` (rede social /
  diáspora); cresce com os fluxos e entra via `ln(1+N_ij)`.
- **`taxa_entrada_recente_j`**: `inflow_j / L_j` do **período anterior**.
- **`L0`**: população de referência das fórmulas de preço. *Default* = média das
  populações iniciais (configurável em `Params.L0`).
- **Conjunto de escolha**: para cada origem amostram-se `k` destinos por sorteio
  ponderado por `1/d_ij` (informação imperfeita). `k = None` (ou `k ≥ n−1`)
  recupera o **caso completo**. A amostragem sem reposição usa chaves de
  Efraimidis–Spirakis (`log(U)/w`, top-k por linha), totalmente vetorizada.

A **matriz de utilidade `n×n`** é construída sem laço duplo em Python; o softmax é
estável (subtrai o máximo da linha; `−inf` fora do conjunto de escolha vira 0).

### Conservação
Sem natalidade/mortalidade. A atualização `L ← L − outflow + inflow` (com `outflow`
= somas por linha e `inflow` = somas por coluna da matriz de fluxos, diagonal
zerada) **conserva a população exatamente por construção** (deriva de ~1e-13 em
200 períodos).

---

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, scipy, matplotlib, pytest
# opcional: instalar o pacote (expõe o console script `migracao`)
pip install -e .
```

Requer **Python 3.11+**.

---

## Uso rápido (API)

```python
from migracao import Params, Simulator, load_cities

cities = load_cities()                     # dataset sintético de 14 cidades
res = Simulator(cities, Params()).run(periodos=200)

res.panel      # (t, cidade, L, w, h, u)
res.flows      # (t, origem, destino, fluxo)  — migração efetiva (fora da diagonal)
res.metrics    # (t, gini, share_maior, hhi, taxa_migracao, distancia_media)
res.od_accumulated  # matriz origem-destino acumulada (n×n)
```

Carregar dados reais (mesmo formato do CSV):

```python
from migracao import load_cities
cities = load_cities("caminho/para/cidades.csv")
```

Formato do CSV: `nome, x, y, A, amenidade, populacao_inicial`.

---

## Parâmetros (`migracao.Params`)

Todos os parâmetros comportamentais vivem numa `dataclass` com *defaults*
documentados. Uma **única `seed`** controla toda a estocasticidade.

| Grupo | Parâmetro | Default | Significado |
|-------|-----------|---------|-------------|
| Congestão/preços | `alpha` | 0.15 | elasticidade de congestão do salário |
| | `beta` | 0.30 | elasticidade do aluguel à população |
| | `h0` | 0.30 | nível-base de aluguel |
| | `u0` | 0.05 | desemprego de base |
| | `gamma` | 0.50 | sensibilidade do desemprego à entrada recente |
| Escolha/mobilidade | `kappa` | 0.50 | desutilidade de distância |
| | `lam` | 0.30 | força do efeito de rede (diáspora) |
| | `theta` | 2.0 | escala do logit (temperatura inversa) |
| | `F` | 0.20 | custo fixo de migração |
| | `m0` | 0.10 | taxa-base de mobilidade |
| Liquidez | `w_F` | 1.0 | salário de referência da sigmoide |
| | `k_liq` | 3.0 | inclinação da sigmoide |
| | `use_liquidity` | `True` | liga/desliga a restrição |
| Conjunto de escolha | `k` | `None` | nº de destinos amostrados (`None` = completo) |
| Normalização | `L0` | `None` | pop. de referência (`None` = média inicial) |
| Reprodutibilidade | `seed` | 42 | semente-mestra |

---

## Testes

```bash
python -m pytest -q
```

As **verificações obrigatórias** (`tests/`):

1. **Conservação** (`test_conservacao.py`) — soma de `L` constante a cada período,
   tolerância `1e-9`, em várias configurações.
2. **Monotonicidade** (`test_monotonicidade.py`) — `kappa` maior ⇒ distância média
   dos fluxos menor.
3. **Congestão** (`test_congestao.py`) — `alpha=0` colapsa para o **canto**
   (concentração extrema, poucas cidades); `alpha` alto converge para um
   equilíbrio **interior** disperso (variação de `L` < epsilon por 20 períodos).
4. **Liquidez** (`test_liquidez.py`) — desligar a restrição aumenta a saída das
   cidades de menor salário.
5. **Path dependence** (`test_path_dependence.py`) — com `lam` alto, dois choques
   iniciais distintos levam a matrizes O-D divergentes; com `lam` baixo, dissipam.
6. **Logit** (`test_logit.py`) — probabilidades somam 1 por origem; sem NaN com
   `w ≈ 0`.

Mais testes auxiliares cobrem métricas, o loader de CSV, casos de borda (cidade
vazia, cidades coincidentes, valores não-finitos) e sanidade do logit.

### Nota metodológica sobre o teste de congestão
Num modelo de **população finita** com termo de rede **log-saturante**
(`ln(1+N)`), o regime `alpha=0` acaba *congelando* no canto após um horizonte
longo — logo, afirmar "nunca converge" seria um artefato da escolha de horizonte.
O teste verifica, em vez disso, a propriedade **robusta e invariante ao
horizonte** que de fato distingue os regimes: o **nível de concentração de
equilíbrio** — canto (poucas cidades, HHI alto) sob `alpha=0` vs. interior
(todas as cidades vivas, HHI baixo) sob `alpha` alto.

---

## Limitações conhecidas

- **População homogênea.** Todos os habitantes de uma origem são idênticos; não há
  heterogeneidade de renda, idade, qualificação ou preferências (a liquidez atua
  como uma fração agregada `s_i`, não sobre indivíduos).
- **Sem retorno migratório explícito.** `N_ij` é um estoque **acumulado**
  (monotônico) de fluxos históricos; não há decaimento da diáspora nem migração de
  retorno modelada explicitamente.
- **Sem demografia.** Sem natalidade, mortalidade ou envelhecimento — a população
  total é conservada.
- **Amenidades exógenas.** `a_j` é fixo; não responde à população nem à
  composição da cidade.
- **Sensibilidade de escala do termo de rede.** `lam·ln(1+N_ij)` usa `N` em
  **nível**; a escala absoluta da população/fluxos afeta o peso da rede. O dataset
  sintético usa "milhares de habitantes" para manter `ln(1+N)` numa faixa
  confortável. Ao usar dados reais, `theta`/`lam` devem ser recalibrados à escala.
- **Dados sintéticos.** Nenhuma constante é calibrada empiricamente (ver o aviso no
  topo). A calibração por máxima verossimilhança é uma etapa separada (`calibra.py`,
  em construção).

---

## Estrutura

```
migracao/
  params.py     Params (dataclass) + defaults documentados
  cities.py     Cities, loader de CSV, dataset sintético
  engine.py     Simulator (motor vetorizado) + SimulationResult
  metrics.py    Gini, HHI, share, taxa bruta, distância média, O-D, CPC
data/
  cidades.csv   dataset SINTÉTICO de 14 cidades
tests/          verificações obrigatórias + auxiliares
```

> Itens em construção (próximas etapas): CLI (`python -m migracao run ...`),
> gráficos, varredura de sensibilidade, baselines (gravitacional e *radiation
> model*) e o módulo de calibração `calibra.py`.
