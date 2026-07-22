"""Interface de linha de comando.

Uso principal:

    python -m migracao run --cidades data/cidades.csv --periodos 200 --out out/

Subcomandos: ``run`` (simula, salva painel/fluxos/métricas/O-D e figuras),
``sweep`` (varredura de sensibilidade => heatmaps), ``baselines`` (ajuste
gravitacional e radiation à matriz O-D) e ``gen-data`` (escreve o CSV sintético).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import compare_against_baselines
from .cities import load_cities, write_synthetic_csv
from .engine import Simulator
from .params import Params
from .sensitivity import sweep_2d

# Parâmetros comportamentais expostos como flags --<nome>.
_FLOAT_PARAMS = ["alpha", "beta", "h0", "u0", "gamma", "kappa", "lam", "theta",
                 "F", "m0", "w_F", "k_liq", "L0"]


def _add_param_overrides(parser: argparse.ArgumentParser) -> None:
    for name in _FLOAT_PARAMS:
        parser.add_argument(f"--{name}", type=float, default=None,
                            help=f"sobrescreve o parâmetro {name}")
    parser.add_argument("--k", type=int, default=None,
                        help="nº de destinos amostrados (default: caso completo)")
    parser.add_argument("--seed", type=int, default=None, help="semente-mestra")
    parser.add_argument("--no-liquidity", action="store_true",
                        help="desliga a restrição de liquidez")


def _params_from_args(args) -> Params:
    overrides = {}
    for name in _FLOAT_PARAMS:
        v = getattr(args, name, None)
        if v is not None:
            overrides[name] = v
    if getattr(args, "k", None) is not None:
        overrides["k"] = args.k
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "no_liquidity", False):
        overrides["use_liquidity"] = False
    return Params().with_(**overrides)


# ----------------------------------------------------------------------
def cmd_run(args) -> None:
    cities = load_cities(args.cidades)
    params = _params_from_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    res = Simulator(cities, params).run(args.periodos)

    res.panel.to_csv(out / "panel.csv", index=False)
    res.flows.to_csv(out / "flows.csv", index=False)
    res.metrics.to_csv(out / "metrics.csv", index=False)
    res.od_dataframe().to_csv(out / "od_matrix.csv")
    with open(out / "params.json", "w") as fh:
        json.dump({**params.to_dict(), "periodos": args.periodos,
                   "L0_efetivo": res.L0}, fh, indent=2)

    print(f"[run] {args.periodos} períodos, {cities.n} cidades -> {out}/")
    print(f"      Gini final={res.metrics['gini'].iloc[-1]:.4f}  "
          f"share_maior={res.metrics['share_maior'].iloc[-1]:.4f}  "
          f"HHI={res.metrics['hhi'].iloc[-1]:.4f}")

    if not args.no_plots:
        from . import plots
        plots.plot_evolution(res, out / "fig_evolution.png")
        plots.plot_flow_map(res, cities, out / "fig_flow_map.png")
        print(f"      figuras: fig_evolution.png, fig_flow_map.png")


# ----------------------------------------------------------------------
def _parse_values(spec: str) -> np.ndarray:
    """Aceita 'a,b,c' (lista) ou 'min:max:n' (linspace)."""
    spec = spec.strip()
    if ":" in spec:
        lo, hi, num = spec.split(":")
        return np.linspace(float(lo), float(hi), int(num))
    return np.array([float(x) for x in spec.split(",")], dtype=float)


def cmd_sweep(args) -> None:
    cities = load_cities(args.cidades)
    base = _params_from_args(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from . import plots

    if args.x and args.y:
        pairs = [(args.x, _parse_values(args.xvals), args.y, _parse_values(args.yvals))]
    else:
        # Três heatmaps canônicos em alpha, kappa e lam (Gini final).
        a = np.linspace(0.0, 0.8, 5)
        k = np.linspace(0.1, 3.0, 5)
        lm = np.linspace(0.0, 1.5, 5)
        pairs = [("kappa", k, "alpha", a), ("lam", lm, "alpha", a),
                 ("lam", lm, "kappa", k)]

    for px, vx, py, vy in pairs:
        sw = sweep_2d(cities, base, px, vx, py, vy,
                      periodos=args.periodos, metric=args.metric)
        tag = f"{args.metric}_{px}_x_{py}"
        sw.to_dataframe().to_csv(out / f"sweep_{tag}.csv", index=False)
        plots.plot_sensitivity_heatmap(sw, out / f"heatmap_{tag}.png")
        print(f"[sweep] {tag}: {args.metric} em "
              f"[{sw.grid.min():.3f}, {sw.grid.max():.3f}] -> heatmap_{tag}.png")


# ----------------------------------------------------------------------
def cmd_baselines(args) -> None:
    cities = load_cities(args.cidades)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dphys = cities.distance_matrix(normalize=False)

    if args.od:
        od = pd.read_csv(args.od, index_col=0).to_numpy(dtype=float)
    else:
        params = _params_from_args(args)
        od = Simulator(cities, params).run(args.periodos).od_accumulated

    table, preds, grav = compare_against_baselines(od, cities.L_init, dphys)
    table.to_csv(out / "baseline_comparison.csv", index=False)
    with open(out / "gravity_fit.json", "w") as fh:
        json.dump(grav.summary(), fh, indent=2)

    from . import plots
    plots.plot_baseline_comparison(od, {"modelo (O-D)": od, **preds},
                                   out / "fig_baselines.png")
    print("[baselines] ajuste à matriz O-D:")
    print(table.to_string(index=False))
    print(f"            gravitacional: b={grav.b:.3f} -> {out}/")


# ----------------------------------------------------------------------
def cmd_gen_data(args) -> None:
    path = write_synthetic_csv(args.out, seed=args.seed if args.seed is not None else 7)
    print(f"[gen-data] dataset sintético (14 cidades) -> {path}")


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="migracao",
                                description="Modelo de migração interna por diferencial de riqueza.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="simula e salva saídas + figuras")
    pr.add_argument("--cidades", default=None, help="CSV de cidades (default: sintético)")
    pr.add_argument("--periodos", type=int, default=200)
    pr.add_argument("--out", default="out")
    pr.add_argument("--no-plots", action="store_true", help="não gerar figuras")
    _add_param_overrides(pr)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("sweep", help="varredura de sensibilidade (heatmaps)")
    ps.add_argument("--cidades", default=None)
    ps.add_argument("--periodos", type=int, default=200)
    ps.add_argument("--out", default="out")
    ps.add_argument("--metric", default="gini", choices=["gini", "hhi", "share_maior"])
    ps.add_argument("--x", default=None, help="parâmetro do eixo x")
    ps.add_argument("--xvals", default=None, help="'a,b,c' ou 'min:max:n'")
    ps.add_argument("--y", default=None, help="parâmetro do eixo y")
    ps.add_argument("--yvals", default=None, help="'a,b,c' ou 'min:max:n'")
    _add_param_overrides(ps)
    ps.set_defaults(func=cmd_sweep)

    pb = sub.add_parser("baselines", help="ajuste gravitacional e radiation à O-D")
    pb.add_argument("--cidades", default=None)
    pb.add_argument("--periodos", type=int, default=200)
    pb.add_argument("--out", default="out")
    pb.add_argument("--od", default=None, help="CSV de matriz O-D observada (opcional)")
    _add_param_overrides(pb)
    pb.set_defaults(func=cmd_baselines)

    pg = sub.add_parser("gen-data", help="escreve o CSV sintético de 14 cidades")
    pg.add_argument("--out", default="data/cidades.csv")
    pg.add_argument("--seed", type=int, default=None)
    pg.set_defaults(func=cmd_gen_data)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0
