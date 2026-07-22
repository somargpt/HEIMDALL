"""Testes da varredura de sensibilidade, dos gráficos e da CLI."""

from __future__ import annotations

import numpy as np

from migracao import Params, load_cities
from migracao.sensitivity import sweep_2d


# --------------------------- sensibilidade ---------------------------
def test_sweep_shape_e_reprodutibilidade(cities):
    base = Params(seed=1)
    sw1 = sweep_2d(cities, base, "lam", [0.0, 0.5, 1.0], "alpha", [0.0, 0.4],
                   periodos=60, metric="gini")
    sw2 = sweep_2d(cities, base, "lam", [0.0, 0.5, 1.0], "alpha", [0.0, 0.4],
                   periodos=60, metric="gini")
    assert sw1.grid.shape == (2, 3)  # (len(alpha), len(lam))
    assert np.allclose(sw1.grid, sw2.grid)  # reprodutível


def test_gini_cresce_com_lam_e_cai_com_alpha(cities):
    base = Params(seed=1)
    sw = sweep_2d(cities, base, "lam", [0.0, 0.5, 1.0, 1.5], "alpha", [0.0, 0.4, 0.8],
                  periodos=100, metric="gini")
    # Efeito robusto: lam alto aumenta claramente a concentração (comparação de
    # extremos por linha de alpha). Perto de lam=0 há wiggle numérico desprezível.
    for row in sw.grid:
        assert row[-1] > row[0] + 0.1, f"lam alto não aumentou a concentração: {row}"
    # Congestão (alpha) reduz a concentração: no maior lam, Gini cai de alpha
    # mínimo para alpha máximo.
    assert sw.grid[0, -1] > sw.grid[-1, -1]


# --------------------------- gráficos ---------------------------
def test_plots_geram_arquivos(cities, tmp_path):
    from migracao import Simulator
    from migracao import plots
    from migracao.baselines import compare_against_baselines
    from migracao.sensitivity import sweep_2d

    res = Simulator(cities, Params(seed=1)).run(40)
    p1 = plots.plot_evolution(res, tmp_path / "ev.png")
    p2 = plots.plot_flow_map(res, cities, tmp_path / "map.png")
    sw = sweep_2d(cities, Params(seed=1), "kappa", [0.2, 1.0], "alpha", [0.0, 0.5],
                  periodos=30)
    p3 = plots.plot_sensitivity_heatmap(sw, tmp_path / "hm.png")
    od = res.od_accumulated
    _, preds, _ = compare_against_baselines(od, cities.L_init,
                                            cities.distance_matrix(normalize=False))
    p4 = plots.plot_baseline_comparison(od, preds, tmp_path / "bl.png")
    for p in (p1, p2, p3, p4):
        assert p.exists() and p.stat().st_size > 0


# --------------------------- CLI ---------------------------
def test_cli_run(tmp_path):
    from migracao.cli import main
    out = tmp_path / "run"
    rc = main(["run", "--periodos", "20", "--out", str(out), "--no-plots", "--seed", "1"])
    assert rc == 0
    for f in ["panel.csv", "flows.csv", "metrics.csv", "od_matrix.csv", "params.json"]:
        assert (out / f).exists()


def test_cli_gen_data_e_baselines(tmp_path):
    from migracao.cli import main
    csv = tmp_path / "c.csv"
    assert main(["gen-data", "--out", str(csv)]) == 0
    assert csv.exists()
    out = tmp_path / "bl"
    assert main(["baselines", "--cidades", str(csv), "--periodos", "40",
                 "--out", str(out)]) == 0
    assert (out / "baseline_comparison.csv").exists()


def test_cli_sweep_custom(tmp_path):
    from migracao.cli import main
    out = tmp_path / "sw"
    rc = main(["sweep", "--periodos", "25", "--x", "kappa", "--xvals", "0.2,1.0",
               "--y", "lam", "--yvals", "0.0,1.0", "--out", str(out)])
    assert rc == 0
    assert (out / "sweep_gini_kappa_x_lam.csv").exists()
    assert (out / "heatmap_gini_kappa_x_lam.png").exists()
