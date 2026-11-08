"""
Експерименти зі звіту:
  python -m src.experiments main     — 10 запусків на кожній тестовій задачі (табл. 7)
  python -m src.experiments params   — підбір параметрів на ESC25 (табл. 8)
  python -m src.experiments all
Результати: results/tables/*.csv
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from .sop_instance import load_named, greedy_nearest_feasible, brute_force, exact_dp, REPORT_SET
from .mmas_sop import MMAS_SOP, MMASConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")
DATA = os.path.join(ROOT, "data")
N_RUNS = 10
GREEDY = "Жадібний"


def optimum(inst):
    """Оптимум: з TSPLIB або повним перебором (для прикладів 1 і 2)."""
    if inst.best_known is not None:
        return inst.best_known
    return brute_force(inst)[1] if inst.n <= 10 else exact_dp(inst)[1]


def base_config(n: int) -> MMASConfig:
    """Базові параметри зі звіту: α=1, β=2, ρ=0.05, p_best=0.05, m=min(n,25);
    200 ітерацій для n <= 20 і 500 для більших задач."""
    return MMASConfig(n_ants=min(n, 25), n_iterations=200 if n <= 20 else 500,
                      alpha=1, beta=2, rho=0.05, p_best=0.05)


def _one_run(args):
    name, cfg_dict = args
    inst = load_named(name, DATA)
    res = MMAS_SOP(inst, MMASConfig(**cfg_dict)).run()
    return dict(instance=inst.name, n=inst.n, seed=cfg_dict["seed"], best_cost=res.best_cost,
                iter_found=res.iter_found, time_total=round(res.time_total, 3),
                feasible=inst.is_feasible(res.best_tour), tour=" ".join(map(str, res.best_tour)))


def run_series(name, cfg, pool, label=""):
    jobs = [(name, asdict(replace(cfg, seed=s))) for s in range(N_RUNS)]
    df = pd.DataFrame(list(pool.map(_one_run, jobs)))
    df["label"] = label
    return df


def gap(cost, opt):
    return 100.0 * (cost - opt) / opt


def exp_main(pool):
    """Табл. 7: жадібний алгоритм, MMAS і MMAS + or-opt на кожній тестовій задачі."""
    runs, rows = [], []
    for name in REPORT_SET:
        inst = load_named(name, DATA)
        opt = optimum(inst)
        g_cost = greedy_nearest_feasible(inst)[1]
        row = dict(instance=name, n=inst.n, precedences=inst.n_precedences(), optimum=opt,
                   greedy_gap=round(gap(g_cost, opt), 2))
        for ls, key in ((False, "mmas"), (True, "mmas_ls")):
            df = run_series(name, replace(base_config(inst.n), local_search=ls), pool,
                            label="MMAS + or-opt" if ls else "MMAS")
            runs.append(df)
            c = df.best_cost
            row[f"{key}_best"] = c.min()
            row[f"{key}_mean"] = round(c.mean(), 2)
            row[f"{key}_gap"] = round(gap(c.mean(), opt), 2)
            row[f"{key}_hit"] = f"{int((c <= opt + 1e-9).sum())}/{N_RUNS}"
            row[f"{key}_time_s"] = round(df.time_total.mean(), 2)
        rows.append(row)
        print(row, flush=True)
    pd.concat(runs).to_csv(os.path.join(TABLES, "main_runs.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "main_results.csv"), index=False)


# Табл. 8: значення параметрів (решта — базові)
PARAM_GRID = {
    "alpha": [0.5, 1, 2, 3],
    "beta": [0, 1, 2, 3, 5],
    "rho": [0.01, 0.02, 0.05, 0.1, 0.3],
    "n_ants": [5, 10, 25, 50],
    "p_best": [None, 0.5, 0.05, 0.005],
    "update": ["ib", "gb", "schedule"],   # хто додає феромон: найкраща на ітерації / за весь час / чергування
}


def exp_params(pool, name="ESC25", iters=300):
    """Підбір параметрів: ESC25, 300 ітерацій, 10 запусків, без локального пошуку."""
    inst = load_named(name, DATA)
    base = MMASConfig(n_ants=25, n_iterations=iters, alpha=1, beta=2, rho=0.05, p_best=0.05,
                      update="schedule", local_search=False)
    rows, runs = [], []
    for param, values in PARAM_GRID.items():
        for v in values:
            df = run_series(name, replace(base, **{param: v}), pool, label=f"{param}={v}")
            df["param"], df["value"] = param, str(v)
            runs.append(df)
            c = df.best_cost
            rows.append(dict(param=param, value=str(v), best=c.min(), mean=round(c.mean(), 2),
                             gap=round(gap(c.mean(), inst.best_known), 2),
                             time_s=round(df.time_total.mean(), 2)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, f"params_{name}.csv"), index=False)
    pd.concat(runs).to_csv(os.path.join(TABLES, f"params_runs_{name}.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(TABLES, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    with ProcessPoolExecutor() as pool:
        if what in ("main", "all"):
            exp_main(pool)
        if what in ("params", "all"):
            exp_params(pool)
