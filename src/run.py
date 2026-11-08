"""
Запуск MMAS для однієї задачі з командного рядка.
  python -m src.run toy_6                    — один запуск з ходом ітерацій
  python -m src.run ESC25 --runs 10          — 10 запусків MMAS і MMAS + or-opt (протокол експерименту)
Задачі: toy_5, toy_6 (приклади 1 і 2) або назва задачі TSPLIB з data/ (br17.10, ESC11, ESC25, ry48p.1, ft53.1).
"""
import argparse
import os

import numpy as np

from .sop_instance import load_named, greedy_nearest_feasible, brute_force, exact_dp
from .mmas_sop import MMAS_SOP, MMASConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def optimum(inst):
    if inst.best_known is not None:
        return inst.best_known
    return brute_force(inst)[1] if inst.n <= 10 else exact_dp(inst)[1]


def main():
    ap = argparse.ArgumentParser(description="MMAS для Sequential Ordering Problem")
    ap.add_argument("instance", help="toy_5 | toy_6 | br17.10 | ESC11 | ESC25 | ry48p.1 | ft53.1")
    ap.add_argument("--ants", type=int, default=None, help="за замовчуванням min(n, 25)")
    ap.add_argument("--iters", type=int, default=None, help="за замовчуванням 200 (n<=20) або 500")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--pbest", type=float, default=0.05)
    ap.add_argument("--ls", action="store_true", help="локальний пошук or-opt (для одного запуску)")
    ap.add_argument("--runs", type=int, default=1, help="кількість запусків (>1 — режим експерименту)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    inst = load_named(a.instance, os.path.join(ROOT, "data"))
    opt = optimum(inst)
    ants = a.ants or min(inst.n, 25)
    iters = a.iters or (200 if inst.n <= 20 else 500)
    lab = lambda tour: " ".join(inst.labels[v] for v in tour)
    gap = lambda L: 100 * (L - opt) / opt
    print(f"Задача {inst.name}: n = {inst.n}, прецедентів = {inst.n_precedences()}, оптимум = {opt:.0f}")
    g_tour, g_cost = greedy_nearest_feasible(inst)
    print(f"Жадібний алгоритм: L = {g_cost:.0f} (gap {gap(g_cost):.2f} %)")
    base = dict(n_ants=ants, n_iterations=iters, alpha=a.alpha, beta=a.beta, rho=a.rho, p_best=a.pbest)
    print(f"Параметри: m = {ants}, ітерацій = {iters}, α = {a.alpha}, β = {a.beta}, ρ = {a.rho}, p_best = {a.pbest}")

    if a.runs == 1:
        res = MMAS_SOP(inst, MMASConfig(**base, local_search=a.ls, seed=a.seed)).run()
        step = max(1, iters // 10)
        for it in [1] + list(range(step, iters + 1, step)):
            print(f"  ітерація {it:4d}: best = {res.history_best[it - 1]:.0f}")
        print(f"Найкращий маршрут: {lab(res.best_tour)}")
        print(f"L = {res.best_cost:.0f}, gap = {gap(res.best_cost):.2f} %, знайдено на ітерації {res.iter_found}, "
              f"допустимий: {'так' if inst.is_feasible(res.best_tour) else 'НІ'}")
        return

    best_overall = None
    for ls, name in ((False, "MMAS"), (True, "MMAS + or-opt")):
        costs, tours = [], []
        for s in range(a.runs):
            r = MMAS_SOP(inst, MMASConfig(**base, local_search=ls, seed=s)).run()
            assert inst.is_feasible(r.best_tour)
            costs.append(r.best_cost); tours.append(r.best_tour)
        c = np.array(costs)
        hit = int((c <= opt + 1e-9).sum())
        print(f"{name:14s} ({a.runs} запусків): best = {c.min():.0f}, mean = {c.mean():.1f}, "
              f"gap = {gap(c.mean()):.2f} %, оптимум знайдено {hit}/{a.runs}")
        k = int(np.argmin(c))
        if best_overall is None or c[k] < best_overall[1]:
            best_overall = (tours[k], c[k])
    t, L = best_overall
    print(f"Найкращий маршрут: {lab(t)}")
    print(f"L = {L:.0f}, допустимий: {'так' if inst.is_feasible(t) else 'НІ'}")


if __name__ == "__main__":
    main()
