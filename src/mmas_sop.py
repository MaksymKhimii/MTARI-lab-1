"""
Max–Min Ant System (MMAS) для Sequential Ordering Problem.

Модель відповідності SOP -> ACO:
  * компонент рішення — перехід i->j (дуга орієнтованого графа);
  * феромон tau[i, j] — на дузі i->j, матриця НЕСИМЕТРИЧНА;
  * евристика eta[i, j] = 1 / (c[i, j] + delta);
  * мураха стартує з 0; допустимі кандидати на кожному кроці — невідвідані вершини,
    у яких усі попередники вже відвідані (вершина n-1 автоматично стає останньою);
  * правило вибору p_ij ∝ tau_ij^alpha * eta_ij^beta (рулетка);
  * якість — довжина відкритого маршруту 0 -> ... -> n-1.

Особливості MMAS (Stützle & Hoos; Dorigo & Stützle 2004, розд. 3.3.4):
  * феромон відкладає лише одна мураха: iteration-best або global-best;
  * tau обмежено [tau_min, tau_max]; tau_max = 1/(rho * L_gb),
    tau_min = tau_max * (1 - p_best^(1/n)) / ((avg - 1) * p_best^(1/n)), avg = n/2;
  * ініціалізація tau = tau_max; перезапуск феромону при стагнації.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Callable, List, Optional

import numpy as np

from .sop_instance import SOPInstance, greedy_nearest_feasible
from .local_search import or_opt


@dataclass
class MMASConfig:
    n_ants: int = 20
    n_iterations: int = 300
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.05
    p_best: Optional[float] = 0.05      # None => без меж (≈ elitist Ant System з одним мурахою)
    update: str = "schedule"            # 'ib' | 'gb' | 'schedule' (ib, але кожну gb_every-ту ітерацію — gb)
    gb_every: int = 5
    delta: float = 1.0                  # зсув у eta = 1/(c + delta), бо в SOP бувають нульові вартості
    local_search: bool = False          # or-opt над iteration-best
    restart_after: Optional[int] = 100  # ітерацій без покращення до перезапуску феромону
    seed: Optional[int] = 0

    def short(self) -> str:
        return (f"m={self.n_ants}, α={self.alpha}, β={self.beta}, ρ={self.rho}, "
                f"p_best={self.p_best}, upd={self.update}, LS={self.local_search}")


@dataclass
class MMASResult:
    best_tour: List[int]
    best_cost: float
    iter_found: int
    time_found: float
    time_total: float
    history_best: List[float] = field(default_factory=list)
    history_iter_best: List[float] = field(default_factory=list)
    history_mean: List[float] = field(default_factory=list)
    tau_history: List[np.ndarray] = field(default_factory=list)
    tau_iters: List[int] = field(default_factory=list)
    tau_best_tours: List[List[int]] = field(default_factory=list)
    n_restarts: int = 0


class MMAS_SOP:
    def __init__(self, inst: SOPInstance, cfg: MMASConfig):
        self.inst = inst
        self.cfg = cfg
        self.n = inst.n
        self.rng = np.random.default_rng(cfg.seed)
        c = inst.cost.copy()
        finite = np.isfinite(c)
        self.allowed = finite                       # дуги, якими можна йти
        self.eta = np.zeros_like(c)
        self.eta[finite] = 1.0 / (c[finite] + cfg.delta)
        self.eta_beta = self.eta ** cfg.beta
        # прямі попередники як лічильники (для швидкого визначення кандидатів)
        self.n_preds = np.array([len(inst.preds[v]) for v in range(self.n)], dtype=int)
        self.succ_lists = [np.array(sorted(inst.succs[v]), dtype=int) for v in range(self.n)]
        self.tau = None
        self.tau_min = 0.0
        self.tau_max = np.inf

    # ------------------------------------------------------------------
    def _bounds(self, L_gb: float):
        cfg = self.cfg
        tau_max = 1.0 / (cfg.rho * L_gb) if L_gb > 0 else 1.0
        if cfg.p_best is None:
            return 1e-12, np.inf, tau_max
        pr = cfg.p_best ** (1.0 / self.n)
        avg = self.n / 2.0
        tau_min = tau_max * (1 - pr) / ((avg - 1) * pr)
        tau_min = min(tau_min, tau_max)
        return tau_min, tau_max, tau_max

    def construct(self, record_steps: bool = False):
        """Побудова одного маршруту мурахою. Якщо record_steps — повертає також кроки
        (поточна вершина, кандидати, ймовірності, обрана вершина) для візуалізації."""
        n = self.n
        visited = np.zeros(n, dtype=bool)
        remaining = self.n_preds.copy()
        tour = [0]
        visited[0] = True
        for s in self.succ_lists[0]:
            remaining[s] -= 1
        cur = 0
        steps = []
        weights_tau = self.tau ** self.cfg.alpha if self.cfg.alpha != 1 else self.tau
        for _ in range(n - 1):
            cand_mask = (~visited) & (remaining == 0) & self.allowed[cur]
            cand = np.flatnonzero(cand_mask)
            if cand.size == 0:   # немає прямої дуги — йдемо в будь-яку допустиму вершину
                cand = np.flatnonzero((~visited) & (remaining == 0))
                w = np.ones(cand.size)
            else:
                w = weights_tau[cur, cand] * self.eta_beta[cur, cand]
            tot = w.sum()
            if tot <= 0 or not np.isfinite(tot):
                p = np.full(cand.size, 1.0 / cand.size)
            else:
                p = w / tot
            j = int(cand[self.rng.choice(cand.size, p=p)]) if cand.size > 1 else int(cand[0])
            if record_steps:
                steps.append((cur, cand.copy(), p.copy(), j))
            tour.append(j)
            visited[j] = True
            for s in self.succ_lists[j]:
                remaining[s] -= 1
            cur = j
        if record_steps:
            return tour, steps
        return tour

    def _deposit(self, tour: List[int], L: float):
        idx_from = np.array(tour[:-1])
        idx_to = np.array(tour[1:])
        self.tau[idx_from, idx_to] += 1.0 / L if L > 0 else 1.0

    # ------------------------------------------------------------------
    def run(self, record_every: Optional[int] = None,
            callback: Optional[Callable[[int, "MMAS_SOP"], None]] = None,
            time_limit: Optional[float] = None) -> MMASResult:
        cfg, inst = self.cfg, self.inst
        t0 = time.perf_counter()
        g_tour, g_cost = greedy_nearest_feasible(inst)
        self.tau_min, self.tau_max, tau0 = self._bounds(g_cost)
        self.tau = np.full((self.n, self.n), tau0 if np.isfinite(tau0) else 1.0)
        best_tour, best_cost = None, np.inf
        iter_found, time_found = 0, 0.0
        res = MMASResult([], np.inf, 0, 0.0, 0.0)
        no_improve = 0

        for it in range(1, cfg.n_iterations + 1):
            tours = [self.construct() for _ in range(cfg.n_ants)]
            costs = np.array([inst.tour_cost(t) for t in tours])
            k = int(np.argmin(costs))
            ib_tour, ib_cost = tours[k], float(costs[k])
            if cfg.local_search:
                ib_tour, ib_cost = or_opt(inst, ib_tour, ib_cost)

            if ib_cost < best_cost - 1e-9:
                best_tour, best_cost = list(ib_tour), ib_cost
                iter_found, time_found = it, time.perf_counter() - t0
                self.tau_min, self.tau_max, _ = self._bounds(best_cost)
                no_improve = 0
            else:
                no_improve += 1

            # випаровування
            self.tau *= (1.0 - cfg.rho)
            # депозит однієї мурахи
            use_gb = cfg.update == "gb" or (cfg.update == "schedule" and it % cfg.gb_every == 0)
            if use_gb:
                self._deposit(best_tour, best_cost)
            else:
                self._deposit(ib_tour, ib_cost)
            # межі MMAS
            np.clip(self.tau, self.tau_min, self.tau_max, out=self.tau)

            if cfg.restart_after and no_improve >= cfg.restart_after:
                self.tau.fill(self.tau_max if np.isfinite(self.tau_max) else 1.0)
                no_improve = 0
                res.n_restarts += 1

            res.history_best.append(best_cost)
            res.history_iter_best.append(ib_cost)
            res.history_mean.append(float(costs.mean()))
            if record_every and (it == 1 or it % record_every == 0 or it == cfg.n_iterations):
                res.tau_history.append(self.tau.copy())
                res.tau_iters.append(it)
                res.tau_best_tours.append(list(best_tour))
            if callback:
                callback(it, self)
            if time_limit and time.perf_counter() - t0 > time_limit:
                break

        res.best_tour, res.best_cost = best_tour, best_cost
        res.iter_found, res.time_found = iter_found, time_found
        res.time_total = time.perf_counter() - t0
        return res


def solve(inst: SOPInstance, **kwargs) -> MMASResult:
    return MMAS_SOP(inst, MMASConfig(**kwargs)).run()
