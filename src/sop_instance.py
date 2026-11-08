"""
Sequential Ordering Problem (SOP): опис задачі, завантаження даних, перевірка допустимості,
точні розв'язувачі для малих задач (повний перебір і динамічне програмування).

Постановка:
    Орієнтований граф G=(V,A), V={0..n-1}; 0 — старт, n-1 — фініш.
    c[i][j] — вартість дуги i->j (несиметрична).
    Прецеденти: пара (a,b) означає, що a треба відвідати раніше за b.
    Шукаємо гамільтонів шлях 0 -> ... -> n-1 мінімальної вартості,
    що задовольняє всі прецеденти (маршрут відкритий, без повернення в 0).

Формат TSPLIB SOP:
    після EDGE_WEIGHT_SECTION іде число n, далі матриця n x n;
    c[i][j] = -1 означає, що j має бути відвіданий РАНІШЕ за i.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

INF = math.inf


@dataclass
class SOPInstance:
    name: str
    n: int
    cost: np.ndarray                 # (n, n) float; INF — дуги немає / заборонена
    preds: List[set]                 # preds[v] — прямі попередники v (без 0 і без v)
    labels: List[str] = field(default_factory=list)
    coords: Optional[np.ndarray] = None   # (n, 2) для візуалізації, якщо є
    best_known: Optional[float] = None

    def __post_init__(self):
        if not self.labels:
            self.labels = [str(i) for i in range(self.n)]
        # Фініш n-1 має йти після всіх вершин, старт 0 — перед усіма.
        last = self.n - 1
        self.preds[last] = set(range(self.n - 1))
        for v in range(1, self.n):
            self.preds[v].add(0)
        self.preds[0] = set()
        for v in range(self.n):
            self.preds[v].discard(v)
        self.closure = transitive_closure(self.preds, self.n)
        # Бітові маски (Python int підтримує довільну довжину).
        self.pred_mask = [sum(1 << u for u in self.closure[v]) for v in range(self.n)]
        self.succs = [set() for _ in range(self.n)]
        for v in range(self.n):
            for u in self.preds[v]:
                self.succs[u].add(v)

    # --- властивості ---
    @property
    def start(self) -> int:
        return 0

    @property
    def end(self) -> int:
        return self.n - 1

    def precedence_pairs(self, include_trivial: bool = False) -> List[Tuple[int, int]]:
        """Список пар (a, b): a раніше за b (прямі прецеденти з даних)."""
        pairs = []
        for b in range(self.n):
            for a in self.preds[b]:
                if not include_trivial and (a == 0 or b == self.n - 1):
                    continue
                pairs.append((a, b))
        return sorted(pairs)

    def n_precedences(self) -> int:
        return len(self.precedence_pairs())

    # --- оцінка розв'язку ---
    def tour_cost(self, tour: Sequence[int]) -> float:
        return float(sum(self.cost[tour[k], tour[k + 1]] for k in range(len(tour) - 1)))

    def is_feasible(self, tour: Sequence[int]) -> bool:
        return check_feasible(self, tour)[0]

    def describe(self) -> str:
        return (f"{self.name}: n={self.n}, прецедентів={self.n_precedences()}, "
                f"best_known={self.best_known}")


def transitive_closure(preds: List[set], n: int) -> List[set]:
    """Повна множина всіх (прямих і непрямих) попередників кожної вершини."""
    closure: List[Optional[set]] = [None] * n
    visiting = set()

    def dfs(v):
        if closure[v] is not None:
            return closure[v]
        if v in visiting:
            raise ValueError("Цикл у прецедентах — задача не має допустимих розв'язків")
        visiting.add(v)
        res = set(preds[v])
        for u in preds[v]:
            res |= dfs(u)
        visiting.discard(v)
        closure[v] = res
        return res

    for v in range(n):
        dfs(v)
    return closure  # type: ignore


def check_feasible(inst: SOPInstance, tour: Sequence[int]) -> Tuple[bool, str]:
    """Повертає (True, 'ok') або (False, причина)."""
    n = inst.n
    if len(tour) != n:
        return False, f"довжина маршруту {len(tour)} != {n}"
    if sorted(tour) != list(range(n)):
        return False, "маршрут не є перестановкою всіх вершин"
    if tour[0] != 0 or tour[-1] != n - 1:
        return False, "маршрут має починатися в 0 і закінчуватися в n-1"
    pos = {v: k for k, v in enumerate(tour)}
    for b in range(n):
        for a in inst.preds[b]:
            if pos[a] > pos[b]:
                return False, f"порушено прецедент {inst.labels[a]} -> {inst.labels[b]}"
    for k in range(n - 1):
        if not math.isfinite(inst.cost[tour[k], tour[k + 1]]):
            return False, f"дуги {tour[k]}->{tour[k+1]} не існує"
    return True, "ok"


# ---------------------------------------------------------------------------
# Завантаження даних
# ---------------------------------------------------------------------------
def load_tsplib_sop(path: str, best_known: Optional[float] = None) -> SOPInstance:
    with open(path) as f:
        text = f.read()
    header, _, body = text.partition("EDGE_WEIGHT_SECTION")
    name = None
    for line in header.splitlines():
        if line.strip().upper().startswith("NAME"):
            name = line.split(":", 1)[1].strip().replace(".sop", "")
    tokens = body.replace("EOF", " ").split()
    n = int(tokens[0])
    vals = np.array([float(t) for t in tokens[1:1 + n * n]]).reshape(n, n)
    cost = vals.copy()
    preds: List[set] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if vals[i, j] == -1:
                preds[i].add(j)          # j має передувати i
                cost[i, j] = INF         # дуга i->j неможлива у допустимому маршруті
    np.fill_diagonal(cost, INF)
    return SOPInstance(name=name or path, n=n, cost=cost, preds=preds, best_known=best_known)


def from_dict(name: str, nodes: Sequence[str], dist: Dict[str, Dict[str, float]],
              precedence: Iterable[Tuple[str, str]], start: str, end: str,
              coords: Optional[Dict[str, Tuple[float, float]]] = None,
              best_known: Optional[float] = None) -> SOPInstance:
    """Задача у форматі методички: dist={A:{B:5,...}}, precedence=[('A','B'),...].
    Вершини перенумеровуються так, щоб start=0, end=n-1. Відсутні дуги = INF."""
    order = [start] + [v for v in nodes if v not in (start, end)] + [end]
    idx = {v: k for k, v in enumerate(order)}
    n = len(order)
    cost = np.full((n, n), INF)
    for a, row in dist.items():
        for b, c in row.items():
            cost[idx[a], idx[b]] = float(c)
    preds: List[set] = [set() for _ in range(n)]
    for a, b in precedence:
        preds[idx[b]].add(idx[a])
    xy = None
    if coords:
        xy = np.array([coords[v] for v in order], dtype=float)
    return SOPInstance(name=name, n=n, cost=cost, preds=preds, labels=list(order),
                       coords=xy, best_known=best_known)


# ---------------------------------------------------------------------------
# Власні приклади
# ---------------------------------------------------------------------------
TOY5_COST = {
    "S": {"A": 3, "B": 2, "C": 4},
    "A": {"B": 4, "C": 5, "T": 2},
    "B": {"A": 2, "C": 3, "T": 5},
    "C": {"A": 4, "B": 1, "T": 3},
}
TOY5_COORDS = {"S": (0, 0), "A": (2, 3), "B": (3, -2), "C": (6, 2), "T": (8, 0)}


def toy_5() -> SOPInstance:
    """Приклад 1 зі звіту: 5 вершин, прецедент A -> C.
    Без обмеження найдешевшим був би S-C-B-A-T (L=9), але він порушує A -> C;
    оптимум SOP: S-B-A-C-T, L=12."""
    return from_dict("toy_5", ["S", "A", "B", "C", "T"], TOY5_COST, [("A", "C")],
                     start="S", end="T", coords=TOY5_COORDS)


TOY6_COST = {
    "S": {"A": 1, "B": 2, "C": 2, "D": 6},
    "A": {"B": 5, "C": 6, "D": 4, "T": 1},
    "B": {"A": 3, "C": 7, "D": 7, "T": 9},
    "C": {"A": 6, "B": 9, "D": 8, "T": 9},
    "D": {"A": 5, "B": 1, "C": 1, "T": 6},
}
TOY6_COORDS = {"S": (0, 0), "A": (2, 3), "B": (6, 3), "C": (2, -3), "D": (6, -3), "T": (8, 0)}


def toy_6() -> SOPInstance:
    """Приклад 2 зі звіту: 6 вершин, прецеденти A -> B, C -> D.
    24 перестановки, допустимих 6; жадібний вибір дає S-A-B-C-D-T (L=27),
    оптимум SOP: S-C-A-D-B-T, L=22."""
    return from_dict("toy_6", ["S", "A", "B", "C", "D", "T"], TOY6_COST, [("A", "B"), ("C", "D")],
                     start="S", end="T", coords=TOY6_COORDS)


# ---------------------------------------------------------------------------
# Точні методи (для перевірки ACO на малих задачах)
# ---------------------------------------------------------------------------
def brute_force(inst: SOPInstance, return_count: bool = False):
    """Повний перебір усіх перестановок внутрішніх вершин (n <= 11).
    Повертає (оптимальний маршрут, вартість[, кількість допустимих маршрутів])."""
    n = inst.n
    if n > 12:
        raise ValueError("Повний перебір лише для n <= 12")
    best, best_tour, feasible = INF, None, 0
    for perm in itertools.permutations(range(1, n - 1)):
        tour = (0,) + perm + (n - 1,)
        ok, _ = check_feasible(inst, tour)
        if not ok:
            continue
        feasible += 1
        c = inst.tour_cost(tour)
        if c < best:
            best, best_tour = c, list(tour)
    if return_count:
        return best_tour, best, feasible
    return best_tour, best


def exact_dp(inst: SOPInstance, max_states: int = 3_000_000):
    """Динамічне програмування Беллмана–Гельда–Карпа з урахуванням прецедентів.
    Стан = (множина відвіданих вершин, остання вершина). Розглядаються лише
    множини, замкнені щодо прецедентів, тож для SOP станів значно менше, ніж 2^n * n.
    Повертає (оптимальний маршрут, вартість)."""
    n = inst.n
    pm = inst.pred_mask
    cost = inst.cost
    layer = {(1, 0): 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    total = 0
    for _ in range(n - 1):
        nxt: Dict[Tuple[int, int], float] = {}
        for (mask, last), val in layer.items():
            for v in range(1, n):
                bit = 1 << v
                if mask & bit or (pm[v] & ~mask):
                    continue
                c = cost[last, v]
                if not math.isfinite(c):
                    continue
                key = (mask | bit, v)
                nv = val + c
                if nv < nxt.get(key, INF):
                    nxt[key] = nv
                    parent[key] = (mask, last)
        total += len(nxt)
        if total > max_states:
            raise MemoryError("Забагато станів для точного DP")
        layer = nxt
    full = (1 << n) - 1
    key = (full, n - 1)
    best = layer[key]
    tour = []
    while key in parent:
        tour.append(key[1])
        key = parent[key]
    tour.append(0)
    return tour[::-1], best


def greedy_nearest_feasible(inst: SOPInstance) -> Tuple[List[int], float]:
    """Базовий жадібний алгоритм: щоразу йдемо в найдешевшу допустиму вершину."""
    n = inst.n
    visited = 1
    tour = [0]
    cur = 0
    for _ in range(n - 1):
        best_v, best_c = None, INF
        for v in range(1, n):
            if visited >> v & 1 or (inst.pred_mask[v] & ~visited):
                continue
            if inst.cost[cur, v] < best_c:
                best_v, best_c = v, inst.cost[cur, v]
        if best_v is None:  # трапляється лише якщо немає дуги; беремо будь-яку допустиму
            best_v = next(v for v in range(1, n)
                          if not visited >> v & 1 and not (inst.pred_mask[v] & ~visited))
        tour.append(best_v)
        visited |= 1 << best_v
        cur = best_v
    return tour, inst.tour_cost(tour)


# ---------------------------------------------------------------------------
# Набір тестових задач (як у звіті, табл. 6)
# ---------------------------------------------------------------------------
# Оптимальні значення з бібліотеки TSPLIB (для ESC11 і br17.10 підтверджено exact_dp).
BEST_KNOWN = {"ESC11": 2075, "br17.10": 55, "ESC25": 1681, "ry48p.1": 15805, "ft53.1": 7531}
REPORT_SET = ["toy_5", "toy_6", "br17.10", "ESC11", "ESC25", "ry48p.1", "ft53.1"]


def load_named(name: str, data_dir: str = "data") -> SOPInstance:
    """toy_5, toy_6 — приклади 1 і 2 зі звіту; інші назви — файли TSPLIB з data/."""
    import os
    if name == "toy_5":
        return toy_5()
    if name == "toy_6":
        return toy_6()
    return load_tsplib_sop(os.path.join(data_dir, f"{name}.sop"), BEST_KNOWN.get(name))
