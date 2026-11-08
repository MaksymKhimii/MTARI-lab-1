"""
Локальний пошук для SOP: допустимий or-opt.

Хід: вирізати сегмент довжиною 1..3 (орієнтацію не змінюємо — це важливо,
бо вартості несиметричні) і вставити його в іншу позицію маршруту.
Допустимість перевіряється інкрементально, як у SOP-3-exchange (Gambardella & Dorigo):
  * перенесення сегмента S ВПЕРЕД через вершини X: жодна вершина з X не повинна мати
    попередника в S;
  * перенесення сегмента S НАЗАД через вершини X: жодна вершина з S не повинна мати
    попередника в X.
Щойно умова порушується, подальші позиції в цьому напрямку теж недопустимі — цикл
обривається (тому пошук швидкий).
Стратегія — first improvement, повторюємо, доки є покращення.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def or_opt(inst, tour: List[int], cost: float, max_seg: int = 3) -> Tuple[List[int], float]:
    c = inst.cost
    pm = inst.pred_mask
    tour = list(tour)
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for L in range(1, max_seg + 1):
            i = 1
            while i + L - 1 <= n - 2:
                seg = tour[i:i + L]
                a, b = tour[i - 1], tour[i + L]          # сусіди сегмента
                s0, s1 = seg[0], seg[-1]
                seg_mask = 0
                for v in seg:
                    seg_mask |= 1 << v
                seg_preds = 0
                for v in seg:
                    seg_preds |= pm[v]
                remove_gain = c[a, s0] + c[s1, b] - c[a, b]
                if not np.isfinite(remove_gain):
                    i += 1
                    continue
                best = None
                # --- вперед: вставити після tour[j], j >= i+L
                for j in range(i + L, n - 1):
                    x = tour[j]
                    if pm[x] & seg_mask:          # x має попередника в сегменті
                        break
                    y = tour[j + 1]
                    add = c[x, s0] + c[s1, y] - c[x, y]
                    if remove_gain - add > 1e-9:
                        best = ("f", j, remove_gain - add)
                        break
                # --- назад: вставити перед tour[j], j <= i-1
                if best is None:
                    for j in range(i - 1, 0, -1):
                        x = tour[j]
                        if seg_preds & (1 << x):  # сегмент має попередника серед X
                            break
                        w = tour[j - 1]
                        add = c[w, s0] + c[s1, x] - c[w, x]
                        if remove_gain - add > 1e-9:
                            best = ("b", j, remove_gain - add)
                            break
                if best is not None:
                    kind, j, gain = best
                    rest = tour[:i] + tour[i + L:]
                    if kind == "f":
                        pos = j - L + 1                 # індекс у rest після вилучення
                    else:
                        pos = j
                    tour = rest[:pos] + seg + rest[pos:]
                    cost -= gain
                    improved = True
                else:
                    i += 1
    return tour, float(inst.tour_cost(tour))
