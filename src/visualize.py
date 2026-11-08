"""
Візуалізація роботи MMAS для SOP (розділ «Візуалізація» звіту):
  * покрокова побудова маршруту однією мурахою: допустимі вершини та ймовірності вибору (рис. 8);
  * карта феромону на графі: товщина/непрозорість стрілки ∝ tau_ij, червоним — найкращий маршрут,
    пунктиром — прецеденти; послідовність знімків (рис. 9) і GIF-анімація.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
import numpy as np

C_TOUR = "#d1495b"
C_PREC = "#2e86ab"
C_EDGE = "#333333"
C_NODE = "#f4f1de"
C_CAND = "#edae49"


def layout(inst) -> np.ndarray:
    if inst.coords is not None:
        return np.asarray(inst.coords, dtype=float)
    n = inst.n
    ang = np.linspace(np.pi, -np.pi, n, endpoint=False)
    return np.c_[np.cos(ang), np.sin(ang)] * 10


def _arrow(ax, p, q, lw=1.0, alpha=1.0, color=C_EDGE, rad=0.12, ls="-", z=1, shrink=9):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7 + 2 * lw, lw=lw, alpha=alpha,
                        color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls,
                        shrinkA=shrink, shrinkB=shrink, zorder=z)
    ax.add_patch(a)


def _nodes(ax, inst, xy, highlight=None, visited=None):
    n = inst.n
    for v in range(n):
        fc = C_NODE
        if v in (0, n - 1):
            fc = "#81b29a"
        if visited is not None and v in visited:
            fc = "#bbbbbb"
        if highlight is not None and v in highlight:
            fc = C_CAND
        ax.scatter(*xy[v], s=330 if n <= 20 else 160, c=fc, edgecolors="black", zorder=5)
        ax.text(*xy[v], inst.labels[v], ha="center", va="center", fontsize=8 if n <= 20 else 6, zorder=6)


def plot_pheromone(inst, tau: np.ndarray, best_tour: Optional[Sequence[int]] = None, ax=None,
                   title: str = "", show_prec: bool = True, top_k: Optional[int] = None,
                   norm=None):
    xy = layout(inst)
    n = inst.n
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    t = tau.copy().astype(float)
    np.fill_diagonal(t, 0)
    t[~np.isfinite(inst.cost)] = 0
    lo, hi = norm if norm else (t[t > 0].min() if (t > 0).any() else 0, t.max())
    den = max(hi - lo, 1e-12)
    edges = [(i, j) for i in range(n) for j in range(n) if t[i, j] > 0]
    if top_k is None:
        top_k = len(edges) if n <= 12 else 4 * n
    edges.sort(key=lambda e: -t[e])
    for i, j in edges[:top_k]:
        w = (t[i, j] - lo) / den
        if w < 0.03:
            continue
        _arrow(ax, xy[i], xy[j], lw=0.3 + 3.2 * w, alpha=0.08 + 0.75 * w, z=1)
    if best_tour is not None:
        for a, b in zip(best_tour[:-1], best_tour[1:]):
            _arrow(ax, xy[a], xy[b], lw=1.6, color=C_TOUR, alpha=0.95, rad=0.12, z=3)
    if show_prec:
        for a, b in inst.precedence_pairs():
            _arrow(ax, xy[a], xy[b], lw=1.0, color=C_PREC, alpha=0.8, rad=-0.25, ls=(0, (3, 2)), z=2)
    _nodes(ax, inst, xy)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    pad = (xy.max(0) - xy.min(0)).max() * 0.08 + 0.5
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    return ax


def legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], color=C_EDGE, lw=3, alpha=0.6, label="феромон τ (товщина ∝ τ)"),
            Line2D([0], [0], color=C_TOUR, lw=2, label="найкращий маршрут"),
            Line2D([0], [0], color=C_PREC, lw=1.5, ls="--", label="прецедент a → b")]


def pheromone_sequence(inst, res, path, picks=None, top_k=None):
    k = len(res.tau_history)
    picks = picks or sorted(set([0, k // 4, k // 2, k - 1]))
    fig, axes = plt.subplots(1, len(picks), figsize=(4.4 * len(picks), 4.6))
    for ax, p in zip(np.atleast_1d(axes), picks):
        L = inst.tour_cost(res.tau_best_tours[p])
        plot_pheromone(inst, res.tau_history[p], res.tau_best_tours[p], ax=ax, top_k=top_k,
                       title=f"ітерація {res.tau_iters[p]}, best = {L:.0f}")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{inst.name}: еволюція феромону (MMAS)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate_pheromone(inst, res, path, fps=4, top_k=None):
    fig, ax = plt.subplots(figsize=(7, 6.4))

    def draw(k):
        ax.clear()
        L = inst.tour_cost(res.tau_best_tours[k])
        gap = ""
        if inst.best_known:
            gap = f", gap = {100 * (L - inst.best_known) / inst.best_known:.2f}%"
        plot_pheromone(inst, res.tau_history[k], res.tau_best_tours[k], ax=ax, top_k=top_k,
                       title=f"{inst.name} — ітерація {res.tau_iters[k]}: best = {L:.0f}{gap}")
        ax.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.08),
                  ncol=3, fontsize=8, frameon=False)
        return []

    ani = animation.FuncAnimation(fig, draw, frames=len(res.tau_history), blit=False)
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def construction_animation(inst, aco, path, fps=1):
    """Покрокова побудова маршруту однією мурахою: кандидати + їхні ймовірності."""
    tour, steps = aco.construct(record_steps=True)
    xy = layout(inst)
    fig, ax = plt.subplots(figsize=(7, 6.2))
    frames = len(steps) + 1

    def draw(k):
        ax.clear()
        part = tour[:k + 1]
        for a, b in zip(part[:-1], part[1:]):
            _arrow(ax, xy[a], xy[b], lw=2, color=C_TOUR, z=3)
        for a, b in inst.precedence_pairs():
            _arrow(ax, xy[a], xy[b], lw=1, color=C_PREC, alpha=0.7, rad=-0.25, ls=(0, (3, 2)), z=2)
        if k < len(steps):
            cur, cand, p, j = steps[k]
            for v, pv in zip(cand, p):
                _arrow(ax, xy[cur], xy[v], lw=0.6 + 4 * pv, color=C_CAND, alpha=0.9, rad=0.05, z=2)
                mid = (xy[cur] + xy[v]) / 2
                ax.text(*mid, f"{pv:.2f}", fontsize=8, color="#7a4f00", zorder=7,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
            blocked = [v for v in range(inst.n) if v not in part and v not in cand]
            txt = (f"Крок {k + 1}: мураха у вершині {inst.labels[cur]}. "
                   f"Допустимі: {', '.join(inst.labels[v] for v in cand)}"
                   + (f"; заблоковані прецедентами: {', '.join(inst.labels[v] for v in blocked)}" if blocked else "")
                   + f"\nОбрано: {inst.labels[j]}")
            _nodes(ax, inst, xy, highlight=set(cand.tolist()), visited=set(part))
        else:
            txt = f"Маршрут: {' → '.join(inst.labels[v] for v in tour)},  L = {inst.tour_cost(tour):.0f}"
            _nodes(ax, inst, xy, visited=set(part))
        ax.set_title(txt, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
        return []

    ani = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    ani.save(path, writer=animation.PillowWriter(fps=fps))
    # також окремі кадри PNG для звіту
    base = os.path.splitext(path)[0]
    for k in (0, min(2, frames - 1), frames - 1):
        draw(k)
        fig.savefig(f"{base}_step{k + 1}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return tour, steps
