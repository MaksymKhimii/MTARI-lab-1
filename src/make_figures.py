"""
Рисунки зі звіту:
  python -m src.make_figures screenshots   — рис. 1–7: запуск програми для кожної тестової задачі
  python -m src.make_figures viz           — рис. 8–9 і GIF-анімації для прикладу 2 (toy_6)
  python -m src.make_figures all
"""
from __future__ import annotations

import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sop_instance import toy_6, REPORT_SET
from .mmas_sop import MMAS_SOP, MMASConfig
from . import visualize as V

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
GIF = os.path.join(ROOT, "results", "gifs")
CONSOLE = os.path.join(ROOT, "results", "console")


def terminal_png(command: str, text: str, path: str, width: int = 100):
    """Зберігає вивід програми як зображення термінала."""
    out = []
    for line in [command] + text.rstrip().splitlines():
        while len(line) > width:
            cut = line.rfind(" ", 0, width)
            out.append(line[:cut])
            line = "   " + line[cut + 1:]
        out.append(line)
    h = 0.2 * len(out) + 0.3
    fig = plt.figure(figsize=(9.5, h))
    fig.patch.set_facecolor("#1e1e1e")
    for k, line in enumerate(out):
        fig.text(0.012, 1 - (k + 0.75) * 0.2 / h, line, family="DejaVu Sans Mono", fontsize=9,
                 color="#7ee787" if line.startswith("$") else "#e6e6e6", va="top")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def screenshots():
    """Рис. 1–7: python -m src.run <задача> --runs 10."""
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(CONSOLE, exist_ok=True)
    for name in REPORT_SET:
        cmd = [sys.executable, "-m", "src.run", name, "--runs", "10"]
        text = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
        with open(os.path.join(CONSOLE, f"{name}.txt"), "w") as f:
            f.write(text)
        terminal_png(f"$ python -m src.run {name} --runs 10", text, os.path.join(FIG, f"run_{name}.png"))
        print("screenshot:", name)


def viz():
    """Рис. 8 (перший крок мурахи) і рис. 9 (зміна феромону) для прикладу 2, плюс GIF."""
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(GIF, exist_ok=True)
    inst = toy_6()
    # Рис. 8: побудова маршруту однією мурахою при однаковому феромоні (перша ітерація)
    ant = MMAS_SOP(inst, MMASConfig(seed=1))
    ant.tau = np.ones((inst.n, inst.n))
    V.construction_animation(inst, ant, os.path.join(GIF, "toy6_construction.gif"))
    # Рис. 9: зміна феромону під час роботи MMAS
    res = MMAS_SOP(inst, MMASConfig(n_ants=6, n_iterations=40, rho=0.1, seed=0)).run(record_every=2)
    V.pheromone_sequence(inst, res, os.path.join(FIG, "toy6_pheromone_sequence.png"),
                         picks=[0, 1, 4, len(res.tau_history) - 1])
    V.animate_pheromone(inst, res, os.path.join(GIF, "toy6_pheromone.gif"))
    print("toy_6: L =", res.best_cost, "знайдено на ітерації", res.iter_found)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("screenshots", "all"):
        screenshots()
    if what in ("viz", "all"):
        viz()
