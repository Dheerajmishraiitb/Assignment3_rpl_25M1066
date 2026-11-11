#!/usr/bin/env python3
"""Assignment 3 report figures (Tasks 1, 2, 4, 5 overlays; Tasks 6, 7 plots).

    python3 submission/plot_results.py            # from the Assignment_3 folder

Reads the raw result folders under results/ (or submission/results/) and
writes every figure into submission/plots/.

Exploration progress (%) is the known map AREA (known cells x resolution^2)
as a percentage of the known area of the manually built Task 1 map, so both
exploration methods are measured against the same reference and are directly
comparable even if their grid resolution differs from Task 1.
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "plots")


def results_dir(task):
    for base in (os.path.join(ROOT, "results"), os.path.join(HERE, "results")):
        d = os.path.join(base, task)
        if os.path.isdir(d) and os.listdir(d):
            return d
    return None


def load_map(d, stem):
    occ = np.loadtxt(os.path.join(d, f"{stem}.csv"), delimiter=",")
    meta = yaml.safe_load(open(os.path.join(d, f"{stem}.yaml")))
    return occ, float(meta["resolution"]), meta["origin"]


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}


def draw_map(ax, occ, res, origin):
    img = np.full(occ.shape + (3,), 0.80)            # unknown: grey
    img[occ == 0] = 1.0                              # free: white
    img[occ == 100] = 0.0                            # occupied: black
    h, w = occ.shape
    ax.imshow(img, origin="lower",
              extent=(origin[0], origin[0] + w * res, origin[1], origin[1] + h * res))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def overlay(task, stem, title, out_name, traj_name="trajectory.csv", checkpoints=False):
    d = results_dir(task)
    if d is None:
        print(f"skip {task}: no results")
        return
    occ, res, origin = load_map(d, stem)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    draw_map(ax, occ, res, origin)
    tp = os.path.join(d, traj_name)
    if os.path.exists(tp):
        t = read_csv(tp)
        x, y = t["x_m"].astype(float), t["y_m"].astype(float)
        ax.plot(x, y, "-", color="tab:blue", lw=1.6, label="robot trajectory")
        ax.plot(x[0], y[0], "o", color="tab:green", ms=9, label="start")
        ax.plot(x[-1], y[-1], "s", color="tab:red", ms=8, label="end")
    if checkpoints and os.path.exists(os.path.join(d, "checkpoints.csv")):
        c = read_csv(os.path.join(d, "checkpoints.csv"))
        for n, cx, cy in zip(c["name"], c["x_m"].astype(float), c["y_m"].astype(float)):
            ax.plot(cx, cy, "*", color="tab:orange", ms=16, mec="k")
            ax.annotate(n, (cx, cy), xytext=(6, 6), textcoords="offset points",
                        fontsize=10, fontweight="bold")
        ax.plot([], [], "*", color="tab:orange", ms=12, mec="k", label="Home / checkpoints")
    ax.plot([], [], "s", color="0.8", label="unknown"); ax.plot([], [], "s", color="k", label="occupied")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(title)
    fig.tight_layout()
    p = os.path.join(OUT, out_name)
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)


def reference_area():
    d = results_dir("task1")
    occ, res, _ = load_map(d, "manual_map")
    return float((occ != -1).sum()) * res * res


def coverage_plots(task, stem, prefix, label, ref_area):
    d = results_dir(task)
    if d is None:
        print(f"skip {task}: no results")
        return None
    _, res, _ = load_map(d, stem)
    c = read_csv(os.path.join(d, f"{stem}_coverage.csv"))
    t = c["time_s"].astype(float)
    prog = 100.0 * c["known_cells"].astype(float) * res * res / ref_area
    ent = c["map_entropy"].astype(float)
    for y, ylabel, name in ((prog, "Exploration progress (%)", "progress"),
                            (ent, "Map entropy", "entropy")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, y, color="tab:blue" if task == "task4" else "tab:purple", lw=1.8, label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f"{label}: {ylabel.lower()} vs time")
        fig.tight_layout()
        p = os.path.join(OUT, f"{prefix}_{name}_vs_time.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print("wrote", p)
    return t, prog, ent, c["travel_m"].astype(float)


def comparison(g, m):
    if g is None or m is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for (t, prog, ent, _), lab, col in ((g, "greedy (largest frontier)", "tab:blue"),
                                        (m, "MCTS", "tab:purple")):
        ax[0].plot(t, prog, color=col, label=lab)
        ax[1].plot(t, ent, color=col, label=lab)
    ax[0].set_ylabel("Exploration progress (%)"); ax[1].set_ylabel("Map entropy")
    for a in ax:
        a.set_xlabel("Time (s)"); a.grid(alpha=0.3); a.legend()
    fig.suptitle("Task 8: greedy vs MCTS on identical axes")
    fig.tight_layout()
    p = os.path.join(OUT, "task8_greedy_vs_mcts.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)


def summary(name, data):
    if data is None:
        return
    t, prog, ent, travel = data
    def t_at(p):
        idx = np.nonzero(prog >= p)[0]
        return f"{t[idx[0]]:.0f} s" if len(idx) else "never"
    print(f"{name}: duration {t[-1]:.0f} s, final progress {prog[-1]:.1f} %, "
          f"t(50%) {t_at(50)}, t(80%) {t_at(80)}, t(90%) {t_at(90)}, "
          f"travel {travel[-1]:.1f} m, entropy {ent[0]:.0f} -> {ent[-1]:.0f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    overlay("task1", "manual_map", "Task 1: occupancy-grid map with the manual mapping trajectory",
            "task1_trajectory_overlay.png")
    overlay("task2", None, "", "") if False else None
    d2 = results_dir("task2")
    if d2 and results_dir("task1"):
        # Task 2 runs on the Task 1 map
        occ, res, origin = load_map(results_dir("task1"), "manual_map")
        fig, ax = plt.subplots(figsize=(10, 6.5))
        draw_map(ax, occ, res, origin)
        t = read_csv(os.path.join(d2, "trajectory.csv"))
        ax.plot(t["x_m"].astype(float), t["y_m"].astype(float), "-", lw=1.6,
                color="tab:blue", label="executed trajectory")
        c = read_csv(os.path.join(d2, "checkpoints.csv"))
        for n, cx, cy in zip(c["name"], c["x_m"].astype(float), c["y_m"].astype(float)):
            ax.plot(cx, cy, "*", color="tab:orange", ms=16, mec="k")
            ax.annotate(n, (cx, cy), xytext=(6, 6), textcoords="offset points",
                        fontsize=10, fontweight="bold")
        ax.plot([], [], "*", color="tab:orange", ms=12, mec="k", label="Home / CP1-CP6")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Task 2: A* checkpoint mission HOME -> CP1 ... CP6 -> HOME on the Task 1 map")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "task2_final_path_overlay.png"), dpi=130); plt.close(fig)
        print("wrote task2_final_path_overlay.png")
    for task, stem, lab in (("task4", "exploration_greedy", "Task 4: greedy largest-frontier exploration"),
                            ("task5", "exploration_mcts", "Task 5: MCTS frontier exploration")):
        d = results_dir(task)
        if d:
            trajs = [os.path.basename(p) for p in glob.glob(os.path.join(d, "*traj*.csv"))]
            overlay(task, stem, f"{lab}: final map and trajectory", f"{task}_trajectory_overlay.png",
                    traj_name=trajs[0] if trajs else "trajectory.csv")
    if results_dir("task1"):
        ref = reference_area()
        g = coverage_plots("task4", "exploration_greedy", "task6_greedy", "Greedy", ref)
        m = coverage_plots("task5", "exploration_mcts", "task7_mcts", "MCTS", ref)
        comparison(g, m)
        summary("greedy", g)
        summary("mcts", m)
