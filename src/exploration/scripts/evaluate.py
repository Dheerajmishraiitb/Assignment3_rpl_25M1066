#!/usr/bin/env python3
import argparse
import os

import numpy as np

import ekf_localization as EKF


PKG = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def dead_reckoning(dataset):
    o, gt = np.asarray(dataset["odom"]), np.asarray(dataset["gt"])
    th0 = gt[0, 2] - o[0, 2]
    c, s = np.cos(th0), np.sin(th0)
    R = np.array([[c, -s], [s, c]])
    xy = (R @ (o[:, :2] - o[0, :2]).T).T + gt[0, :2]
    th = EKF.wrapToPi(o[:, 2] + th0)
    return np.column_stack([xy, th])


def errors(est, gt):
    n = min(len(est), len(gt))
    pos = np.hypot(est[:n, 0] - gt[:n, 0], est[:n, 1] - gt[:n, 1])
    head = np.abs(np.degrees(EKF.wrapToPi(est[:n, 2] - gt[:n, 2])))
    return pos, head


def summarise(name, pos, head, extra=""):
    print(f"  {name:<16} position  mean {pos.mean():7.4f}  p95 {np.percentile(pos,95):7.3f}  "
          f"max {pos.max():7.3f} m")
    print(f"  {'':<16} heading   mean {head.mean():7.2f}  p95 {np.percentile(head,95):7.2f}  "
          f"max {head.max():7.2f} deg"
          + (f"   {extra}" if extra else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=EKF.DEFAULT_DATASET)
    ap.add_argument("--gate", default=str(EKF.INNOVATION_GATE),
                    help="innovation gate, or 'none' to accept every observation")
    ap.add_argument("--out", default=os.path.join(PKG, "docs",
                                                  "ekf_vs_groundtruth.png"))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--csv", nargs="?", const=os.path.join(PKG, "docs"),
                    default=None, metavar="DIR",
                    help="Task 6: write TWO trajectory CSVs into DIR -- "
                         "ground_truth.csv and ekf.csv, one row per filter step. "
                         "Diagnostics do NOT go here: record the /diag/* topics to a "
                         "ros2 bag instead (see the README).")
    args = ap.parse_args()

    gate = None if args.gate.lower() == "none" else float(args.gate)
    d = EKF.load_dataset(args.dataset)
    gt = np.asarray(d["gt"])
    M = np.asarray(d["M"])

    _, _, est = EKF.run_ekf_localization(args.dataset, plot=False, gate=gate)
    dr = dead_reckoning(d)

    e_pos, e_head = errors(est, gt)
    d_pos, d_head = errors(dr, gt)
    jumps = int((np.abs(np.diff(e_pos)) > 0.5).sum())
    seen = sum(1 for z in d["z"] if np.size(z) > 0)
    dist = float(np.sum(np.hypot(np.diff(gt[:, 0]), np.diff(gt[:, 1]))))

    print(f"\n{os.path.basename(args.dataset)}: {len(gt)} filter steps, {dist:.2f} m driven, "
          f"{100*seen/len(gt):.1f} % of steps carry an observation")
    print(f"gate = {gate}, alpha = {EKF.ALPHA}\n")
    summarise("EKF", e_pos, e_head, f"jumps > 0.5 m: {jumps}")
    summarise("dead reckoning", d_pos, d_head)
    print(f"\n  improvement over dead reckoning: "
          f"{d_pos.mean()/max(e_pos.mean(),1e-12):.0f}x in position, "
          f"{d_head.mean()/max(e_head.mean(),1e-12):.0f}x in heading")

    if args.csv:
  
        import csv as _csv
        os.makedirs(args.csv, exist_ok=True)
        step = np.arange(len(gt))
        travelled = np.concatenate(([0.0], np.cumsum(
            np.hypot(np.diff(gt[:, 0]), np.diff(gt[:, 1])))))
        est_a = np.asarray(est)
        for name, traj, err_p, err_h in (
                ("ground_truth.csv", gt, None, None),
                ("ekf.csv", est_a, e_pos, e_head)):
            path = os.path.join(args.csv, name)
            with open(path, "w", newline="") as f:
                w = _csv.writer(f)
                head = ["step", "distance_m", "x", "y", "yaw_deg"]
                if err_p is not None:
      
                    head += ["err_pos_m", "err_yaw_deg"]
                w.writerow(head)
                for k in range(len(gt)):
                    row = [step[k], travelled[k], traj[k, 0], traj[k, 1],
                           np.degrees(EKF.wrapToPi(traj[k, 2]))]
                    if err_p is not None:
                        row += [err_p[k], err_h[k]]
                    w.writerow([f"{x:.6g}" for x in row])
            print(f"\nwrote {path}  ({len(gt)} rows)")

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], hspace=0.32, wspace=0.2)
    ax = fig.add_subplot(gs[:, 0])

    ax.plot(dr[:, 0], dr[:, 1], color="#c0392b", lw=1.2, ls="--",
            label=f"dead reckoning (odometry only), mean {d_pos.mean():.2f} m")
    ax.plot(gt[:, 0], gt[:, 1], color="#27ae60", lw=2.6, label="ground truth", zorder=3)
    ax.plot(est[:, 0], est[:, 1], color="#2c6fbb", lw=1.5,
            label=f"EKF estimate, mean {e_pos.mean():.3f} m", zorder=4)
    ax.scatter(M[:, 0], M[:, 1], marker="s", s=52, facecolor="#f4f4f4",
               edgecolor="#555", linewidth=1.0, zorder=5, label="ArUco markers")
    for i, (mx, my) in enumerate(M):
        ax.annotate(str(i), (mx, my), fontsize=7, ha="center", va="center",
                    zorder=6, color="#333")
    ax.plot(gt[0, 0], gt[0, 1], "o", ms=11, mfc="none", mec="black", mew=1.8,
            zorder=7, label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"EKF localization vs ground truth - {dist:.1f} m of teleoperation")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.93)
    ax.grid(alpha=0.25, lw=0.5)

    travelled = np.concatenate([[0], np.cumsum(
        np.hypot(np.diff(gt[:, 0]), np.diff(gt[:, 1])))])[:len(e_pos)]

    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(travelled, d_pos, color="#c0392b", lw=1.0, ls="--", label="dead reckoning")
    a1.plot(travelled, e_pos, color="#2c6fbb", lw=1.3, label="EKF")
    a1.set_yscale("log")
    a1.set_ylabel("position error [m]")
    a1.set_title("Error against distance travelled (log scale)", fontsize=10)
    a1.legend(fontsize=8.5); a1.grid(alpha=0.25, lw=0.5, which="both")

    a2 = fig.add_subplot(gs[1, 1])
    a2.plot(travelled, d_head, color="#c0392b", lw=1.0, ls="--", label="dead reckoning")
    a2.plot(travelled, e_head, color="#2c6fbb", lw=1.3, label="EKF")
    a2.set_yscale("log")
    a2.set_xlabel("distance travelled [m]"); a2.set_ylabel("heading error [deg]")
    a2.legend(fontsize=8.5); a2.grid(alpha=0.25, lw=0.5, which="both")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=135, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
