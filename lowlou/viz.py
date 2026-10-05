"""可視化: 磁場の断面と、学習の経過。

コマンドとして使う:
    python -m lowlou.viz slice B_pred.npz --ref exact.npz --z 0 -o slice.png        # 水平断面
    python -m lowlou.viz slice B_pred.npz --ref exact.npz --y 32 --symlog -o s.png  # 鉛直断面（対数的な色）
    python -m lowlou.viz loss run_dir1 run_dir2 -o loss.png                          # 学習の経過を重ねて描く
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .io import load_field, read_jsonl  # noqa: E402


def plot_slice(b, ref=None, z=0, y=None, symlog=False):
    """Bx, By, Bz の断面。ref を渡すと 3 段（b、ref、差）で描く。

    y を指定すると鉛直断面（x-z 面）、しなければ高さ z の水平断面（x-y 面）。
    symlog=True にすると対数的な色スケールになり、上空の弱い磁場も見える。
    """
    def cut(f):
        return f[:, :, z] if y is None else f[:, y, :]
    rows = [(cut(b), "PINN")]
    if ref is not None:
        rows += [(cut(ref), "exact"), (cut(b) - cut(ref), "PINN - exact")]
    where = f"z = {z}" if y is None else f"y = {y}"

    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.8 * len(rows)), squeeze=False, constrained_layout=True)
    for c, name in enumerate(["Bx", "By", "Bz"]):
        vmax = np.abs(rows[-2 if ref is not None else 0][0][..., c]).max()   # 色の範囲は厳密解に合わせる
        if symlog:
            norm = matplotlib.colors.SymLogNorm(linthresh=1.0, vmin=-vmax, vmax=vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
        for r, (img, label) in enumerate(rows):
            im = axes[r, c].imshow(img[..., c].T, origin="lower", cmap="RdBu_r", norm=norm)
            axes[r, c].set_title(f"{name}  {label}  ({where})")
            axes[r, c].set_xlabel("x")
            axes[r, c].set_ylabel("y" if y is None else "z")
            fig.colorbar(im, ax=axes[r, c], shrink=0.85)
    return fig


def plot_loss(run_dirs):
    """各実験ディレクトリの log.jsonl（損失）と evals.jsonl（評価値）を重ねて描く。"""
    top = [("loss", "total loss"), ("ff", "force-free loss"), ("div", "div B loss"), ("bc", "bottom BC loss")]
    bottom = [("E_n'", "E_n'  (1 = exact)"), ("C_CS", "C_CS  (1 = exact)"), ("sigma_J", "sigma_J  (0 = force-free)"),
              ("div_rel", "div_rel  (0 = div-free)")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for i, d in enumerate(run_dirs):
        label = os.path.basename(os.path.abspath(d))
        log = read_jsonl(os.path.join(d, "log.jsonl"))
        ev = read_jsonl(os.path.join(d, "evals.jsonl"))
        for ax, (k, title) in zip(axes[0], top):
            ax.plot([r["step"] for r in log], [r[k] for r in log], color=f"C{i}", label=label)
            ax.set_title(title)
            ax.set_yscale("log")
        for ax, (k, title) in zip(axes[1], bottom):
            ax.plot([r["step"] for r in ev], [r[k] for r in ev], "o-", color=f"C{i}", label=label)
            ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slice")
    s.add_argument("file")
    s.add_argument("--ref", help="比べる npz（厳密解）")
    s.add_argument("--z", type=int, default=0)
    s.add_argument("--y", type=int)
    s.add_argument("--symlog", action="store_true")
    s.add_argument("-o", "--out", default="slice.png")
    l = sub.add_parser("loss")
    l.add_argument("runs", nargs="+")
    l.add_argument("-o", "--out", default="loss.png")
    args = ap.parse_args()

    if args.cmd == "slice":
        ref = load_field(args.ref) if args.ref else None
        fig = plot_slice(load_field(args.file), ref, args.z, args.y, args.symlog)
    else:
        fig = plot_loss(args.runs)
    fig.savefig(args.out, dpi=100)
    print("saved:", args.out)
