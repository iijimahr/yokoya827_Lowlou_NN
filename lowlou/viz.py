"""可視化: 磁場の断面と、学習の経過（損失・評価値）。

コマンドとしても使える:
    python -m lowlou.viz slice [file.npz] [--ref exact.npz] [--z 0 | --y 32] [--symlog 1] [-o out.png]
    python -m lowlou.viz loss  run_dir [run_dir ...] [-o loss.png]

磁場は下端付近の約 1000 から上空の約 1 まで 3 桁変わるので、上空の弱い磁場を見るときは --symlog を付ける
（|B| < LINTHRESH の範囲は線形、それより外は対数の色スケール）。
"""
import argparse
import os

import numpy as np


def _plt(show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return matplotlib, plt


def take_slice(b, z=0, y=None):
    """水平断面 b[:, :, z]（横軸 x, 縦軸 y）か、鉛直断面 b[:, y, :]（横軸 x, 縦軸 z）。"""
    return (b[:, :, z], f"z = {z}", "y") if y is None else (b[:, y, :], f"y = {y}", "z")


def plot_slice(b, ref=None, z=0, y=None, symlog=None, labels=("B", "ref"), show=False):
    """Bx, By, Bz の断面を描く。ref を渡すと 3 段（b, ref, b − ref）で並べ、その断面の C_vec と E_n を表示する。"""
    matplotlib, plt = _plt(show)
    img, where, yl = take_slice(b, z, y)
    rows = [(img, labels[0])]
    if ref is not None:
        rimg = take_slice(ref, z, y)[0]
        rows += [(rimg, labels[1]), (img - rimg, "difference")]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.8 * len(rows)), squeeze=False, constrained_layout=True)
    for c, name in enumerate(["Bx", "By", "Bz"]):
        vmax = np.abs(rows[1][0][..., c] if ref is not None else img[..., c]).max()   # 色の範囲は ref に合わせる
        for r, (data, label) in enumerate(rows):
            norm = (matplotlib.colors.SymLogNorm(linthresh=symlog, vmin=-vmax, vmax=vmax) if symlog
                    else matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax))      # 0 を白にそろえる
            im = axes[r, c].imshow(data[..., c].T, origin="lower", cmap="RdBu_r", norm=norm)
            axes[r, c].set_title(f"{name}  {label}  ({where})")
            axes[r, c].set_xlabel("x")
            axes[r, c].set_ylabel(yl)
            fig.colorbar(im, ax=axes[r, c], shrink=0.85)
    if ref is not None:
        p, q = rows[0][0], rows[1][0]
        cvec = (p * q).sum() / np.sqrt((p**2).sum() * (q**2).sum())
        en = np.linalg.norm(p - q, axis=-1).sum() / np.linalg.norm(q, axis=-1).sum()
        fig.suptitle(f"this slice: C_vec = {cvec:.4f},  E_n = {en:.3f}")
    return fig


LOSS_TERMS = [("ff", "force-free  |J×B|²/|B|²"), ("div", "div B  (∇·B)²"), ("bc", "bottom BC  |B−B_obs|²"),
              ("lr", "learning rate")]
EVAL_TERMS = [("E_n'", "E_n'  (1 = exact)"), ("C_CS", "C_CS  (1 = exact)"), ("sigma_J", "σ_J  (0 = force-free)"),
              ("div_rel", "div_rel")]
EXACT_FD = {"sigma_J": 0.025, "div_rel": 0.012}   # 厳密解を 2 次精度の中央差分で評価した値（破線で表示）


def plot_loss(run_dirs, show=False):
    """複数のランの log.jsonl（損失）と evals.jsonl（評価値）を重ねて描く。"""
    from .io import read_jsonl
    _, plt = _plt(show)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for i, d in enumerate(run_dirs):
        color, label = f"C{i}", os.path.basename(os.path.normpath(d))
        log, ev = read_jsonl(os.path.join(d, "log.jsonl")), read_jsonl(os.path.join(d, "evals.jsonl"))
        for ax, (k, title) in zip(axes[0], LOSS_TERMS):
            vals = [r.get(k, 0.0) for r in log]
            if any(v > 0 for v in vals):          # 0 だけの項（vp の div など）は描かない
                ax.plot([r["step"] for r in log], vals, color=color, lw=1.5, label=label)
            ax.set_title(title)
        for ax, (k, title) in zip(axes[1], EVAL_TERMS):
            if ev:
                ax.plot([r["step"] for r in ev], [r[k] for r in ev], "o-", color=color, lw=2, ms=5, label=label)
            ax.set_title(title)
    for ax in axes[0]:
        ax.set_yscale("log")
    for ax, (k, _) in zip(axes[1], EVAL_TERMS):
        if k in EXACT_FD:
            ax.axhline(EXACT_FD[k], color="gray", ls="--", lw=1)
    for ax in axes.flat:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    if axes[1, 0].lines:
        axes[1, 0].legend()
    return fig


def main():
    from .io import DEFAULT_DATA, load_field
    ap = argparse.ArgumentParser(description="磁場の断面や学習の経過を描く")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slice", help="npz の磁場を断面で表示する")
    s.add_argument("file", nargs="?", default=DEFAULT_DATA)
    s.add_argument("--ref", default=None, help="比べる npz（厳密解など）。指定すると 3 段（file, ref, 差）で描く")
    s.add_argument("--z", type=int, default=0, help="水平断面の高さ（格子番号）")
    s.add_argument("--y", type=int, default=None, help="指定すると、この y での鉛直断面（x-z 面）を描く")
    s.add_argument("--symlog", type=float, default=None, metavar="LINTHRESH", help="対数的な色スケール（例: 1）")
    l = sub.add_parser("loss", help="学習の経過（損失と評価値）を描く")
    l.add_argument("runs", nargs="+", help="実験ディレクトリ（log.jsonl, evals.jsonl があるもの）")
    for p in (s, l):
        p.add_argument("-o", "--out", default=None, help="保存するファイル名")
        p.add_argument("--show", action="store_true", help="画面にも表示する")
    args = ap.parse_args()

    if args.cmd == "slice":
        ref = load_field(args.ref) if args.ref else None
        fig = plot_slice(load_field(args.file), ref, args.z, args.y, args.symlog, show=args.show,
                         labels=(os.path.basename(args.file), os.path.basename(args.ref or "")))
        where = f"y{args.y}" if args.y is not None else f"z{args.z}"
        out = args.out or f"slice_{where}{'_symlog' if args.symlog else ''}.png"
    else:
        fig = plot_loss(args.runs, show=args.show)
        out = args.out or "loss.png"
    fig.savefig(out, dpi=110)
    print("saved:", out)
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
