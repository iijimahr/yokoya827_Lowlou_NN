"""磁場の評価: 厳密解との比較と、中央差分による div B・J×B の確認。

コマンドとしても使える（npz の成分・座標の解釈と格子幅の比 dz/dx の確認）:
    python -m lowlou.metrics [file.npz]
"""
import argparse
import itertools

import numpy as np


def central_diff(f, axis, h=1.0, order=2):
    """中央差分 ∂f/∂x_axis（2 次または 4 次精度）。結果は各軸の両端（2 次: 1 点、4 次: 2 点）を除いた内部点だけ。"""
    f = np.moveaxis(f, axis, 0)
    if order == 2:
        d, m = (f[2:] - f[:-2]) / (2.0 * h), 1
    else:
        d, m = (-f[4:] + 8 * f[3:-1] - 8 * f[1:-3] + f[:-4]) / (12.0 * h), 2
    d = np.moveaxis(d, 0, axis)
    sl = [slice(m, -m)] * 3
    sl[axis] = slice(None)
    return d[tuple(sl)]


def fd_metrics(B, h=(1.0, 1.0, 1.0), order=2):
    """中央差分による div B と J×B の指標。

    div_rel = Σ|∇·B| / Σ(|∂xBx| + |∂yBy| + |∂zBz|)   （0 なら div B = 0）
    sigma_J = Σ(|J×B|/|B|) / Σ|J|                     （0 ならフォースフリー。Wheatland et al. 2000）
    Low & Lou の厳密解でも、2 次精度・格子幅 1 では打ち切り誤差で div_rel ≈ 0.012、sigma_J ≈ 0.025 になる。
    """
    m = 1 if order == 2 else 2
    D = [[central_diff(B[..., c], k, h[k], order) for c in range(3)] for k in range(3)]   # D[k][c] = ∂B_c/∂x_k
    Bi = B[m:-m, m:-m, m:-m]
    div = D[0][0] + D[1][1] + D[2][2]
    J = np.stack([D[1][2] - D[2][1], D[2][0] - D[0][2], D[0][1] - D[1][0]], -1)
    absB, absJ = np.linalg.norm(Bi, axis=-1), np.linalg.norm(J, axis=-1)
    return {
        "div_rel": float(np.abs(div).sum() / (np.abs(D[0][0]) + np.abs(D[1][1]) + np.abs(D[2][2])).sum()),
        "sigma_J": float((np.linalg.norm(np.cross(J, Bi), axis=-1) / absB).sum() / absJ.sum()),
    }


def _schrijver(p, q):
    dp = np.linalg.norm(p - q, axis=-1)
    nq, np_ = np.linalg.norm(q, axis=-1), np.linalg.norm(p, axis=-1)
    return {
        "C_vec": float((p * q).sum() / np.sqrt((p**2).sum() * (q**2).sum())),
        "C_CS": float(np.mean((p * q).sum(-1) / (nq * np_ + 1e-30))),
        "E_n'": float(1 - dp.sum() / nq.sum()),
        "E_m'": float(1 - np.mean(dp / nq)),
        "energy": float((p**2).sum() / (q**2).sum()),
    }


def compare(B, b, dx=1.0, dy=1.0, dz=1.0, margin=8):
    """予測 B と厳密解 b（どちらも shape (nx, ny, nz, 3)、物理単位）の比較。

    C_vec, C_CS, E_n' = 1 - E_n, E_m' = 1 - E_m, energy: Schrijver et al. (2006) の指標（いずれも 1 が完全一致）
    *_center:   側面付近を除いた中央部（margin <= ix, iy < n - margin）での同じ指標
    E_n_bottom: 下端 iz = 0 での正規化誤差（0 が完全一致）
    div_rel, sigma_J: 予測 B を中央差分で評価した値（fd_metrics を参照）
    """
    res = _schrijver(B, b)
    m = margin
    res.update({k + "_center": v for k, v in _schrijver(B[m:-m, m:-m], b[m:-m, m:-m]).items()})
    res["E_n_bottom"] = float(np.linalg.norm(B[:, :, 0] - b[:, :, 0], axis=-1).sum()
                              / np.linalg.norm(b[:, :, 0], axis=-1).sum())
    res.update(fd_metrics(B, (dx, dy, dz)))
    return res


def check_axes(b):
    """配列の軸と (x, y, z) の対応（並べ替え 6 通り × 向き 8 通り）ごとの div_rel と sigma_J。

    正しい解釈 b[ix, iy, iz] = (Bx, By, Bz) のときだけ両方が小さくなるはず。div_rel の小さい順に返す。
    """
    rows = []
    for axes in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            # 仮定: 物理座標 x_k = 配列軸 axes[k]（向き signs[k]）、b[..., k] = B_k。
            # 配列をその並びと向きに直してから評価する（成分の符号はそのまま）
            bb = np.transpose(b, (*axes, 3))
            for k, s in enumerate(signs):
                if s < 0:
                    bb = np.flip(bb, axis=k)
            rows.append((axes, signs, fd_metrics(bb)))
    return sorted(rows, key=lambda r: r[2]["div_rel"])


def estimate_dz_ratio(b, order=4):
    """div B を最小にする dz/dx を最小二乗で求める（全体と高さごと）。

    s = dx/dz とすると div B = (∂xBx + ∂yBy) + s ∂zBz（微分は格子単位）。s について線形なので
    Σ(div B)² を最小にする s は s = -Σ(A C) / Σ(C²)。
    """
    A = central_diff(b[..., 0], 0, order=order) + central_diff(b[..., 1], 1, order=order)
    C = central_diff(b[..., 2], 2, order=order)
    s = -(A * C).sum() / (C * C).sum()
    sz = -(A * C).sum(axis=(0, 1)) / (C * C).sum(axis=(0, 1))
    return float(1 / s), 1 / sz


def main():
    from .io import DEFAULT_DATA, load_field
    ap = argparse.ArgumentParser(description="npz の磁場の成分・座標の解釈と div B, J×B を中央差分で確認する")
    ap.add_argument("file", nargs="?", default=DEFAULT_DATA)
    b = load_field(ap.parse_args().file)
    print("shape:", b.shape)
    print("|B| の水平平均: 下端 iz=0 →", np.linalg.norm(b[:, :, 0], axis=-1).mean().round(3),
          " 上端 iz=-1 →", np.linalg.norm(b[:, :, -1], axis=-1).mean().round(3))

    print("\n[1] 軸の対応ごとの評価（div_rel が小さい順に上位 6 件。全体の向きを反転したものは同じ値になる）")
    print(f"{'x,y,z の配列軸':>14} {'向き':>12} {'div_rel':>9} {'sigma_J':>9}")
    for axes, signs, m in check_axes(b)[:6]:
        print(f"{str(axes):>14} {str(signs):>12} {m['div_rel']:9.4f} {m['sigma_J']:9.4f}")

    print("\n[2] 標準の解釈 b[ix, iy, iz] = (Bx, By, Bz) での値（厳密解なら 2 次: 0.012 / 0.025 程度）")
    for order in (2, 4):
        m = fd_metrics(b, order=order)
        print(f"  {order} 次精度: div_rel = {m['div_rel']:.5f}, sigma_J = {m['sigma_J']:.5f}")

    r, rz = estimate_dz_ratio(b)
    print(f"\n[3] div B を最小にする dz/dx（4 次精度、最小二乗）: 全体 {r:.4f}")
    print("  高さごと: " + "  ".join(f"iz={k}:{rz[k - 2]:.3f}" for k in (2, 4, 16, 32, b.shape[2] - 3)))


if __name__ == "__main__":
    main()
