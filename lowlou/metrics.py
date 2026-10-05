"""磁場の評価: 厳密解との比較と、中央差分による div B・J×B の確認。"""
import numpy as np


def central_diff(f, axis, h):
    """2 次精度の中央差分 ∂f/∂x_axis。結果は内部点（各軸の両端 1 点を除く）だけ。"""
    f = np.moveaxis(f, axis, 0)
    d = np.moveaxis((f[2:] - f[:-2]) / (2.0 * h), 0, axis)
    sl = [slice(1, -1)] * 3
    sl[axis] = slice(None)
    return d[tuple(sl)]


def compare(B, b, dx=1.0, dy=1.0, dz=1.0):
    """予測 B と厳密解 b（どちらも shape (nx, ny, nz, 3)）を比べる。

    C_vec, C_CS, E_n', E_m', energy: Schrijver et al. (2006) の指標。いずれも 1 が完全一致
    E_n_bottom: 下端 iz = 0 での誤差。0 が完全一致
    div_rel:    div B の大きさ。0 なら div B = 0（厳密解でも差分の誤差で 0.012 になる）
    sigma_J:    電流と磁場のなす角。0 ならフォースフリー（厳密解でも差分の誤差で 0.025 になる）
    """
    err = np.linalg.norm(B - b, axis=-1)
    nB, nb = np.linalg.norm(B, axis=-1), np.linalg.norm(b, axis=-1)
    res = {
        "C_vec": (B * b).sum() / np.sqrt((B**2).sum() * (b**2).sum()),
        "C_CS": np.mean((B * b).sum(-1) / (nB * nb)),
        "E_n'": 1 - err.sum() / nb.sum(),
        "E_m'": 1 - np.mean(err / nb),
        "energy": (B**2).sum() / (b**2).sum(),
        "E_n_bottom": err[:, :, 0].sum() / nb[:, :, 0].sum(),
    }

    # 予測 B の div B と J = ∇×B を中央差分で求める
    h = (dx, dy, dz)
    D = [[central_diff(B[..., c], k, h[k]) for c in range(3)] for k in range(3)]   # D[k][c] = ∂B_c/∂x_k
    div = D[0][0] + D[1][1] + D[2][2]
    J = np.stack([D[1][2] - D[2][1], D[2][0] - D[0][2], D[0][1] - D[1][0]], -1)
    Bi = B[1:-1, 1:-1, 1:-1]
    res["div_rel"] = np.abs(div).sum() / (np.abs(D[0][0]) + np.abs(D[1][1]) + np.abs(D[2][2])).sum()
    res["sigma_J"] = (np.linalg.norm(np.cross(J, Bi), axis=-1) / np.linalg.norm(Bi, axis=-1)).sum() \
        / np.linalg.norm(J, axis=-1).sum()
    return {k: float(v) for k, v in res.items()}
