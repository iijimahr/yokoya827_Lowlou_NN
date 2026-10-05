"""PINN による非線形フォースフリー磁場の再構成（ベースライン）

下端（z = 0）の磁場 B = (Bx, By, Bz) だけを境界条件として与え、フォースフリー条件 J×B = 0 と
div B = 0 を満たす 3 次元磁場を PINN で求める。Low & Lou の解析解を厳密解として評価する。

  - 座標は物理座標（格子幅 DX, DY, DZ）。長さは水平方向の領域の大きさ L0、磁場は下端の |B| の RMS B0 で割って
    無次元化し、ネットワークは無次元量 (x, y, z) → B を学習する
  - 損失（どの項も B について 2 次）:
        フォースフリー  |J×B|² / (|B| + ε)²   （分母の |B| には勾配を流さない）
        div B           (∇·B)²
        下端境界        |B − B_obs|²
  - 内部の点は、水平方向は一様、鉛直方向は密度 ∝ (1 − z/Lz)²（下端と上端の密度比 10:1）
  - ネットワークは全結合 5 層 × 128、活性化関数 tanh、出力は B の 3 成分

実行: このディレクトリの run.sh（学習して図を描く）。出力はすべてこのディレクトリに書かれる。
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from lowlou.io import DEFAULT_DATA, load_field, save_field
from lowlou.metrics import compare

OUT = os.path.dirname(os.path.abspath(__file__))   # 出力先（このディレクトリ）

# ---------------------------------------------------------------- 設定
DATA = DEFAULT_DATA
STEPS = 10000                        # 学習のステップ数（--steps で変えられる）
EVAL_EVERY = 2500                    # このステップごとに評価してモデルを保存する
DX, DY, DZ = 1.0, 1.0, 1.0           # 格子幅
N_PDE = 24576                        # 内部の点の数（毎ステップ取り直す）
N_BC = 2024                          # 下端の点の数（毎ステップ取り直す）
W_FF, W_DIV, W_BC = 1.0, 1.0, 10.0   # 損失の重み
Z_RATIO = 10.0                       # 内部の点の密度比（下端 : 上端）
EPS_PHYS = 0.1                       # フォースフリー項の分母に足す値（物理単位の B）
LR_MAX, LR_MIN = 1e-3, 1e-4          # 学習率の最大値と最小値（tanh 型で下げる。lr_at を参照）
LR_T_HALF = 5000                     # 学習率が最大値と最小値の中間になるステップ
LR_T_WIDTH = 3183                    # 学習率が変化するステップの幅
# LR_T_HALF と LR_T_WIDTH は、10000 ステップの CosineAnnealing と中間点の位置・傾きが同じになるように決めた
# （LR_T_HALF = 10000 / 2、LR_T_WIDTH = 10000 / π）
WIDTH, DEPTH = 128, 5                # ネットワークの幅と層数
SEED = 1234
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- ネットワーク
class MLP(nn.Module):
    """全結合ネットワーク（tanh）。入力は xyz 共通の値 s で [0, s] → [-1, 1] に正規化する。"""

    def __init__(self, Lx, Ly, Lz, width=WIDTH, depth=DEPTH):
        super().__init__()
        self.s = max(Lx, Ly, Lz)
        dims = [3] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(width, 3)

    def forward(self, x, y, z):
        h = 2.0 * torch.cat([x, y, z], dim=1) / self.s - 1.0
        for lin in self.hidden:
            h = torch.tanh(lin(h))
        return self.out(h)


def field(model, x, y, z):
    """ネットワークの出力をそのまま B とする。"""
    return model(x, y, z)


def grad(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]


def curl_div(B, x, y, z):
    """B の回転 J = ∇×B と発散 ∇·B。"""
    d = [[grad(B[:, c:c + 1], v) for v in (x, y, z)] for c in range(3)]  # d[c][k] = ∂B_c/∂x_k
    J = torch.cat([d[2][1] - d[1][2], d[0][2] - d[2][0], d[1][0] - d[0][1]], dim=-1)
    return J, d[0][0] + d[1][1] + d[2][2]


class Bottom:
    """下端 z = 0 の B（無次元）を、無次元座標 (x, y) で双線形補間する。"""

    def __init__(self, b0, dx, dy, B0, device):
        self.g = torch.as_tensor(b0 / B0, dtype=torch.float32, device=device)  # (nx, ny, 3)
        self.dx, self.dy = dx, dy                                             # 無次元の格子幅

    def __call__(self, x, y):
        nx, ny = self.g.shape[:2]
        gx = (x / self.dx).clamp(0, nx - 1).squeeze(-1)
        gy = (y / self.dy).clamp(0, ny - 1).squeeze(-1)
        i0 = gx.floor().long().clamp(max=nx - 2)
        j0 = gy.floor().long().clamp(max=ny - 2)
        tx = (gx - i0)[:, None]
        ty = (gy - j0)[:, None]
        g = self.g
        return ((1 - tx) * (1 - ty) * g[i0, j0] + tx * (1 - ty) * g[i0 + 1, j0]
                + (1 - tx) * ty * g[i0, j0 + 1] + tx * ty * g[i0 + 1, j0 + 1])


def predict_grid(model, shape, h):
    """格子点上の B（無次元）を shape (nx, ny, nz, 3) で返す。h は無次元の格子幅。"""
    nx, ny, nz = shape
    X, Y, Z = torch.meshgrid(torch.arange(nx, device=DEVICE) * h[0], torch.arange(ny, device=DEVICE) * h[1],
                             torch.arange(nz, device=DEVICE) * h[2], indexing="ij")
    pts = [t.reshape(-1, 1).float() for t in (X, Y, Z)]
    out = []
    with torch.no_grad():
        for c in zip(*(p.split(32768) for p in pts)):
            out.append(field(model, *c))
    return torch.cat(out).reshape(nx, ny, nz, 3).cpu().numpy().astype(np.float64)


def lr_at(step):
    """学習率（tanh 型）。ステップ数だけで決まり、総ステップ数には依存しない。"""
    return LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1.0 - math.tanh((step - LR_T_HALF) / LR_T_WIDTH))


# ---------------------------------------------------------------- 学習
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    steps = ap.parse_args().steps

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # データと無次元化
    b = load_field(DATA)                                  # (nx, ny, nz, 3)
    nx, ny, nz = b.shape[:3]
    b0 = b[:, :, 0]                                       # 下端
    L0 = (nx - 1) * DX                                    # 長さ: 水平方向の領域の大きさ
    B0 = float(np.sqrt(np.mean(np.sum(b0**2, axis=-1))))  # 磁場: 下端の |B| の RMS
    Lx, Ly, Lz = (nx - 1) * DX / L0, (ny - 1) * DY / L0, (nz - 1) * DZ / L0   # 無次元の領域
    h = (DX / L0, DY / L0, DZ / L0)                                         # 無次元の格子幅
    eps = EPS_PHYS / B0
    print(f"device = {DEVICE}, L0 = {L0}, B0 = {B0:.3f}")

    bottom = Bottom(b0, h[0], h[1], B0, DEVICE)
    model = MLP(Lx, Ly, Lz).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_at(step) / LR_MAX)

    # 鉛直方向の点の分布: 密度 ∝ (1 - a ζ)²（ζ = z/Lz）、下端:上端 = Z_RATIO:1。逆関数法で生成する
    a = 1.0 - 1.0 / np.sqrt(Z_RATIO)
    c = 1.0 - (1.0 - a) ** 3

    def rand(n, L):
        return (0.0 + 1.0 * torch.rand(n, 1, device=DEVICE)) * L

    def rand_z(n):
        u = torch.rand(n, 1, device=DEVICE)
        return (1.0 - (1.0 - c * u) ** (1.0 / 3.0)) / a * Lz

    def evaluate(step):
        B = predict_grid(model, (nx, ny, nz), h) * B0   # 物理単位に戻す
        r = {"step": step, **compare(B, b, DX, DY, DZ)}
        print(json.dumps(r), file=evals, flush=True)
        en = r["E_n'"]
        print(f"  評価 step {step}: C_vec {r['C_vec']:.4f}, E_n' {en:.3f}, sigma_J {r['sigma_J']:.3f}")
        return B, r

    log = open(os.path.join(OUT, "log.jsonl"), "w")
    evals = open(os.path.join(OUT, "evals.jsonl"), "w")
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()

        # 内部の点: フォースフリーと div B
        x = rand(N_PDE, Lx).requires_grad_(True)
        y = rand(N_PDE, Ly).requires_grad_(True)
        z = rand_z(N_PDE).requires_grad_(True)
        B = field(model, x, y, z)
        J, div = curl_div(B, x, y, z)
        Bm = B.detach().norm(dim=-1) + eps
        loss_ff = (torch.sum(torch.cross(J, B, dim=-1) ** 2, dim=-1) * Bm ** -2.0).mean()
        loss_div = (div[:, 0] ** 2).mean()

        # 下端の点: 境界条件
        xb, yb = rand(N_BC, Lx), rand(N_BC, Ly)
        Bb = field(model, xb, yb, torch.zeros_like(xb))
        loss_bc = torch.sum((Bb - bottom(xb, yb)) ** 2, dim=-1).mean()

        loss = W_FF * loss_ff + W_DIV * loss_div + W_BC * loss_bc
        loss.backward()
        opt.step()
        sched.step()

        if step % 100 == 0 or step == steps - 1:
            rec = {"step": step, "loss": loss.item(), "ff": loss_ff.item(), "div": loss_div.item(),
                   "bc": loss_bc.item(), "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
            print(json.dumps(rec), file=log, flush=True)
            if step % 1000 == 0:
                print(f"step {step:6d}  loss {rec['loss']:.4g}  ({time.time() - t0:.0f} s)")
        if (step + 1) % EVAL_EVERY == 0 and step + 1 < steps:
            evaluate(step + 1)
            torch.save(model.state_dict(), os.path.join(OUT, f"model_{step + 1}.pt"))

    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    B, r = evaluate(steps)
    save_field(os.path.join(OUT, "B_pred.npz"), B)
    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump({**r, "time_s": time.time() - t0}, f, indent=1)


if __name__ == "__main__":
    main()
