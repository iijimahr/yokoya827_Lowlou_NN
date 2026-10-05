"""PINN による非線形フォースフリー磁場の再構成（ベクトルポテンシャル + SIREN 版）

下端（z = 0）の磁場ベクトル B = (Bx, By, Bz) だけを境界条件として与え、
フォースフリー条件 J×B = 0 と div B = 0 を満たす 3 次元磁場を PINN で求める。
テスト問題として Low & Lou の解析解（data/b_0.210_0.124.npz）を使い、厳密解と比べて評価する。

ベースライン版（pinn_baseline.py）との違いは次の 2 点だけ:
  A. ベクトルポテンシャル: ネットワークの出力を A とし、B = ∇×A とする。div B = 0 が式の上で厳密に成り立つので
     div B の損失は使わない。代わりに J = ∇×∇×A に 2 階微分が必要になり、計算時間は約 10 倍、
     GPU メモリは約 20 GB かかる（内部の点 24576 点の場合。足りなければ --n_pde を減らす）
  B. SIREN: 活性化関数を sin(ω0 ·) にする（Sitzmann et al. 2020 の初期化）。ω0 = 3。
     ω0 = 30（論文の標準値）は学習率 1e-3 では不安定で学習できなかった

共通の要点:
  1. 座標は物理座標（格子幅 dx, dy, dz を明示）。データに格子幅の情報はないので引数で与える
  2. 無次元化: 長さは領域の水平方向の大きさ L0、磁場は下端の |B| の RMS B0 で割る。
     ネットワークは無次元量 (x̂, ŷ, ẑ) → B̂ を学習し、損失もすべて無次元量で計算する
  3. 境界条件は下端の B の 3 成分だけ（側面・上端には何も課さない）
  4. 損失は 3 項とも B について 2 次になるようにそろえる
       フォースフリー: |Ĵ×B̂|² / (|B̂| + ε)²   （分母の |B̂| には勾配を流さない）
       div B:          (∇̂·B̂)²       ← この版では使わない（B = ∇×A なので常に 0）
       下端境界:       |B̂ − B̂_obs|²
  5. 内部の点は、水平方向は一様、鉛直方向は密度 ∝ (1 − a ζ)²（下端と上端の密度比 10:1）
  6. ネットワークは全結合 5 層 × 128（この版では活性化関数 sin、出力は Â の 3 成分）
  7. 学習率は Adam 1e-3 から CosineAnnealing で単調に 1e-5 まで下げる

データは lowlou.io の約束（npz の "b" に shape (nx, ny, nz, 3)、b[ix, iy, iz] = (Bx, By, Bz)、iz = 0 が下端）。
読み書き・評価・可視化はリポジトリの共通ライブラリ lowlou を使う（リポジトリで pip install -e . しておく）。

使い方（出力はこのスクリプトと同じディレクトリに書かれる）:
  python results/vp_siren/train.py                          # 既定の設定で 1 万ステップ
  python results/vp_siren/train.py --steps 2000             # 短く試す
  CUDA_VISIBLE_DEVICES=0 python results/vp_siren/train.py   # 使う GPU を選ぶ

出力: config.json, log.jsonl, evals.jsonl, model_<step>.pt, model.pt, B_pred.npz, result.json（lowlou.io.Run を参照）
結果を見る:
  python -m lowlou.viz loss results/vp_siren
  python -m lowlou.viz slice results/vp_siren/B_pred.npz --ref data/b_0.210_0.124.npz --y 32 --symlog 1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from lowlou import DEFAULT_DATA, Run, compare, load_field

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 設定
ap = argparse.ArgumentParser(description="PINN による NLFFF 再構成（ベクトルポテンシャル + SIREN 版）")
ap.add_argument("--data", default=DEFAULT_DATA)
ap.add_argument("--out", default=HERE, help="出力ディレクトリ（既定はこのスクリプトのディレクトリ）")
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--eval_every", type=int, default=2500, help="このステップごとに評価してモデルを保存する")
ap.add_argument("--dx", type=float, default=1.0)
ap.add_argument("--dy", type=float, default=1.0)
ap.add_argument("--dz", type=float, default=1.0)
ap.add_argument("--n_pde", type=int, default=24576, help="内部の点の数（毎ステップ取り直す）")
ap.add_argument("--n_bc", type=int, default=2024, help="下端の点の数（毎ステップ取り直す）")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

W_FF, W_BC = 1.0, 10.0               # 損失の重み（div B の項はない）
Z_RATIO = 10.0                       # 内部の点の密度比（下端 : 上端）
EPS_PHYS = 0.1                       # フォースフリー項の分母に足す値（物理単位の B）
LR_MIN = 1e-5                        # 学習率の最終値
WIDTH, DEPTH = 128, 5                # ネットワークの幅と層数
OMEGA0 = 3.0                         # SIREN の ω0


# ---------------------------------------------------------------- ネットワーク
class MLP(nn.Module):
    """全結合ネットワーク（SIREN: 活性化関数 sin(ω0 ·)）。入力は xyz 共通の値 s で [0, s] → [-1, 1] に正規化する。"""

    def __init__(self, Lx, Ly, Lz, width=WIDTH, depth=DEPTH, omega0=OMEGA0):
        super().__init__()
        self.s = max(Lx, Ly, Lz)
        self.omega0 = omega0
        dims = [3] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(width, 3)
        # SIREN の初期化: 1 層目は U(-1/in, 1/in)、それ以降は U(-sqrt(6/in)/ω0, sqrt(6/in)/ω0)（バイアスは既定のまま）
        with torch.no_grad():
            for k, lin in enumerate(self.hidden):
                bound = 1.0 / lin.in_features if k == 0 else np.sqrt(6.0 / lin.in_features) / omega0
                lin.weight.uniform_(-bound, bound)
            bound = np.sqrt(6.0 / width) / omega0
            self.out.weight.uniform_(-bound, bound)

    def forward(self, x, y, z):
        h = 2.0 * torch.cat([x, y, z], dim=1) / self.s - 1.0
        for lin in self.hidden:
            h = torch.sin(self.omega0 * lin(h))
        return self.out(h)


def field(model, x, y, z, create_graph=True):
    """ネットワークの出力を Â とし、B̂ = ∇̂×Â を返す（x, y, z は requires_grad=True であること）。"""
    out = model(x, y, z)
    g = lambda u, v: torch.autograd.grad(u, v, grad_outputs=torch.ones_like(u), create_graph=create_graph,
                                         retain_graph=True)[0]
    A = [out[:, c:c + 1] for c in range(3)]
    return torch.cat([g(A[2], y) - g(A[1], z), g(A[0], z) - g(A[2], x), g(A[1], x) - g(A[0], y)], dim=-1)


# ---------------------------------------------------------------- 微分
def grad(u, x):
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]


def curl_div(B, x, y, z):
    """B の回転 J = ∇×B と発散 ∇·B。"""
    d = [[grad(B[:, c:c + 1], v) for v in (x, y, z)] for c in range(3)]  # d[c][k] = ∂B_c/∂x_k
    J = torch.cat([d[2][1] - d[1][2], d[0][2] - d[2][0], d[1][0] - d[0][1]], dim=-1)
    return J, d[0][0] + d[1][1] + d[2][2]


# ---------------------------------------------------------------- 下端境界
class Bottom:
    """下端 z = 0 の B̂（無次元）を、無次元座標 (x̂, ŷ) で双線形補間する。周期境界は使わない。"""

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


# ---------------------------------------------------------------- 評価
def predict_grid(model, shape, h, device):
    """格子点上の B̂ を (nx, ny, nz, 3) で返す。h は無次元の格子幅 (dx/L0, dy/L0, dz/L0)。"""
    nx, ny, nz = shape
    X, Y, Z = torch.meshgrid(torch.arange(nx, device=device) * h[0], torch.arange(ny, device=device) * h[1],
                             torch.arange(nz, device=device) * h[2], indexing="ij")
    pts = [t.reshape(-1, 1).float() for t in (X, Y, Z)]
    out = []
    for c in zip(*(p.split(32768) for p in pts)):   # B = ∇×A に微分が要るので no_grad にはしない
        c = [t.clone().requires_grad_(True) for t in c]
        out.append(field(model, *c, create_graph=False).detach())
    return torch.cat(out).reshape(nx, ny, nz, 3).cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------- main
def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # データ
    b = load_field(args.data)                    # (nx, ny, nz, 3)
    nx, ny, nz = b.shape[:3]
    b0 = b[:, :, 0]                              # 下端 (nx, ny, 3)
    dx, dy, dz = args.dx, args.dy, args.dz

    # 無次元化の代表値
    L0 = (nx - 1) * dx                                   # 長さ: 水平方向の領域の大きさ
    B0 = float(np.sqrt(np.mean(np.sum(b0**2, axis=-1)))) # 磁場: 下端の |B| の RMS
    Lx, Ly, Lz = (nx - 1) * dx / L0, (ny - 1) * dy / L0, (nz - 1) * dz / L0   # 無次元の領域
    h = (dx / L0, dy / L0, dz / L0)                                         # 無次元の格子幅
    eps = EPS_PHYS / B0
    print(f"device = {device}, L0 = {L0}, B0 = {B0:.3f}, 領域 = [0, {Lx:.3f}] x [0, {Ly:.3f}] x [0, {Lz:.3f}]")
    run = Run(args.out, {"args": vars(args), "L0": L0, "B0": B0, "W": [W_FF, W_BC], "Z_RATIO": Z_RATIO,
               "EPS_PHYS": EPS_PHYS, "model": "siren", "omega0": OMEGA0, "vp": True})

    bottom = Bottom(b0, h[0], h[1], B0, device)
    model = MLP(Lx, Ly, Lz).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=LR_MIN)

    # 鉛直方向の点の分布: 密度 ∝ (1 - a ζ)²（ζ = z/Lz）、下端:上端 = Z_RATIO:1。逆関数法で生成する
    a = 1.0 - 1.0 / np.sqrt(Z_RATIO)
    c = 1.0 - (1.0 - a) ** 3

    def rand(n, L):
        return (0.0 + 1.0 * torch.rand(n, 1, device=device)) * L

    def rand_z(n):
        u = torch.rand(n, 1, device=device)
        return (1.0 - (1.0 - c * u) ** (1.0 / 3.0)) / a * Lz

    def evaluate():
        B = predict_grid(model, (nx, ny, nz), h, device) * B0   # 物理単位に戻す
        return B, compare(B, b, dx, dy, dz)

    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()

        # 内部の点: フォースフリー（div B は B = ∇×A なので常に 0）
        x = rand(args.n_pde, Lx).requires_grad_(True)
        y = rand(args.n_pde, Ly).requires_grad_(True)
        z = rand_z(args.n_pde).requires_grad_(True)
        B = field(model, x, y, z)
        J, div = curl_div(B, x, y, z)
        Bm = B.detach().norm(dim=-1) + eps
        loss_ff = (torch.sum(torch.cross(J, B, dim=-1) ** 2, dim=-1) * Bm ** -2.0).mean()

        # 下端の点: 境界条件
        xb, yb = rand(args.n_bc, Lx).requires_grad_(True), rand(args.n_bc, Ly).requires_grad_(True)
        Bb = field(model, xb, yb, torch.zeros_like(xb).requires_grad_(True))
        loss_bc = torch.sum((Bb - bottom(xb, yb)) ** 2, dim=-1).mean()

        loss = W_FF * loss_ff + W_BC * loss_bc
        loss.backward()
        opt.step()
        sched.step()

        if step % 100 == 0 or step == args.steps - 1:
            rec = {"step": step, "loss": loss.item(), "ff": loss_ff.item(), "div": 0.0,
                   "bc": loss_bc.item(), "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
            run.log(rec)
            if step % 1000 == 0:
                print(f"step {step:6d}  loss {rec['loss']:.4g}  (ff {rec['ff']:.3g}, bc {rec['bc']:.3g})  "
                      f"{rec['time_s']:.0f} s")
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            _, r = evaluate()
            run.log_eval({"step": step + 1, "time_s": time.time() - t0, **r})
            run.save_model(model, step + 1)
            en = r["E_n'"]
            print(f"  評価 step {step + 1}: C_vec {r['C_vec']:.4f}, E_n' {en:.3f}, sigma_J {r['sigma_J']:.3f}")

    elapsed = time.time() - t0
    run.save_model(model)
    B, r = evaluate()
    run.log_eval({"step": args.steps, "time_s": elapsed, **r})
    run.finish({"time_s": elapsed, **r}, B)
    print(json.dumps({"time_s": round(elapsed), **{k: round(v, 4) for k, v in r.items()}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
