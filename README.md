# Lowlou_NN

PINN（Physics-Informed Neural Network）で非線形フォースフリー磁場を再構成するコードです。
下端（光球面）の磁場ベクトル B = (Bx, By, Bz) だけを境界条件として与え、フォースフリー条件 J×B = 0 と div B = 0 を満たす 3 次元の磁場を求めます。
テスト問題として Low & Lou の解析解（`data/b_0.210_0.124.npz`、64³）を使い、厳密解と比べて評価します。

## ディレクトリ構成

```text
Lowlou_NN/
├── data/b_0.210_0.124.npz   # Low & Lou の解析解
├── lowlou/                  # 共通ライブラリ（どの実験でも同じ部分）
│   ├── io.py                #   データの読み書き、実験の出力（設定・ログ・評価値・モデル）の保存
│   ├── metrics.py           #   厳密解との比較、中央差分による div B・J×B の確認
│   └── viz.py               #   磁場の断面、学習の経過（損失・評価値）の図
├── results/                 # 実験ごとに 1 ディレクトリ。学習スクリプトとその出力を同じ場所に置く
│   ├── baseline/train.py    #   ベースライン（tanh、B を直接出力）
│   └── vp_siren/train.py    #   ベクトルポテンシャル + SIREN
└── pyproject.toml
```

- **`lowlou/`** には、データ形式・評価・可視化のように、手法を変えても同じ部分だけを置きます。
- **`results/<実験名>/train.py`** に手法のすべて（ネットワーク、損失、点の取り方、学習）を 1 ファイルで書きます。新しい実験は、既存のディレクトリをコピーして `train.py` を書き換えるのが基本です。
- 出力（`config.json`, `log.jsonl`, `evals.jsonl`, `result.json`, モデル、予測）は、`train.py` と同じディレクトリに書かれます。`*.pt`, `*.npz`, `*.png` は大きいので git 管理外です（`.gitignore`）。
- `config.json` には、すべての設定と git のコミット（未コミットの変更があったかどうかも）を記録します。

## 環境構築

```bash
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib torch
.venv/bin/pip install -e .          # lowlou をどこからでも import できるようにする
```

PyTorch は、GPU ドライバの CUDA バージョンに合ったものを入れてください。たとえば CUDA 12.x のドライバでは `torch==2.8.0`（PyPI では CUDA 12.8 版）が使えます。CUDA 13 版の torch を入れると GPU が使えず、CPU で動いてしまいます（`torch.cuda.is_available()` で確認できます）。

## 使い方

```bash
# データを確認する
python -m lowlou.metrics                         # 軸の解釈、div B と J×B、格子幅の比 dz/dx
python -m lowlou.viz slice --y 32 --symlog 1     # y = 32 の鉛直断面（対数的な色スケール）

# 学習する（出力は results/baseline/ に書かれる）
CUDA_VISIBLE_DEVICES=0 python results/baseline/train.py
python results/baseline/train.py --steps 1000    # まず短く試すとき

# 結果を見る
python -m lowlou.viz loss results/baseline results/vp_siren -o loss.png
python -m lowlou.viz slice results/baseline/B_pred.npz --ref data/b_0.210_0.124.npz --y 32 --symlog 1
python -m lowlou.metrics results/baseline/B_pred.npz
```

`pip install -e .` すると、`python -m lowlou.metrics` / `python -m lowlou.viz` の代わりに `lowlou-check` / `lowlou-viz` とも書けます。

学習スクリプトの主な引数: `--steps`（既定 1 万）、`--eval_every`（評価とモデル保存の間隔、既定 2500）、`--dx --dy --dz`（格子幅、既定 1）、`--n_pde`（内部の点の数）、`--n_bc`（下端の点の数）、`--lr`、`--seed`、`--device`。
損失の重みや点の分布などの設定は、各 `train.py` の先頭に定数としてまとめてあります。

## データの約束

npz の `"b"` に shape `(nx, ny, nz, 3)` の配列を入れます。`b[ix, iy, iz] = (Bx, By, Bz)`、`iz = 0` が下端です。
格子幅の情報は持たないので、学習スクリプトの引数で与えます。同梱のデータは dx = dy = dz です（`python -m lowlou.metrics` で確認できます）。
学習スクリプトが保存する予測 `B_pred.npz` も同じ形式（物理単位）なので、データと同じツールで扱えます。

## 手法

### 共通（`baseline`, `vp_siren`）

1. **格子幅を明示して無次元化する**: 格子点 (ix, iy, iz) の位置を (ix·dx, iy·dy, iz·dz) とし、長さは水平方向の領域の大きさ L0 = (nx−1)·dx、磁場は下端の |B| の RMS B0 で割ります。ネットワークは無次元量 (x̂, ŷ, ẑ) → B̂ を学習し、損失もすべて無次元量で計算します。長さの単位を変えると物理の損失と境界の損失の比が L0² 倍変わるので、この無次元化は重要です。
2. **境界条件は下端の B の 3 成分だけ**: 側面と上端には何も課しません。下端の値は格子点の値から双線形補間します。
3. **損失の 3 項を、B について同じ 2 次にそろえる**:

   ```text
   loss = 1 × mean( |Ĵ×B̂|² / (|B̂| + ε)² )   … フォースフリー（分母の |B̂| には勾配を流さない）
        + 1 × mean( (∇̂·B̂)² )                … div B
        + 10 × mean( |B̂ − B̂_obs|² )          … 下端境界
   ```

   次数がそろっていないと、B0 の選び方で項の釣り合いが変わってしまいます。また、|B| の高いべきで割ると、ネットワークが |B| を大きくするだけで損失を下げられてしまいます。
4. **内部の点は下端付近に多く取る**: 水平方向は一様、鉛直方向は密度 ∝ (1 − z/Lz)² で、下端と上端の比は 10:1 です（毎ステップ取り直す）。
5. **学習率は単調に下げる**: Adam 1e-3 から CosineAnnealing で 1e-5 まで（T_max = 総ステップ数）。
6. ネットワークは全結合 5 層 × 128。入力は xyz 共通の値で [−1, 1] に正規化します。

### `vp_siren` だけ

- **ベクトルポテンシャル**: ネットワークの出力を A とし、B = ∇×A とします。div B = 0 が式の上で厳密に成り立つので、div B の損失は使いません。その代わり J = ∇×∇×A に 2 階微分が必要で、計算時間はベースラインの約 10 倍、GPU メモリは約 20 GB かかります（足りなければ `--n_pde` を減らしてください）。
- **SIREN**: 活性化関数を sin(ω0 ·) にします（Sitzmann et al. 2020 の初期化）。ω0 = 3 です。ω0 = 30（論文の標準値）は、学習率 1e-3 では不安定で学習できませんでした。

## 評価指標

学習スクリプトは、格子点上の予測を厳密解と比べて `evals.jsonl` と `result.json` に書きます（`lowlou.metrics.compare`）。

| 指標 | 意味 | 完全一致 |
| --- | --- | --- |
| C_vec | ベクトル相関 Σ B·b / sqrt(Σ\|B\|² Σ\|b\|²) | 1 |
| C_CS | 各点での B と b のなす角の cos の平均 | 1 |
| E_n' | 1 − Σ\|B − b\| / Σ\|b\| | 1（0 なら B = 0 と同じ） |
| E_m' | 1 − 各点の相対誤差 \|B − b\|/\|b\| の平均 | 1 |
| energy | 磁気エネルギーの比 Σ\|B\|² / Σ\|b\|² | 1 |
| E_n_bottom | 下端での正規化誤差 | 0 |
| div_rel | div B の相対的な大きさ（2 次精度の中央差分） | 厳密解でも 0.012 |
| sigma_J | 電流と磁場のなす角の指標（2 次精度の中央差分） | 0（厳密解でも 0.025） |

C_vec, C_CS, E_n', E_m', energy は Schrijver et al. (2006) の指標、sigma_J は Wheatland et al. (2000) によるものです。
`*_center` は、側面付近を除いた中央部（8 ≤ ix, iy < nx − 8）での値です。
div_rel と sigma_J は差分で評価するので、厳密解でも打ち切り誤差の分だけ 0 になりません。

## 参考: 1 万ステップでの結果

| | E_n' | E_m' | C_CS | sigma_J | div_rel | 時間 |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 0.871 | 0.526 | 0.754 | 0.095 | 0.034 | 約 15 分（4 本同時に実行） |
| ベクトルポテンシャルのみ（tanh） | 0.923 | 0.728 | 0.932 | 0.069 | 0.012 | 約 60 分（単独で実行） |
| SIREN（ω0 = 3）のみ | 0.873 | 0.537 | 0.816 | 0.048 | 0.019 | – |

`vp_siren`（両方の組み合わせ）は、まだ最後まで回していません。
