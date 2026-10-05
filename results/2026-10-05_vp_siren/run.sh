#!/bin/bash
# 学習して、結果の図を描く。出力はすべてこのディレクトリに書かれる。
#   ./run.sh                  # 1 万ステップ
#   ./run.sh --steps 1000     # ステップ数を変える
set -e
cd "$(dirname "$0")"
EXACT=../../input/b_0.210_0.124.npz

python train.py "$@"
python -m lowlou.viz loss . -o loss.png
python -m lowlou.viz slice B_pred.npz --ref $EXACT --z 0 -o slice_z0.png
python -m lowlou.viz slice B_pred.npz --ref $EXACT --y 32 --symlog -o slice_y32.png
