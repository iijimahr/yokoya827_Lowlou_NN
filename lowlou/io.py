"""データの読み書き。

磁場データは npz の "b" に shape (nx, ny, nz, 3) の配列で入れる。
b[ix, iy, iz] = (Bx, By, Bz)、iz = 0 が下端。
"""
import json
import os

import numpy as np

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "input", "b_0.210_0.124.npz")   # Low & Lou の解析解（64³）


def load_field(path=DEFAULT_DATA):
    """npz から磁場 b（shape (nx, ny, nz, 3)）を読む。"""
    return np.load(path)["b"]


def save_field(path, b):
    """磁場 b（shape (nx, ny, nz, 3)）を npz に保存する。"""
    np.savez(path, b=b)


def read_jsonl(path):
    """1 行に 1 つの JSON が書かれたファイルを読む。"""
    with open(path) as f:
        return [json.loads(line) for line in f]
