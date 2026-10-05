"""データの読み書きと、実験の出力（設定・ログ・評価値・モデル）の保存。

磁場データの約束:
    npz の "b" に shape (nx, ny, nz, 3) の配列を入れる。b[ix, iy, iz] = (Bx, By, Bz)、iz = 0 が下端。
    格子幅の情報は持たない（必要なら学習スクリプトの引数で与える）。
"""
import json
import os
import subprocess

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "input")
DEFAULT_DATA = os.path.join(DATA_DIR, "b_0.210_0.124.npz")   # Low & Lou の解析解（64³）


def load_field(path=DEFAULT_DATA):
    """npz から磁場 b（shape (nx, ny, nz, 3)）を読む。"""
    b = np.load(path)["b"]
    if b.ndim != 4 or b.shape[-1] != 3:
        raise ValueError(f"{path}: 'b' の shape は (nx, ny, nz, 3) のはずが {b.shape}")
    return b


def save_field(path, b):
    """磁場 b（shape (nx, ny, nz, 3)）を load_field で読める形式で保存する。"""
    np.savez(path, b=np.asarray(b))


def read_jsonl(path):
    """1 行 1 つの JSON のファイルを読む（無ければ空のリスト）。"""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def git_info(path=REPO_DIR):
    """リポジトリのコミットと、未コミットの変更があるかどうか。"""
    def run(*cmd):
        return subprocess.run(["git", "-C", path, *cmd], capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except OSError:
        return {"commit": None, "dirty": None}


class Run:
    """1 回の学習の出力を out_dir にまとめる。

    config.json   設定（git のコミットも記録する）
    log.jsonl     学習中の損失など（log で 1 行ずつ追記）
    evals.jsonl   途中評価（log_eval で 1 行ずつ追記）
    model_<step>.pt, model.pt   モデル
    B_pred.npz    格子点上の予測（save_field の形式）
    result.json   最終的な評価値
    """

    def __init__(self, out_dir, config):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        with open(self.path("config.json"), "w") as f:
            json.dump({**config, "git": git_info()}, f, indent=1, ensure_ascii=False)
        self._log = open(self.path("log.jsonl"), "w")
        self._evals = open(self.path("evals.jsonl"), "w")

    def path(self, name):
        return os.path.join(self.dir, name)

    def log(self, rec):
        print(json.dumps(rec), file=self._log, flush=True)

    def log_eval(self, rec):
        print(json.dumps(rec), file=self._evals, flush=True)

    def save_model(self, model, step=None):
        import torch
        torch.save(model.state_dict(), self.path("model.pt" if step is None else f"model_{step}.pt"))

    def finish(self, result, B=None):
        """最終的な評価値を保存する。B を渡すと B_pred.npz も保存する。"""
        with open(self.path("result.json"), "w") as f:
            json.dump(result, f, indent=1)
        if B is not None:
            save_field(self.path("B_pred.npz"), B)
        self._log.close()
        self._evals.close()
