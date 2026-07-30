#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CFG_7x7.py
==========
Michi-123/AlphaZeroCode を 7x7 五目並べ (5目) で訓練するための設定。

AlphaZeroCode 側が要求する属性は以下の 27 個（grep で抽出済み）:
    Dirichlet_alpha, Dirichlet_epsilon, action_size, batch_size, board_width,
    check_point_relative_dir, cpuct, dataset_path, device, first_player,
    hidden_size, history_size, iteration_counter_path, learning_rate,
    make_check_point_frequency, max_dataset_size, model_path, n_residual_block,
    num_epoch, num_simulation, resnet_channels, second_player,
    sub_dataset_path, tau, tau_limit, weight_decay, (pass_)

【重要】pass_ は定義しないこと。
    MCTS.add_child_nodes() が hasattr(CFG,'pass_') でパス手ノードを追加するため、
    定義すると p[action_size] で IndexError になります（五目並べにパスは不要）。
"""

import os
import torch


class BaseCFG:
    """7x7 五目並べ共通設定"""

    # ---------------- ゲーム ----------------
    board_width = 7          # 7x7 盤
    row = 5                  # 5目並べ
    action_size = 49         # board_width ** 2
    first_player = -1        # Gomoku.reset() が player=-1 で始まるので固定
    second_player = 1

    # ---------------- ネットワーク ----------------
    history_size = 2         # 入力チャネル = history_size*2 + 1
    n_residual_block = 5
    resnet_channels = 128
    hidden_size = 256

    # ---------------- MCTS ----------------
    num_simulation = 200
    cpuct = 1.5
    tau = 1.0
    tau_limit = 12           # 序盤12手までは確率的に着手
    Dirichlet_alpha = 0.3    # ≒ 10 / action_size(49) → 0.2〜0.4 が妥当域
    Dirichlet_epsilon = 0.25

    # ---------------- 自己対局 ----------------
    num_selfplay_games = 25  # 1イテレーションあたりの対局数（本リポジトリ独自）
    max_dataset_size = 20000

    # ---------------- 学習 ----------------
    num_epoch = 3
    batch_size = 128
    learning_rate = 0.01
    weight_decay = 1e-4
    lr_schedule = {0: 0.01, 40: 0.005, 70: 0.001}  # {iteration: lr}

    # ---------------- ループ制御 ----------------
    num_iteration = 100
    make_check_point_frequency = 10
    eval_frequency = 10
    eval_games = 20

    # ---------------- ブートストラップ ----------------
    use_bootstrap = True
    bootstrap_samples = 4000

    # ---------------- パス ----------------
    work_dir = '.'
    model_name = 'gomoku_7x7'

    # ---------------- 実行環境 ----------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    render_mode = 0          # 0:なし 1:テキスト 2:画像

    # ------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """派生値の確定とディレクトリ作成。使う前に必ず1回呼ぶこと。"""
        cls.action_size = cls.board_width ** 2

        cls.model_dir = os.path.join(cls.work_dir, 'models')
        cls.data_dir = os.path.join(cls.work_dir, 'data')
        cls.result_dir = os.path.join(cls.work_dir, 'results')
        cls.check_point_relative_dir = os.path.join(cls.model_dir, 'checkpoint')

        for d in (cls.model_dir, cls.data_dir, cls.result_dir,
                  cls.check_point_relative_dir):
            os.makedirs(d, exist_ok=True)

        cls.model_path = os.path.join(cls.model_dir, cls.model_name + '.pt')
        cls.dataset_path = os.path.join(cls.data_dir, cls.model_name + '.npy')
        cls.sub_dataset_path = os.path.join(cls.data_dir, cls.model_name + '_sub.npy')
        cls.iteration_counter_path = os.path.join(cls.model_dir, cls.model_name + '_counter.txt')
        cls.bootstrap_path = os.path.join(cls.data_dir, cls.model_name + '_bootstrap.npy')
        cls.log_path = os.path.join(cls.result_dir, cls.model_name + '_log.csv')

        assert not hasattr(cls, 'pass_'), 'pass_ を定義すると MCTS が壊れます'
        return cls

    @classmethod
    def summary(cls):
        return (
            f"board={cls.board_width}x{cls.board_width} row={cls.row} | "
            f"blocks={cls.n_residual_block} ch={cls.resnet_channels} "
            f"hidden={cls.hidden_size} hist={cls.history_size} | "
            f"sims={cls.num_simulation} cpuct={cls.cpuct} "
            f"alpha={cls.Dirichlet_alpha} tau_limit={cls.tau_limit} | "
            f"games/iter={cls.num_selfplay_games} epoch={cls.num_epoch} "
            f"bs={cls.batch_size} lr={cls.learning_rate} | device={cls.device}"
        )


# =====================================================================
#  プリセット
# =====================================================================

class FastCFG(BaseCFG):
    """動作確認・デバッグ用。CPUでも1イテレーション数分。"""
    model_name = 'gomoku_7x7_fast'
    n_residual_block = 3
    resnet_channels = 64
    hidden_size = 128
    num_simulation = 50
    num_selfplay_games = 8
    max_dataset_size = 5000
    batch_size = 64
    num_epoch = 2
    learning_rate = 0.02
    lr_schedule = {0: 0.02, 20: 0.01, 35: 0.005}
    num_iteration = 40
    tau_limit = 10
    bootstrap_samples = 2000
    eval_frequency = 5


class BalancedCFG(BaseCFG):
    """★推奨★ GPUで実用的に強くなる設定。"""
    model_name = 'gomoku_7x7_balanced'
    # BaseCFG の値をそのまま使用


class StrongCFG(BaseCFG):
    """時間をかけて最大限強くする設定。GPU 必須。"""
    model_name = 'gomoku_7x7_strong'
    history_size = 3
    n_residual_block = 8
    resnet_channels = 192
    hidden_size = 256
    num_simulation = 400
    num_selfplay_games = 50
    max_dataset_size = 60000
    batch_size = 256
    num_epoch = 4
    cpuct = 1.25
    Dirichlet_alpha = 0.25
    tau_limit = 14
    num_iteration = 200
    lr_schedule = {0: 0.01, 60: 0.005, 120: 0.001, 170: 0.0005}
    bootstrap_samples = 8000
    make_check_point_frequency = 20


class CpuCFG(BaseCFG):
    """GPUなしで一晩回す想定の設定。"""
    model_name = 'gomoku_7x7_cpu'
    n_residual_block = 3
    resnet_channels = 64
    hidden_size = 128
    num_simulation = 100
    num_selfplay_games = 10
    max_dataset_size = 8000
    batch_size = 64
    num_epoch = 3
    num_iteration = 60
    lr_schedule = {0: 0.01, 25: 0.005, 45: 0.001}
    bootstrap_samples = 3000
    device = 'cpu'


PRESETS = {
    'fast': FastCFG,
    'balanced': BalancedCFG,
    'strong': StrongCFG,
    'cpu': CpuCFG,
}


def get_cfg(name='balanced', **overrides):
    """プリセット取得 + 任意の上書き。 例) get_cfg('balanced', num_simulation=300)"""
    if name not in PRESETS:
        raise KeyError(f'unknown preset: {name} (choose from {list(PRESETS)})')
    cfg = PRESETS[name]
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg.setup()
