#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
az_common.py
============
AlphaZeroCode (Michi-123) を 7x7 五目並べで回すための共通部品。

ライブラリ本体には手を入れず、この層で以下の3点を吸収します。

1. Gomoku.get_legal_actions() が引数を取らないのに
   Agent.random(state) が get_legal_actions(state) を呼ぶ問題
2. Train.update() が running_loss に「勾配付きテンソル」を足し込むため
   1エポック内で計算グラフが溜まる問題（長時間学習でメモリ増）
3. Evaluate クラスが勝敗数を return しないため、勝率を数値で取れない問題
"""

import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from AlphaZeroCode import Node, Util, Train
from AlphaZeroCode.MCTS import MCTS
from AlphaZeroCode.Agent import Agent


# =====================================================================
# 1. 環境ラッパ
# =====================================================================
def make_env(CFG):
    """CFG に合わせた Gomoku 環境を返す。"""
    from AlphaZeroCode.env.Gomoku import Gomoku

    class GomokuEnv(Gomoku):
        """get_legal_actions を state 省略可にしたラッパ。"""

        def get_legal_actions(self, state=None):
            s = self.state if state is None else state
            return np.where(np.array(s).reshape(-1) == 0)[0]

        def clone(self):
            return copy.deepcopy(self)

    return GomokuEnv(row=CFG.row, width=CFG.board_width)


def is_winning_move(state, action, player, row, width):
    """state に player が action を打つと row 連が成立するか。"""
    x1, x2 = action // width, action % width
    if state[x1][x2] != 0:
        return False

    for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
        count = 1
        for sign in (1, -1):
            i = 1
            while True:
                r, c = x1 + i * dr * sign, x2 + i * dc * sign
                if 0 <= r < width and 0 <= c < width and state[r][c] == player:
                    count += 1
                    i += 1
                else:
                    break
        if count >= row:
            return True
    return False


def winning_moves(state, player, row, width):
    """player にとっての即勝ち手のリスト。"""
    legal = np.where(np.array(state).reshape(-1) == 0)[0]
    return [int(a) for a in legal
            if is_winning_move(state, int(a), player, row, width)]


# =====================================================================
# 2. メモリ安全 & 学習率可変の Train
# =====================================================================
class TrainFixed(Train):
    """
    Train を継承し、update() の running_loss をスカラー化。
    さらに学習率を後から差し替えられるようにする。
    """

    def update(self, input_features, pi, z):
        p, v = self.model(input_features)

        p = p + 1e-10
        policy_loss = -(pi * torch.log(p)).sum(dim=1).mean()
        value_loss = (z - v).pow(2).mean()
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        # ← ここが本家との違い。detach しないとグラフが残る
        self.running_loss_policy += policy_loss.item()
        self.running_loss_value += value_loss.item()

    def __call__(self, dataset, quiet=True):
        """
        本家 Train.__call__ とほぼ同じだが
        ・エポックごとの損失を self.history に残す（本家は毎エポック 0 に戻すため取れない）
        ・quiet=True で標準出力を抑制
        """
        self.model.train()
        n_batch = max(1, math.ceil(len(dataset) / self.CFG.batch_size))
        self.history = []

        for epoch in range(1, self.num_epoch + 1):
            dataset = random.sample(dataset, len(dataset))

            for i in range(0, len(dataset), self.CFG.batch_size):
                input_features, pi, z = self.util.make_batch(
                    dataset[i:i + self.CFG.batch_size])
                self.update(input_features, pi, z)

            pol = self.running_loss_policy / n_batch
            val = self.running_loss_value / n_batch
            self.history.append((pol, val))

            if not quiet:
                print(f'  epoch {epoch:02}/{self.num_epoch} lr={self.CFG.learning_rate:.5f}'
                      f'  pi_loss={pol:.5f}  v_loss={val:.5f}')

            self.running_loss_policy = 0.0
            self.running_loss_value = 0.0

        return self.history[-1]

    def set_lr(self, lr):
        self.CFG.learning_rate = lr          # ログ表示用
        for g in self.optimizer.param_groups:
            g['lr'] = lr


# =====================================================================
# 3. 勝率が取れる評価
# =====================================================================
def play_vs_random(env, model, CFG, n_games=20, az_first=True, seed=None):
    """
    AlphaZero vs ランダムプレイヤー。
    戻り値: dict(win, draw, lose, win_rate, score_rate, avg_moves)
    score_rate = (勝ち + 0.5*引分) / 対局数
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    agent = Agent(env, model, CFG, train=False)
    util = Util(CFG)

    win = draw = lose = 0
    total_moves = 0

    for _ in range(n_games):
        state = env.reset()
        node = Node(CFG, state)
        node.player = CFG.first_player if az_first else CFG.second_player
        az_turn = az_first
        moves = 0
        result = None

        while True:
            if az_turn:
                legal = env.get_legal_actions()
                if len(legal) == 0:
                    result = 'draw'
                    break
                node = agent.alpha_zero(node)
                action = node.action
                state, reward, done = env.step(action)
                moves += 1
                if done:
                    # reward は「着手後に手番となる側」から見た値なので、
                    # 打った本人が勝った場合は -1 になる（Gomoku.py 参照）
                    result = 'win' if reward == -1 else 'draw'
                    break
            else:
                legal = env.get_legal_actions()
                if len(legal) == 0:
                    result = 'draw'
                    break
                action = int(random.choice(legal))
                state, reward, done = env.step(action)
                moves += 1
                if done:
                    result = 'lose' if reward == -1 else 'draw'
                    break
                node = util.get_next_node(node, action, env)

            az_turn = not az_turn

        total_moves += moves
        if result == 'win':
            win += 1
        elif result == 'lose':
            lose += 1
        else:
            draw += 1

    n = max(1, n_games)
    return {
        'win': win, 'draw': draw, 'lose': lose,
        'win_rate': win / n,
        'score_rate': (win + 0.5 * draw) / n,
        'avg_moves': total_moves / n,
    }


def evaluate_both_sides(env, model, CFG, n_games=20, seed=0):
    """先手・後手を半分ずつ持って評価する（先手有利の影響を除く）。"""
    half = max(1, n_games // 2)
    a = play_vs_random(env, model, CFG, half, az_first=True, seed=seed)
    b = play_vs_random(env, model, CFG, half, az_first=False, seed=seed + 1)

    total = half * 2
    return {
        'win': a['win'] + b['win'],
        'draw': a['draw'] + b['draw'],
        'lose': a['lose'] + b['lose'],
        'win_rate': (a['win'] + b['win']) / total,
        'score_rate': (a['score_rate'] + b['score_rate']) / 2,
        'first': a['score_rate'],
        'second': b['score_rate'],
        'avg_moves': (a['avg_moves'] + b['avg_moves']) / 2,
    }


# =====================================================================
# 4. 戦術テスト（学習の中身を直接見る）
# =====================================================================
_TACTIC_CACHE = {}


def build_tactic_testset(CFG, n_positions=100, seed=12345, min_empty=15):
    """
    「1手で勝てる局面」のテストセットを作る。

    盤面を途中(4〜16手)で打ち切る方式だと 4連が出来ている確率が低く、
    探索が延々と空回りして極端に遅くなる。
    そこでランダム対局を最後まで進めながら該当局面を拾う(bootstrap と同方式)。
    seed を bootstrap と変えることで hold-out として機能する。
    """
    key = (CFG.board_width, CFG.row, CFG.history_size, n_positions, seed, min_empty)
    if key in _TACTIC_CACHE:
        return _TACTIC_CACHE[key]

    rng = random.Random(seed)
    util = Util(CFG)
    env = make_env(CFG)
    samples = []
    guard = n_positions * 500

    while len(samples) < n_positions and guard > 0:
        guard -= 1
        state = env.reset()
        states = copy.deepcopy(Node(CFG, state).states)
        player = CFG.first_player

        for _ in range(CFG.action_size):
            legal = [int(a) for a in env.get_legal_actions()]
            if not legal:
                break
            wins = winning_moves(env.state, player, CFG.row, CFG.board_width)
            # 空きマスが少ない終盤局面はランダムでも当たるので除外する
            if wins and len(legal) >= min_empty and len(samples) < n_positions:
                samples.append((copy.deepcopy(states), player,
                                set(wins), copy.deepcopy(env.state), len(legal)))
            a = int(rng.choice(legal))
            states = util.get_next_states(states, a, player, env)
            _, _, done = env.step(a)
            player = -player
            if done:
                break

    _TACTIC_CACHE[key] = samples
    return samples


def tactical_test(env, model, CFG, n_positions=100, seed=12345):
    """
    policy の argmax が「即勝ち手」を指す割合。
    勝率より遥かに早く学習の進み具合が見える指標。
    ランダム初期化のベースラインは概ね 3〜8%。
    """
    samples = build_tactic_testset(CFG, n_positions, seed)
    if not samples:
        return {'accuracy': 0.0, 'tested': 0}

    util = Util(CFG)
    model.eval()
    hit = 0

    for states, player, wins, board, n_legal in samples:
        node = Node(CFG, board)
        node.states = copy.deepcopy(states)
        node.player = player
        with torch.no_grad():
            p, _ = model(util.state2feature(node))
        p = p[0].detach().cpu().numpy()
        mask = (np.array(board).reshape(-1) == 0)
        p = np.where(mask, p, -1.0)
        if int(np.argmax(p)) in wins:
            hit += 1

    base = float(np.mean([len(w) / n for _, _, w, _, n in samples]))
    return {'accuracy': hit / len(samples), 'tested': len(samples),
            'random_baseline': base}
