#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_gomoku.py
==============
7x7 五目並べの環境と MCTS の最低限の回帰テスト。

    cd trainGomoku && python test_gomoku.py

確認する内容:
  1. Gomoku の勝利判定（横・縦・斜め・逆斜め）と引き分け、誤検知しないこと
  2. step() の報酬の符号（着手後に手番となる側から見た値 = 勝ち手の直後は -1）
  3. MCTS が「1手で勝てる局面」でその手を選ぶこと
     -> MCTS.select() の Q の符号が逆だと、勝ち手をむしろ避けるため落ちる
  4. bootstrap データの教師信号が正しい手に向いていること
"""

import copy
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append('.')
sys.path.append('..')

from AlphaZeroCode import Node, Util
from AlphaZeroCode.MCTS import MCTS
from AlphaZeroCode.env.Gomoku import Gomoku

from CFG_7x7 import get_cfg
from az_common import make_env, winning_moves
from bootstrap_7x7 import generate_bootstrap_dataset


failures = []


def check(name, cond):
    print(f'  {"ok  " if cond else "FAIL"}  {name}')
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------------
def test_env():
    print('[env] Gomoku rules')

    def board(cells, player):
        env = Gomoku(row=5, width=7)
        env.reset()
        for r, c in cells[:-1]:
            env.state[r][c] = player
        env.player = player
        r, c = cells[-1]
        return env.step(r * 7 + c)

    _, reward, done = board([(3, c) for c in range(5)], -1)
    check('horizontal 5 wins', done and reward == -1)

    _, reward, done = board([(r, 2) for r in range(5)], 1)
    check('vertical 5 wins', done and reward == -1)

    _, reward, done = board([(i, i) for i in range(5)], 1)
    check('diagonal 5 wins', done and reward == -1)

    _, reward, done = board([(i, 6 - i) for i in range(5)], -1)
    check('anti-diagonal 5 wins', done and reward == -1)

    _, reward, done = board([(0, c) for c in range(4)], 1)
    check('4 in a row does not win', not done)

    env = Gomoku(row=5, width=7)
    env.reset()
    env.state[0][0] = 1
    check('get_legal_actions excludes occupied',
          0 not in env.get_legal_actions() and len(env.get_legal_actions()) == 48)

    """ 5目並べが成立しない 3x3 盤を埋めきると引き分け """
    env = Gomoku(row=5, width=3)
    env.reset()
    player = -1
    for a in range(9):
        env.player = player
        _, reward, done = env.step(a)
        player = -player
    check('full board is a draw', done and reward == 0 and env.winner == 0)


# ---------------------------------------------------------------------
class UniformNet(nn.Module):
    """方策は一様、価値は 0 のダミー。終端報酬の扱いだけを見るために使う。"""

    def __init__(self, action_size):
        super().__init__()
        self.action_size = action_size

    def forward(self, x):
        b = x.shape[0]
        return (torch.full((b, self.action_size), 1.0 / self.action_size),
                torch.zeros((b, 1)))


def test_mcts_finds_immediate_win(CFG):
    print('[mcts] immediate win')
    np.random.seed(0)

    env = make_env(CFG)
    env.reset()
    for c in range(4):
        env.state[3][c] = -1          # 先手(-1) は (3,4) で5連
    for c in (0, 2, 4, 6):
        env.state[6][c] = 1           # 後手(1) 側に脅威はない
    env.player = -1

    win_action = 3 * CFG.board_width + 4
    check('test position has exactly one winning move',
          winning_moves(env.state, -1, CFG.row, CFG.board_width) == [win_action])

    mcts = MCTS(env, UniformNet(CFG.action_size), CFG, train=False)
    node = Node(CFG, copy.deepcopy(env.state))
    node.player = -1
    next_node = mcts(node)

    visits = {int(c.action): c.n for c in node.child_nodes}
    check(f'MCTS plays the winning move (chose {int(next_node.action)}, '
          f'visits={visits.get(win_action)})',
          int(next_node.action) == win_action)


def test_mcts_blocks(CFG):
    """相手のリーチを止められるか。

    「止めなければ負ける」ことを示すには 2 手先まで読む必要があり、方策が
    一様なダミーでは 400 シミュレーションでは足りない（自分の 44 通りの手それぞれ
    について、相手の 44 通りの応手から勝ち手を見つける必要がある）。
    ここでは十分な回数を回して、探索そのものが機能していることを確認する。
    """
    print('[mcts] blocks the opponent')
    np.random.seed(0)

    env = make_env(CFG)
    env.reset()
    for c in range(4):
        env.state[3][c] = 1           # 後手(1) が (3,4) で5連のリーチ
    env.state[0][0] = -1
    env.player = -1                   # 先手(-1) は止めるしかない

    block_action = 3 * CFG.board_width + 4

    mcts = MCTS(env, UniformNet(CFG.action_size), CFG, train=False)
    node = Node(CFG, copy.deepcopy(env.state))
    node.player = -1
    next_node = mcts(node)

    check(f'MCTS blocks the threat (chose {int(next_node.action)})',
          int(next_node.action) == block_action)


# ---------------------------------------------------------------------
def test_bootstrap(CFG):
    print('[bootstrap] labels')
    ds = generate_bootstrap_dataset(CFG, 200, verbose=False)
    check('dataset is not empty', len(ds) > 0)

    ok = True
    for features, pi, z, plain in ds:
        state = plain['state']
        player = plain['player']
        wins = winning_moves(state, player, CFG.row, CFG.board_width)
        threats = winning_moves(state, -player, CFG.row, CFG.board_width)
        targets = wins if wins else threats
        if int(np.argmax(pi)) not in targets:
            ok = False
            break
    check('argmax(pi) is a winning or blocking move', ok)
    check('z is in [-1, 1]', all(-1.0 <= d[2][0] <= 1.0 for d in ds))


# ---------------------------------------------------------------------
def cfg(num_simulation):
    return get_cfg('fast', work_dir='/tmp/az_test',
                   num_simulation=num_simulation, Dirichlet_epsilon=0.0)


def main():
    torch.manual_seed(0)

    test_env()
    test_mcts_finds_immediate_win(cfg(400))
    test_mcts_blocks(cfg(1500))
    test_bootstrap(cfg(400))

    print()
    if failures:
        print(f'{len(failures)} test(s) failed: {failures}')
        return 1
    print('all tests passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
