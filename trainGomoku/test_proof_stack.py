#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_proof_stack.py
===================
証明用に追加した3ファイル (human_algorithm / arena / prove_stronger) の回帰テスト。

長い学習を回す *前* に 1〜2分で全部の経路を通して、
「1時間学習したあとに対戦コードが落ちる」事故を防ぐためのものです。

    cd trainGomoku && python test_proof_stack.py

確認する内容:
  1. 人手エンジンが即勝ち手を打ち、相手のリーチを止めること
  2. 人手エンジンが非合法手を返さないこと
  3. 先読みありのレベルがランダムを圧倒すること（対照群として使える強さか）
  4. arena が「手番と石の色」を壊さずに1局打てること
     -> Util.get_next_node() の落とし穴を踏んでいないかの実地確認
  5. AZ の入力特徴（履歴）が訓練時と同じ形で作られていること
  6. 統計関数が既知の値と一致すること
"""

import copy
import math
import sys

import numpy as np
import torch

sys.path.append('.')
sys.path.append('..')

from AlphaZeroCode import Node, Util
from AlphaZeroCode.env.Gomoku import Gomoku

from CFG_7x7 import get_cfg
from az_common import make_env, winning_moves
from human_algorithm import HumanAlgorithm, RandomPlayer, engine_vs_engine
from arena import (play_match, play_one_game, _build_node, wilson_interval,
                   binomial_p_value, elo_diff, board_to_text)

failures = []


def check(name, cond, detail=''):
    print(f'  {"ok  " if cond else "FAIL"}  {name}' + (f'   {detail}' if detail else ''),
          flush=True)
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------------
def test_engine_tactics():
    print('[human_algorithm] 戦術')
    w, row = 7, 5
    eng = HumanAlgorithm(width=w, row=row, level=2, seed=0)

    # 先手(-1) が (3,4) で5連
    state = [[0] * w for _ in range(w)]
    for c in range(4):
        state[3][c] = -1
    state[6][0], state[6][2] = 1, 1
    a = eng.select(state, -1)
    check('即勝ち手を打つ', a == 3 * w + 4, f'chose {divmod(a, w)}')

    # 後手(1) が (3,4) で5連 -> 先手(-1) は止めるしかない
    state = [[0] * w for _ in range(w)]
    for c in range(4):
        state[3][c] = 1
    state[0][0] = -1
    a = eng.select(state, -1)
    check('相手のリーチを止める', a == 3 * w + 4, f'chose {divmod(a, w)}')

    # 止めるより自分の勝ちを優先する
    state = [[0] * w for _ in range(w)]
    for c in range(4):
        state[1][c] = -1       # 自分のリーチ
    for c in range(4):
        state[5][c] = 1        # 相手のリーチ
    a = eng.select(state, -1)
    check('自分の勝ちを優先', a == 1 * w + 4, f'chose {divmod(a, w)}')

    # L1(先読みなし) でも上の3つは成立する（ルールで処理しているため）
    l1 = HumanAlgorithm(width=w, row=row, level=1, seed=0)
    state = [[0] * w for _ in range(w)]
    for c in range(4):
        state[3][c] = 1
    a = l1.select(state, -1)
    check('L1 も防御はできる', a == 3 * w + 4, f'chose {divmod(a, w)}')

    # 勝ち判定ユーティリティが az_common と一致するか
    state = [[0] * w for _ in range(w)]
    for c in range(4):
        state[2][c] = -1
    flat = [state[r][c] for r in range(w) for c in range(w)]
    mine = sorted(a for a in range(w * w)
                  if state[a // w][a % w] == 0 and eng._wins_at(flat, a, -1))
    theirs = sorted(winning_moves(state, -1, row, w))
    check('勝ち判定が az_common と一致', mine == theirs, f'{mine} vs {theirs}')


def test_engine_legality():
    print('[human_algorithm] 合法性と終局')
    w, row = 7, 5
    eng = HumanAlgorithm(width=w, row=row, level=2, seed=1)
    rnd = RandomPlayer(width=w, seed=2)
    env = Gomoku(row=row, width=w)

    ok_legal = True
    for g in range(3):
        env.reset()
        while not env.done:
            player = env.player
            a = eng.select(env.state, player) if player == -1 else rnd.select(env.state, player)
            if env.state[a // w][a % w] != 0:
                ok_legal = False
                break
            env.step(a)
    check('非合法手を返さない', ok_legal)
    check('局が必ず終わる', env.done)


def test_engine_strength():
    print('[human_algorithm] 対照群としての強さ（少数サンプル）')
    rnd = RandomPlayer(width=7, seed=5)
    l2 = HumanAlgorithm(width=7, row=5, level=2, seed=6)
    r = engine_vs_engine(l2, rnd, 7, 5, n_games=10, seed=6)
    check('L2 がランダムに 80% 以上', r['score_rate'] >= 0.8,
          f'score={r["score_rate"]:.0%} (w{r["win"]} d{r["draw"]} l{r["lose"]})')


# ---------------------------------------------------------------------
def tiny_cfg():
    """テスト用の最小構成。"""
    return get_cfg('fast', work_dir='/tmp/az_proof_test',
                   model_name='test_proof',
                   n_residual_block=1, resnet_channels=16, hidden_size=32,
                   num_simulation=12, Dirichlet_epsilon=0.25)


def test_node_history(CFG):
    print('[arena] ノード履歴の整合')
    env = make_env(CFG)
    util = Util(CFG)
    opening = [24, 25]
    node = _build_node(CFG, env, util, opening)

    check('states の長さ = history_size', len(node.states) == CFG.history_size,
          f'{len(node.states)}')
    check('states[0] が現在の盤面', node.states[0] == env.state)
    check('手番が正しい', node.player == CFG.first_player and env.player == CFG.first_player,
          f'node={node.player} env={env.player}')

    feat = util.state2feature(node)
    expect = (1, CFG.history_size * 2 + 1, CFG.board_width, CFG.board_width)
    check('入力特徴の形が訓練時と同じ', tuple(feat.shape) == expect,
          f'{tuple(feat.shape)} vs {expect}')

    # オープニングの石が正しい色で置かれているか
    b = np.array(env.state)
    check('オープニングの石数が正しい', int((b != 0).sum()) == len(opening),
          f'{int((b != 0).sum())}')
    check('先手の石が1つ', int((b == CFG.first_player).sum()) == 1)
    check('後手の石が1つ', int((b == CFG.second_player).sum()) == 1)


def test_one_game(CFG):
    print('[arena] 1局通し（手番と石の色が壊れないか）')
    from AlphaZeroCode.network.AlphaZeroNetwork import AlphaZeroNetwork

    env = make_env(CFG)
    model = AlphaZeroNetwork(CFG).to(CFG.device)
    model.eval()
    eng = HumanAlgorithm(width=CFG.board_width, row=CFG.row, level=1, seed=3)

    for az_player, label in ((CFG.first_player, 'AZ先手'), (CFG.second_player, 'AZ後手')):
        g = play_one_game(env, model, CFG, eng, az_player, [24], record=True)
        b = np.array(g['board'])
        n_first = int((b == CFG.first_player).sum())
        n_second = int((b == CFG.second_player).sum())
        check(f'{label}: 石数の差が1以下', abs(n_first - n_second) <= 1,
              f'first={n_first} second={n_second} moves={g["moves"]}')
        check(f'{label}: 石数合計 = 手数', n_first + n_second == g['moves'],
              f'{n_first + n_second} vs {g["moves"]}')
        check(f'{label}: 結果が取れている', g['result'] in ('win', 'draw', 'lose'),
              f'{g["result"]}')
    print(board_to_text(g['board']))


def test_match(CFG):
    print('[arena] マッチ集計')
    from AlphaZeroCode.network.AlphaZeroNetwork import AlphaZeroNetwork

    env = make_env(CFG)
    model = AlphaZeroNetwork(CFG).to(CFG.device)
    model.eval()
    eng = HumanAlgorithm(width=CFG.board_width, row=CFG.row, level=1, seed=4)

    eps_before = CFG.Dirichlet_epsilon
    res = play_match(env, model, CFG, eng, n_pairs=2, seed=0,
                     opening_plies=2, verbose=False, record_games=1)

    check('対局数 = ペア×2', res['games'] == 4, f'{res["games"]}')
    check('勝敗の合計が対局数', res['win'] + res['draw'] + res['lose'] == 4)
    check('先手後手が半々',
          sum(res['as_first'].values()) == 2 and sum(res['as_second'].values()) == 2)
    check('ディリクレノイズが元に戻る', CFG.Dirichlet_epsilon == eps_before,
          f'{CFG.Dirichlet_epsilon} vs {eps_before}')
    check('p値が [0,1]', 0.0 <= res['p_value'] <= 1.0, f'{res["p_value"]:.3g}')


# ---------------------------------------------------------------------
def test_stats():
    print('[stats] 統計関数')
    check('二項検定 10勝0敗', abs(binomial_p_value(10, 0) - 1 / 1024) < 1e-12,
          f'{binomial_p_value(10, 0):.6f}')
    check('二項検定 5勝5敗 > 0.5', binomial_p_value(5, 5) > 0.5,
          f'{binomial_p_value(5, 5):.4f}')
    check('二項検定 全敗なら 1.0', abs(binomial_p_value(0, 8) - 1.0) < 1e-12)
    check('Elo 0.5 -> 0', abs(elo_diff(0.5)) < 1e-9)
    check('Elo 0.76 -> 約200', abs(elo_diff(0.7597) - 200) < 2.0,
          f'{elo_diff(0.7597):.1f}')
    lo, hi = wilson_interval(5, 10)
    check('Wilson 中心が 0.5', abs((lo + hi) / 2 - 0.5) < 1e-9, f'[{lo:.3f}, {hi:.3f}]')
    lo, hi = wilson_interval(40, 40)
    check('Wilson 全勝で下限 > 0.9', lo > 0.9, f'[{lo:.3f}, {hi:.3f}]')


# ---------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    test_engine_tactics()
    test_engine_legality()
    test_engine_strength()

    CFG = tiny_cfg()
    print(f'[cfg] {CFG.summary()}')
    test_node_history(CFG)
    test_one_game(CFG)
    test_match(CFG)
    test_stats()

    print()
    if failures:
        print(f'{len(failures)} 件失敗: {failures}')
        return 1
    print('すべて成功。学習に進んで大丈夫です。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
