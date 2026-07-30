#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
arena.py
========
学習済み AlphaZero モデルと「人の考えたアルゴリズム」を戦わせ、
勝敗差が偶然でないことを統計的に判定するための対戦場。

公平性のために次の4点を守っています。

  1. 先後の入れ替え（ペア方式）
     同一の初期配置(オープニング)から、AZ が先手の1局と後手の1局を必ず
     セットで打つ。五目並べは先手有利なので、これをしないと比較にならない。

  2. オープニングのランダム化
     AZ も相手エンジンも決定的に指すため、同じ初形だと毎局まったく同じ
     棋譜になる。ランダムな数手を初形として与えて局面を散らす。

  3. 評価時はディリクレノイズを切る
     MCTS.select() は is_root なら train/eval に関係なくノイズを足すので、
     対戦中だけ Dirichlet_epsilon = 0 にする（本来の実力で指させる）。

  4. 統計的検定
     引き分けを除いた勝敗数に対する片側二項検定（符号検定）と、
     引き分けを 0.5 と数えたスコア率の Wilson 信頼区間の両方を出す。
"""

import copy
import json
import math
import os
import random
import time

import numpy as np
import torch

from AlphaZeroCode import Node, Util
from AlphaZeroCode.Agent import Agent


# =====================================================================
# 統計
# =====================================================================
def wilson_interval(score, n, z=1.96):
    """スコア率の Wilson 信頼区間。score は引き分けを 0.5 と数えた合計。"""
    if n <= 0:
        return (0.0, 0.0)
    p = score / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def binomial_p_value(wins, losses):
    """H0: 勝ちと負けが五分。片側 p 値 P(X >= wins), X~Bin(wins+losses, 0.5)。"""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(wins, n + 1))
    return tail / (2.0 ** n)


def elo_diff(score_rate):
    """スコア率から Elo 差を推定。"""
    s = min(max(score_rate, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


# =====================================================================
# 対局
# =====================================================================
def _random_opening(width, n_plies, rng):
    """空の盤に n_plies 手だけランダムに置いた初形（手のリスト）を返す。"""
    cells = list(range(width * width))
    rng.shuffle(cells)
    return [int(a) for a in cells[:n_plies]]


def _build_node(CFG, env, util, opening):
    """オープニングを env に適用し、履歴付きの Node を作る。"""
    env.reset()
    node = Node(CFG, env.state)          # states[0] = 空盤, 残りはゼロ埋め
    states = node.states
    actions = node.actions
    player = CFG.first_player

    for a in opening:
        states = util.get_next_states(states, a, player, env)   # step の前に呼ぶ
        actions = util.get_next_actions(actions, a)
        env.step(a)
        player = -player

    node = Node(CFG)
    node.states = states
    node.actions = actions
    node.player = player
    return node


def _advance(node, action, mover, CFG, util, env):
    """
    相手が打った後のノードを作る。

    Util.get_next_node() は子ノードが無いとき
      ・手番を反転しない
      ・置く石の色に -node.player を使う
    ため、「node.player = 打った側」の状況で呼ぶと盤面が壊れる。
    ここでは打った側 (mover) を明示して、必ず正しい色・手番で遷移させる。
    """
    for child in node.child_nodes:
        if int(child.action) == int(action):
            child.player = -mover
            return child

    nxt = Node(CFG)
    nxt.states = util.get_next_states(node.states, action, mover, env)
    nxt.actions = util.get_next_actions(node.actions, action)
    nxt.action = action
    nxt.player = -mover
    return nxt


def play_one_game(env, model, CFG, engine, az_player, opening, util=None,
                  max_moves=None, record=False):
    """
    1局打つ。az_player は AZ が持つ手番 (CFG.first_player / second_player)。
    戻り値: dict(result, winner, moves, board, history)
    """
    util = util or Util(CFG)
    agent = Agent(env, model, CFG, train=False)

    node = _build_node(CFG, env, util, opening)
    max_moves = max_moves or CFG.action_size + 1
    history = list(opening)
    moves = len(opening)
    while not env.done and moves < max_moves:
        if env.player != node.player:
            raise RuntimeError(f'node と env の手番がずれています '
                               f'(env={env.player}, node={node.player})')
        mover = env.player

        if mover == az_player:
            node = agent.alpha_zero(node)
            action = int(node.action)
            env.step(action)
        else:
            action = int(engine.select(env.state, mover))
            env.step(action)
            if not env.done:
                node = _advance(node, action, mover, CFG, util, env)

        history.append(action)
        moves += 1

    if env.winner == az_player:
        result = 'win'
    elif env.winner == 0:
        result = 'draw'
    else:
        result = 'lose'

    out = {'result': result, 'winner': int(env.winner), 'moves': moves,
           'az_player': int(az_player)}
    if record:
        out['history'] = history
        out['board'] = copy.deepcopy(env.state)
    return out


def play_match(env, model, CFG, engine, n_pairs=25, seed=0, opening_plies=2,
               verbose=True, record_games=2, progress_every=5):
    """
    ペア方式のマッチ。1ペア = 同一オープニングで AZ 先手 / AZ 後手 の2局。
    戻り値: 集計 dict
    """
    rng = random.Random(seed)
    util = Util(CFG)

    # --- 評価中はディリクレノイズを切る ---
    saved_eps = CFG.Dirichlet_epsilon
    CFG.Dirichlet_epsilon = 0.0
    torch.manual_seed(seed)
    np.random.seed(seed)

    tally = {'win': 0, 'draw': 0, 'lose': 0}
    by_side = {'first': {'win': 0, 'draw': 0, 'lose': 0},
               'second': {'win': 0, 'draw': 0, 'lose': 0}}
    games, samples = [], []
    t0 = time.time()

    try:
        for pair in range(n_pairs):
            opening = _random_opening(CFG.board_width, opening_plies, rng)

            for side in ('first', 'second'):
                az_player = CFG.first_player if side == 'first' else CFG.second_player
                rec = len(samples) < record_games
                g = play_one_game(env, model, CFG, engine, az_player, opening,
                                  util=util, record=rec)
                tally[g['result']] += 1
                by_side[side][g['result']] += 1
                games.append({'pair': pair, 'side': side, 'result': g['result'],
                              'moves': g['moves'], 'opening': opening})
                if rec:
                    samples.append(g)

            if verbose and (pair + 1) % progress_every == 0:
                n = (pair + 1) * 2
                sc = (tally['win'] + 0.5 * tally['draw']) / n
                print(f'    pair {pair+1:3d}/{n_pairs}  '
                      f'w{tally["win"]} d{tally["draw"]} l{tally["lose"]}  '
                      f'score={sc:.1%}  ({time.time()-t0:.0f}s)', flush=True)
    finally:
        CFG.Dirichlet_epsilon = saved_eps

    n = max(1, len(games))
    score = tally['win'] + 0.5 * tally['draw']
    lo, hi = wilson_interval(score, n)

    return {
        'engine': getattr(engine, 'name', str(engine)),
        'games': n,
        'win': tally['win'], 'draw': tally['draw'], 'lose': tally['lose'],
        'win_rate': tally['win'] / n,
        'score_rate': score / n,
        'ci95': [lo, hi],
        'p_value': binomial_p_value(tally['win'], tally['lose']),
        'elo': elo_diff(score / n),
        'as_first': by_side['first'],
        'as_second': by_side['second'],
        'seconds': round(time.time() - t0, 1),
        'sample_games': samples,
        'all_games': games,
    }


# =====================================================================
# 表示
# =====================================================================
def board_to_text(state, first='X', second='O', empty='.'):
    w = len(state)
    out = ['   ' + ' '.join(f'{i}' for i in range(w))]
    for r in range(w):
        line = []
        for c in range(w):
            v = state[r][c]
            line.append(first if v == -1 else second if v == 1 else empty)
        out.append(f'{r}  ' + ' '.join(line))
    return '\n'.join(out)


def format_result(res, alpha=0.05):
    lo, hi = res['ci95']
    verdict = ('AZ の方が強い（統計的に有意）' if res['p_value'] < alpha and res['score_rate'] > 0.5
               else '有意差なし / AZ が上回っていない')
    return (
        f"  対戦相手 : {res['engine']}\n"
        f"  対局数   : {res['games']}  (先手 {sum(res['as_first'].values())} / "
        f"後手 {sum(res['as_second'].values())})\n"
        f"  勝敗     : {res['win']}勝 {res['draw']}分 {res['lose']}敗\n"
        f"  スコア率 : {res['score_rate']:.1%}  95%CI [{lo:.1%}, {hi:.1%}]\n"
        f"  先手番   : {res['as_first']['win']}勝 {res['as_first']['draw']}分 "
        f"{res['as_first']['lose']}敗\n"
        f"  後手番   : {res['as_second']['win']}勝 {res['as_second']['draw']}分 "
        f"{res['as_second']['lose']}敗\n"
        f"  片側二項検定 p = {res['p_value']:.3g}\n"
        f"  推定 Elo 差 : {res['elo']:+.0f}\n"
        f"  判定     : {verdict}\n"
        f"  所要時間 : {res['seconds']}s"
    )


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
    return path
