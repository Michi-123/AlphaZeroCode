#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bootstrap_7x7.py
================
自己対局の前に「5目並べのルール」を教えるためのデータを作る。

7x7 / 5目のランダム対局は 92.5% が決着するので 5x5 ほど深刻ではないが、
それでも「勝ち手を最後まで打ち切る」「相手のリーチを止める」の2点を
先に教えておくと、序盤のイテレーションが劇的に速くなる。

生成するのは次の2種類だけ（自信のあるラベルしか作らない）:

  A. 即勝ち局面   : pi = 勝ち手に集中,  z = +1.0
  B. 要防御局面   : pi = 防御手に集中,  z = 0.0（相手のリーチが2本なら -1.0）

pi はラベルスムージング付き（完全な one-hot は学習を殺す）。
"""

import copy
import random

import numpy as np

from AlphaZeroCode import Node, Util
from az_common import make_env, winning_moves


def generate_bootstrap_dataset(CFG, n_samples=4000, smoothing=0.1,
                               teacher_ratio=0.5, seed=0, verbose=True):
    """
    teacher_ratio: 1局まるごと「勝ち手/防御手を必ず打つ先生」で進める割合。
        残りはランダム着手で進める。ランダム局の方が「即勝ち局面」が多く出るため、
        両方を混ぜて win / block サンプル数のバランスを取る。
    """
    rng = random.Random(seed)
    util = Util(CFG)
    env = make_env(CFG)
    width = CFG.board_width
    row = CFG.row
    A = CFG.action_size

    quota = n_samples // 2
    buckets = {'win': [], 'block': []}
    games = 0
    n_double = 0
    guard = n_samples * 100

    def add(label, states, player, targets, legal, z):
        node = Node(CFG, states[0])
        node.states = copy.deepcopy(states)
        node.player = player
        util.state2feature(node)
        features = copy.deepcopy(node.input_features).tolist()[0]

        pi = np.zeros(A, dtype=np.float64)
        for a in targets:
            pi[a] = 1.0 / len(targets)
        uni = np.zeros(A, dtype=np.float64)
        for a in legal:
            uni[a] = 1.0 / len(legal)
        pi = (1.0 - smoothing) * pi + smoothing * uni

        plain = {
            'state': copy.deepcopy(states[0]),
            'pi': pi.tolist(),
            'z': z,
            'states': copy.deepcopy(states),
            'player': player,
            'action': int(targets[0]),
            'source': 'bootstrap_' + label,
        }
        buckets[label].append([features, pi.tolist(), [z], plain])

    while (len(buckets['win']) < quota or len(buckets['block']) < quota) and guard > 0:
        guard -= 1
        games += 1
        state = env.reset()

        node = Node(CFG, state)
        states = copy.deepcopy(node.states)
        player = CFG.first_player
        use_teacher = rng.random() < teacher_ratio

        for _ in range(A):
            legal = [int(a) for a in env.get_legal_actions()]
            if not legal:
                break

            wins = winning_moves(env.state, player, row, width)
            threats = winning_moves(env.state, -player, row, width)

            action = None
            if wins:
                if len(buckets['win']) < quota:
                    add('win', states, player, wins, legal, 1.0)
                if use_teacher:
                    action = int(rng.choice(wins))
            elif threats:
                z = -1.0 if len(threats) >= 2 else 0.0
                if len(threats) >= 2:
                    n_double += 1
                if len(buckets['block']) < quota:
                    add('block', states, player, threats, legal, z)
                if use_teacher:
                    action = int(rng.choice(threats))

            if action is None:
                action = int(rng.choice(legal))

            states = util.get_next_states(states, action, player, env)
            _, _, done = env.step(action)
            player = -player
            if done:
                break

    dataset = buckets['win'] + buckets['block']
    rng.shuffle(dataset)
    dataset = dataset[:n_samples]

    if verbose:
        print(f'[bootstrap] {len(dataset)} samples / {games} games  '
              f'(win={len(buckets["win"])}, block={len(buckets["block"])}, '
              f'double-threat={n_double})')
    return dataset


def main():
    import argparse
    import sys
    from CFG_7x7 import get_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='balanced')
    ap.add_argument('--samples', type=int, default=4000)
    args = ap.parse_args()

    CFG = get_cfg(args.preset)
    ds = generate_bootstrap_dataset(CFG, args.samples)

    util = Util(CFG)
    util.save_dataset(CFG.bootstrap_path, ds)
    print('saved ->', CFG.bootstrap_path)


if __name__ == '__main__':
    main()
