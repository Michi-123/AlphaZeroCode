#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @title Gomoku

""" 五目並べ (n 目並べ) の環境

TicTacToe と同じインターフェース（reset / step / get_legal_actions）を持ち、
盤サイズ width と勝利条件 row を可変にしたもの。

【報酬の符号について】
    step() が返す reward は「**着手後に手番となるプレーヤーから見た価値**」です。

        直前の着手で勝ちが決まった  ->  reward = -1  (次の手番の人は負け)
        引き分け / 続行             ->  reward =  0

    AlphaZeroCode 本体がこの符号を前提にしています。

      * SelfPlay.play():  v = -reward
            着手したノード側から見た教師信号 z が +1 になる。
      * MCTS.search():    v = reward を next_node に backup し、
            親には -v を backup する。
            どのノードも「そのノード自身の手番から見た価値」で
            w / Q が積み上がり、NN の価値 (expand) と符号が一致する。

    勝者を直接知りたい場合は self.winner (勝ったプレーヤー / 引き分けは 0)
    を参照してください。
"""

import copy

import numpy as np


class Gomoku():

    """ 8方向のうち、重複しない4方向 """
    DIRECTIONS = ((0, 1), (1, 0), (1, 1), (-1, 1))

    def __init__(self, row=5, width=7):
        self.row = row            # 何目並べか
        self.width = width        # 盤の一辺
        self.action_size = width * width
        self.reset()

    def reset(self):
        self.state = [[0 for _ in range(self.width)] for _ in range(self.width)]
        self.done = False
        self.player = -1          # CFG.first_player と揃えること
        self.reward = 0
        self.winner = 0
        self.last_action = None
        return self.state

    def step(self, a):
        a = int(a)
        x1, x2 = a // self.width, a % self.width

        self.state[x1][x2] = self.player
        self.last_action = a

        if self._is_done(a):
            self.done = True
            self.winner = self.player
            """ 着手した側の勝ち = 次の手番から見れば -1 """
            self.reward = -1

        elif self._is_draw():
            self.done = True
            self.winner = 0
            self.reward = 0

        else:
            self.reward = 0

        """ 手番の交代 """
        self.player = -self.player

        return self.state, self.reward, self.done

    def get_legal_actions(self, state=None):
        s = self.state if state is None else state
        s = np.array(s, dtype=np.float32).reshape(-1)
        return np.where(s == 0)[0]

    def clone(self):
        return copy.deepcopy(self)

    # -----------------------------------------------------------------
    def _is_done(self, a):
        """ 直前の着手 a を含む row 連ができたか """
        x1, x2 = a // self.width, a % self.width
        player = self.state[x1][x2]

        for dr, dc in self.DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = x1 + dr * sign, x2 + dc * sign
                while 0 <= r < self.width and 0 <= c < self.width \
                        and self.state[r][c] == player:
                    count += 1
                    r += dr * sign
                    c += dc * sign

            if count >= self.row:
                return True

        return False

    def _is_draw(self):
        """ 空きマスが無くなったら引き分け """
        for row in self.state:
            for piece in row:
                if piece == 0:
                    return False
        return True

    # -----------------------------------------------------------------
    def render(self):
        print('  ' + ' '.join(str(i) for i in range(self.width)))
        for i, row in enumerate(self.state):
            line = []
            for piece in row:
                line.append('X' if piece == -1 else 'O' if piece == 1 else '-')
            print(str(i) + ' ' + ' '.join(line))
        print()
