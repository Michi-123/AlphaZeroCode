#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @title Agent

import random
import copy
from .Util import Util
from .MCTS import MCTS


class Agent:

    def __init__(self, env, model, CFG, train=True):
        self.CFG = CFG # SemiSelfPlayでAgentをoverrideした時に必要
        self.env = env

        """ MCTSをインスタンス化 """
        self.mcts = MCTS(env, model, CFG, train)
        self.util = Util(CFG)

    def alpha_zero(self, node, play_count=1):
        next_node = self.mcts(node, play_count)
        return next_node

    def human(self, state):
        self.util.show_legal_actions(self.env)

        while True:
            x = input('縦, 横 ')
            try:
                x1, x2 = x.split(',')
                x1, x2 = int(x1), int(x2)
                if state[x1][x2] == 0:
                    break
            except:
                pass

        action = x1 * self.env.width + x2

        return action

    def random(self, state):
        legal_actions = self.env.get_legal_actions()
        action = random.choice(legal_actions)
        return action

    """ 
    Minimaxプレーヤーで最善の手を取得 
    """

    def player_minimax(self, env, player):

        """ 
        Minimax アルゴリズム 
        """

        def minimax(env, player):

            """ ゲーム終了の場合 """
            if env.done:
                # print('reward', env.reward)
                return env.reward  # 報酬: -1, 0, 1

            """ 合法手の取得 """
            legal_actions = env.get_legal_actions(env.state)

            """
            先手にとっては最大となる価値（報酬）を選ばせ、
            後手にとっては最小となる価値（報酬）を選ばせる。
            """
            if player == 1:
                """ 先手プレーヤー """
                value = -float('inf')  # 状態価値に−∞を設定 float('inf')

                """ 合法手で最大値を取得 """
                for action in legal_actions:
                    child = copy.deepcopy(env)  # 環境のコピー
                    child.step(action)  # 実行
                    value = max(value, minimax(child, -player))  # 現在の価値とminimaxの価値のうち大きい方を取得
                # return value # 最大値を返却
                return value / 2

            else:
                """ 後手プレーヤー """
                value = float('inf')  # ∞

                """ 合法手で最小値を取得 """
                for action in legal_actions:
                    child = copy.deepcopy(env)
                    child.step(action)
                    value = min(value, minimax(child, -player))
                # return value # 最小値を返却
                return value / 2

        """ 最善手と状態価値を初期化 """
        best_action = 0  # 行動インデックスの初期化
        best_value = -float('inf')  # 最善の手を-∞で初期化

        """ 合法手の取得 """
        legal_actions = env.get_legal_actions(env.state)
        # print('legal_actions', legal_actions)

        """ 合法手でループ """
        for action in legal_actions:

            child = copy.deepcopy(env)  # 環境のコピー
            child.step(action)  # 実行

            """ 
            mini-maxアルゴリズムで探索して、状態価値を取得。
            後手の価値を先手にとってはマイナスの価値とする。
            """
            value = -minimax(child, -player)

            """ 取得した価値が最善の場合よりも大きい場合は値を交換 """
            if value > best_value:
                best_value = value  # 取得した価値を最善の価値に設定
                best_action = action  # その場合の行動を最善の行動とする

            # print('action: ', action ,value)

        # print('best_action', best_action)
        return best_action  # 最善の行動インデックスを返却

    """
    Alpha-betaプレーヤーで最善の手を取得 
    """

    def player_alphabeta(self, env, player):

        """ 
        Alpha–beta pruning
        """

        def alphabeta(env, α, β, player):

            """ ゲーム終了の場合 """
            if env.done:
                # print('reward', env.reward)
                return env.reward  # 報酬: -1, 0, 1

            """ 合法手の取得 """
            legal_actions = env.get_legal_actions(env.state)

            """
            先手にとっては最大となる価値（報酬）を選ばせ、
            後手にとっては最小となる価値（報酬）を選ばせる。
            """
            if player == 1:

                value = -float('inf')  # 状態価値に−∞を設定 float('inf')

                """ 合法手で最大値を取得 """
                for action in legal_actions:
                    child = copy.deepcopy(env)  # 環境のコピー
                    child.step(action)  # コピーで実行
                    value = max(value, alphabeta(child, α, β, -player))  # 現在の価値とalphabetaの価値のうち大きい方を取得
                    if value >= β:
                        # print('β cutoff')
                        break
                    α = max(α, value)  # 最大値を返却

                # return value 
                return value / 2

            else:

                value = float('inf')  # ∞

                for action in legal_actions:
                    child = copy.deepcopy(env)  # 環境のコピー
                    child.step(action)  # コピーで実行
                    value = min(value, alphabeta(child, α, β, -player))
                    if value <= α:
                        # print('α cutoff')
                        break
                    β = min(β, value)
                # return value  
                return value / 2

        """ 最善手と状態価値を初期化 """
        best_action = 0
        best_value = -float('inf')

        """ αとβを初期化 """
        α = -float('inf')
        β = float('inf')

        """ 合法手でループ """
        legal_actions = env.get_legal_actions(env.state)
        # print('legal_actions', legal_actions)

        for action in legal_actions:

            child = copy.deepcopy(env)
            child.step(action)

            """ 
            Alpha-betaアルゴリズムで探索して、状態価値を取得。
            後手の価値を先手にとってはマイナスの価値とする。
            """
            value = -alphabeta(child, α, β, -player)

            """ 取得した価値が最善の場合よりも大きい場合は値を交換 """
            if value > best_value:
                best_value = value  # 取得した価値を最善の価値に設定
                best_action = action  # その場合の行動を最善の行動とする

            # print('action: ', action, value)

        # print('best_action', best_action)
        return best_action  # 最善の行動インデックスを返却
