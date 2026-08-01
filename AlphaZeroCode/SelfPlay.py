#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @title SelfPlay

""" SelfPlay """
import copy
from . Util import Util
from . MCTS import Node
from . Agent import Agent

class SelfPlay():
    """ 経験を収集する自己対局クラス """
    def __init__(self, CFG, env, model):
        self.env = env
        self.model = model
        self.CFG = CFG
        self.util = Util(CFG)
        
        self.dataset = []
        self.agent = Agent(env, model, CFG, train=True)


    def __call__(self):
        """ Self playのループ処理  (AlphaGo Zero: 250,000 回) """
        state = self.env.reset()
        node = Node(self.CFG, state)
        self.play(node)

        """ 蓄積した経験データのセットを最大サイズで切り捨て """
        self.dataset = self.dataset[-self.CFG.max_dataset_size:]

        return self.dataset

    def play(self, node, play_count=1):
        """ 探索の実行 """
        self.util.indicator(play_count)

        """ AlphaZero player """
        next_node = self.agent.alpha_zero(node, play_count)

        action = next_node.action

        """ ゲームの実行 """
        _next_state, reward, done = self.env.step(action)

        if done:
            # 引き分けなら  0
            # 勝った時は +1 
            v = reward

        else:
            """ 再帰的に自己対局 """
            v = -self.play(next_node, play_count + 1)

        """ 履歴データを追加 """
        self.backup(node, action, v)

        """ 符号を反転させて返却 """
        return v

    # Add to dataset
    def backup(self, node, action, v):

        """ 履歴データの設定 """
        states = copy.deepcopy(node.states)
        input_features = copy.deepcopy(node.input_features).tolist()[0]
        
        # pi = one_hot_encording(node)     # 0 0 0 1 
        pi = self.util.probability_distribution(node) # 0.1 0.2 0.3 0.4 

        plain = { # for debug
                 'state': states[0], 
                 'pi': pi,
                 'z': v,        
                 'states': states,
                 'player': node.player,
                 'action': action,
                 }

        """ データセットに経験データを追加 """
        data = []
        data.append(input_features)
        data.append(pi)
        data.append([v])
        data.append(plain)

        self.dataset.append(data)
