#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @title MCTS
""" Import libraries Original """
import copy
from math import sqrt
import numpy as np
import torch
from . Util import Util
from . Node import Node

class MCTS():
    """
    root node: 探索開始ノード
    """
    def __init__(self, env, model, CFG, train=True):
        self.env = copy.deepcopy(env)
        self.model = model
        self.player = None
        self.train = train
        self.CFG = CFG
        self.util = Util(CFG)

    def __call__(self, node, play_count=1):

        self.model.eval()
        self.player = node.player # Important!        
        root_node = copy.deepcopy(node)
        root_node.is_root = True

        """ シミュレーション """
        for i in range(self.CFG.num_simulation): # AlphaGo Zero 1600 sim / AlphaZero 800 sim
            self.env.reset()
            self.env.state = copy.deepcopy(root_node.states[0])
            self.env.player = root_node.player  # resetされたので、Self play でのプレーヤーに再設定

            """ ルートノードから再帰的に探索を実行 """
            self.search(root_node)

        node.input_features = root_node.input_features # Copy from simulated node. Necessary for dataset.

        if len(root_node.child_nodes) > 0:
            next_node = self.play(root_node, play_count)
        else:
            """ 打つ手がない場合、返却ノードにパスを設定 """
            next_node = root_node
            next_node.player = -root_node.player
            next_node.action = self.CFG.pass_

        """ 訪問回数で算出した方策を現在のノードに格納 """
        node.child_nodes = root_node.child_nodes

        return next_node

    def search(self, node, done=False, reward=0):

        """ ゲームオーバー """
        if done:
            v = reward
            self.backup(node, v)
            return v

        """ リーフ """
        if len(node.child_nodes) == 0:
            v = self.expand(node)
            self.backup(node, v)
            return v

        """ 選択 """
        next_node = self.select(node)
        _, reward, done = self.env.step(next_node.action)

        """ 探索（相手の手番で） """
        v = -self.search(next_node, done, -reward)

        """ バックアップ """ 
        self.backup(node, v) 

        return v

    def select(self, node):
        """ 選択
        Ｑ（相手にとっては－Ｑ）＋Ｕの最大値から、最良の行動を選ぶ
        """
        pucts = [] # PUCTの値
        cpuct = self.CFG.cpuct # 1-6
        s = node.n # Σ_b (N(s,b)) と同じこと
        child_nodes = self.util.get_child_nodes(node) # エッジの取得

        if node.is_root:
            """ 事前確率にディリクレノイズを追加 """
            child_nodes = self.add_dirichlet_noise(child_nodes)

        for child_node in child_nodes:
            p = child_node.p
            n = child_node.n
            Q = child_node.Q
            U = cpuct * p * sqrt(s) / (1 + n)
            pucts.append(-Q + U)
  
        max_index = np.argmax(pucts)
        next_node = node.child_nodes[max_index]

        return next_node


    """ 展開と評価 """
    def expand(self, node):

        """ 入力特徴の作成 """
        features = self.util.state2feature(node)

        """ 推論 """
        p, v = self.model(features)

        """ バッチの次元を削除 """
        p = p[0].tolist()
        v = v[0].tolist()[0] # スカラーに変換

        """ 子ノードの生成 """
        self.add_child_nodes(node, p)

        return v 

    def backup(self, node, v):
        """ バックアップ """
        node.n += 1
        node.w += v
        node.Q = node.w / node.n

    def play(self, node, play_count):
        """ 実行

        探索が完了すると、 N^(1/τ)に比例した探索確率πで行動を決定
        Nはルート状態からの各ノードへの訪問回数、
        τは温度を制御するパラメータ。

        τ: 温度パラメーター
            -∞: 決定的に選択
            1  : 確率的に選択
            ∞ : ランダムに選択
        """

        """ 温度パラメーター """
        if self.train:
            """ 訓練時は最初のｎ手までは確率的に """
            tau = self.CFG.tau if play_count <= self.CFG.tau_limit else 0
        else:
            """ 評価時には決定的に """
            tau = 0
                    
        N = []
        for child_node in node.child_nodes:
            N.append(child_node.n)

        """ 探索(Exploration)か 経験の利用(Exploitation)か """
        if tau > 0:
            """ ソフトマックスで方策を出力 """
            N_pow = np.power(N, 1/tau)
            N_sum = np.sum(N_pow)
            pi = N_pow / N_sum

            """ 最善手から1%以上離れていない手をソフトマックスで出力 """
            M = np.where(pi < pi.max()-0.01, 0, pi)
            M_sum = np.sum(M)
            pi = M / M_sum

            """ 方策からサンプリング """
            p = np.random.choice(pi, p=pi)
            index = np.argwhere(pi==p)[0][0].tolist()

        else:
            """ 決定的に選択 """
            index = np.argmax(N)

        """ 次のノードへ遷移 """
        next_node = node.child_nodes[index]

        # パスの処理
        if hasattr(self.CFG, 'pass_'):
            if next_node.action == self.CFG.pass_:
                next_node.player = -node.player

        return next_node

    def add_dirichlet_noise(self, child_nodes):
        """
        ルートノードの事前確率にディリクレノイズを加えて、さらなる探索
        P(s, a) = (1 - ε) * p(a) + ε*η(a)
        where η～ Dir(0.03), ε= 0.25
        """
        e = self.CFG.Dirichlet_epsilon
        alpha = self.CFG.Dirichlet_alpha

        dirichlet_noise = np.random.dirichlet([alpha] * len(child_nodes))

        for i, child_node in enumerate(child_nodes):
            x = child_node.p
            p = (1-e) * x + e * dirichlet_noise[i]
            child_nodes[i].p = p

        return child_nodes

    """ 子ノードの生成 """
    def add_child_nodes(self, node, p):
        """ 合法手の取得 """
        legal_actions = self.env.get_legal_actions()

        for action in legal_actions:
            states = self.util.get_next_states(node.states, action, node.player, self.env)
            actions = self.util.get_next_actions(node.actions, action)

            child_node = Node(self.CFG)
            child_node.p = p[action]
            child_node.action = action
            child_node.actions = actions
            child_node.states = states
            child_node.player = -node.player
            node.child_nodes.append(child_node)

        if hasattr(self.CFG, 'pass_'):
            # Passのノードを追加
            action = self.CFG.pass_
            states = copy.deepcopy(node.states)
            actions = self.util.get_next_actions(node.actions, action)
            child_node = Node(self.CFG)

            child_node.p = p[action]
            child_node.action = action
            child_node.actions = actions
            child_node.states = states
            child_node.player = -node.player
            node.child_nodes.append(child_node)
                
