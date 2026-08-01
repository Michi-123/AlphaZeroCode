#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @title Evaluate
import numpy as np
from tqdm import tqdm
import copy

from .Agent import Agent
from .Util import Util
from .MCTS import MCTS, Node


class Evaluate:
    def __init__(self, env, model, CFG, render_mode=0):
        self.env = env
        self.model = model
        self.CFG = CFG
        self.util = Util(CFG)
        self.render_mode = render_mode

    def play_AZ_vs_AZ(self, model1, model2, play_count=1):
        print('AlphaZero vs AlphaZero')

        env = self.env

        player1 = Agent(env, model1, self.CFG, train=False)
        player2 = Agent(env, model2, self.CFG, train=False)

        win_model1 = 0
        win_model2 = 0

        for count in (range(play_count)):

            state = env.reset()
            #env.render()

            node1 = Node(self.CFG, state)
            node2 = Node(self.CFG, state)

            node1.player = self.CFG.first_player
            node2.player = self.CFG.second_player

            action_count = 0

            while True:
                """ AlphaZero 1 turn """
                action_count += 1

                legal_actions = self.env.get_legal_actions()
                if len(legal_actions) > 0:
                    node1 = player1.alpha_zero(node1)
                    action = node1.action
                else:
                    action = self.CFG.pass_
                state, reward, done = env.step(action)

                self.util.show_board(state, self.render_mode)

                if done:
                    win_model1 += abs(reward)
                    # win_model1 += reward
                    break

                node2 = self.util.get_next_node(node2, action, env)

                """ AlphaZero 2 turn """
                action_count += 1

                legal_actions = self.env.get_legal_actions()
                if len(legal_actions) > 0:
                    node2 = player2.alpha_zero(node2)
                    action = node2.action
                else:
                    action = self.CFG.pass_

                state, reward, done = env.step(action)

                self.util.show_board(state, self.render_mode)

                if done:
                    win_model2 += abs(reward)
                    # win_model2 += reward
                    break

                node1 = self.util.get_next_node(node1, action, env)

            print("count model1 model2", action_count, win_model1, win_model2)
        print()

    def play_human_vs_AZ(self):

        print('Human vs AlphaZero')

        env = self.env

        win_human = 0
        win_alpha_zero = 0

        player = Agent(env, self.model, self.CFG, train=False)

        """ Initialize  """
        state = env.reset()
        node = Node(self.CFG, state)
        node.player = self.CFG.second_player  # 後手
        self.util.show_board(state, self.render_mode)

        while True:

            """ Human turn """
            action = player.human(node.states[0])
            state, reward, done = env.step(action)

            self.util.show_board(state, self.render_mode)

            if done:
                win_human += reward
                break

            # ここは、player_human に入れるか
            node = self.util.get_next_node(node, action, env=env)

            """ AlphaZero turn """
            legal_actions = self.env.get_legal_actions()

            if len(legal_actions) > 0:
                node = player.alpha_zero(node)
                action = node.action

            else:
                action = self.CFG.pass_

            state, reward, done = env.step(action)

            self.util.show_board(state, self.render_mode)

            if done:
                win_alpha_zero += reward
                break

        print("AlphaZero, Human", win_alpha_zero, win_human)

    def play_AZ_vs_human(self):

        env = self.env

        print('AlphaZero vs Human')

        win_human = 0
        win_alpha_zero = 0

        player = Agent(env, self.model, self.CFG, train=False)

        """ Initialize  """
        state = env.reset()
        node = Node(self.CFG, state)
        node.player = self.CFG.first_player  # 先手

        self.util.show_board(state, self.render_mode)

        while True:

            """ AlphaZero turn """
            legal_actions = self.env.get_legal_actions()

            if len(legal_actions) > 0:
                node = player.alpha_zero(node)
                action = node.action
            else:
                action = self.CFG.pass_

            state, reward, done = env.step(action)

            self.util.show_board(state, self.render_mode)

            if done:
                win_alpha_zero += reward
                break

            """ Human turn """
            action = player.human(env.state)
            state, reward, done = env.step(action)

            self.util.show_board(state, self.render_mode)

            if done:
                win_human += reward
                break

            node = self.util.get_next_node(node, action, env)

        print("AlphaZero, Human", win_alpha_zero, win_human)

    def play_random_vs_AZ(self, play_count=1, show_board=None):
        env = self.env

        print('Random vs AlphaZero')
        win_random = 0
        win_alpha_zero = 0

        player = Agent(env, self.model, self.CFG, train=False)

        for count in (range(play_count)):

            state = env.reset()
            node = Node(self.CFG, state)
            node.player = self.CFG.second_player

            while True:
                """ Random turn """
                action = player.random(state)
                state, reward, done = env.step(action)

                self.util.show_board(state, self.render_mode)

                if done:
                    win_random += abs(reward)
                    break

                node = self.util.get_next_node(node, action, env)

                """ AlphaZero turn """
                legal_actions = self.env.get_legal_actions()

                if len(legal_actions) > 0:
                    node = player.alpha_zero(node)
                    action = node.action
                else:
                    action = self.CFG.pass_

                state, reward, done = env.step(action)

                self.util.show_board(state, self.render_mode)

                if done:
                    win_alpha_zero += abs(reward)
                    break

            print("Random , AlphaZero ", win_random, win_alpha_zero)
        print()

    def play_AZ_vs_random(self, play_count=1, show_board=None):

        print('AlphaZero vs Random')
        win_random = 0
        win_alpha_zero = 0

        env = self.env

        player = Agent(env, self.model, self.CFG, train=False)

        for count in (range(play_count)):

            state = env.reset()
            node = Node(self.CFG, state)
            node.player = self.CFG.first_player

            while True:

                """ AlphaZero turn """
                legal_actions = self.env.get_legal_actions()

                if len(legal_actions) > 0:
                    node = player.alpha_zero(node)
                    action = node.action
                else:
                    action = self.CFG.pass_

                state, reward, done = env.step(action)

                self.util.show_board(state, self.render_mode)

                if done:
                    win_alpha_zero += abs(reward)
                    break

                """ Random turn """
                action = player.random(state)
                state, reward, done = env.step(action)

                self.util.show_board(state, self.render_mode)

                if done:
                    win_random += abs(reward)
                    break

                node = self.util.get_next_node(node, action, env)

            print("AlphaZero , Random ", win_alpha_zero, win_random)
        print()
