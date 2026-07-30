#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_gomoku_7x7.py
===================
Michi-123/AlphaZeroCode を使った 7x7 五目並べ(5目) の訓練スクリプト。

  python train_gomoku_7x7.py --preset balanced
  python train_gomoku_7x7.py --preset fast --iterations 5   # 動作確認
  python train_gomoku_7x7.py --preset balanced --resume     # 続きから

前提: AlphaZeroCode パッケージが import できること。
      env/Gomoku.py が無い場合は同ディレクトリの Gomoku.py を自動で使います。
"""

import argparse
import contextlib
import copy
import io
import os
import sys
import time

import numpy as np
import torch

sys.path.append('.')
sys.path.append('..')

from AlphaZeroCode import SelfPlay, Util
from AlphaZeroCode.network.AlphaZeroNetwork import AlphaZeroNetwork

from CFG_7x7 import get_cfg, PRESETS
from az_common import make_env, TrainFixed, evaluate_both_sides, tactical_test
from bootstrap_7x7 import generate_bootstrap_dataset

# 戦術テストは毎回同じ hold-out 局面で測る（seed を変えると比較できない）
TACTIC_SEED = 999


# ---------------------------------------------------------------------
def build_model(CFG):
    model = AlphaZeroNetwork(CFG).to(CFG.device)
    n = sum(p.numel() for p in model.parameters())
    print(f'model parameters: {n/1e6:.2f}M ({n:,})')
    return model


def current_lr(CFG, iteration):
    lr = CFG.learning_rate
    for start in sorted(CFG.lr_schedule):
        if iteration >= start:
            lr = CFG.lr_schedule[start]
    return lr


def write_log_header(CFG):
    if not os.path.exists(CFG.log_path):
        with open(CFG.log_path, 'w') as f:
            f.write('iteration,lr,dataset,policy_loss,value_loss,'
                    'win_rate,score_rate,tactic_acc,selfplay_sec,train_sec\n')


def append_log(CFG, row):
    with open(CFG.log_path, 'a') as f:
        f.write(','.join(str(x) for x in row) + '\n')


# ---------------------------------------------------------------------
def run_selfplay(self_play, n_games, quiet=True):
    """SelfPlay.__call__ は 1 局しか打たないので、ここで n 局まわす。"""
    dataset = self_play.dataset
    for _ in range(n_games):
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                dataset = self_play()
        else:
            dataset = self_play()
    return dataset


def train_loop(CFG, args):
    print('=' * 78)
    print('AlphaZero  7x7 Gomoku (5 in a row)')
    print(CFG.summary())
    print('=' * 78)

    env = make_env(CFG)
    model = build_model(CFG)
    util = Util(CFG)

    start_iteration = 1
    if args.resume and os.path.exists(CFG.model_path):
        start_iteration = util.load_model(model)

    trainer = TrainFixed(model, CFG)
    self_play = SelfPlay(CFG, env, model)

    # ---- 既存データセットの読み込み ----
    if args.resume and os.path.exists(CFG.dataset_path):
        util.load_dataset(self_play)

    # ---- ブートストラップ ----
    if CFG.use_bootstrap and start_iteration == 1:
        print('\n--- bootstrap phase ---')
        boot = generate_bootstrap_dataset(CFG, CFG.bootstrap_samples)
        trainer.set_lr(current_lr(CFG, 0))
        for k in range(args.bootstrap_epochs):
            pol, val = trainer(boot, quiet=True)
            acc = tactical_test(env, model, CFG, 200, seed=TACTIC_SEED)['accuracy']
            print(f'  boot round {k+1}/{args.bootstrap_epochs}  '
                  f'pi_loss={pol:.4f} v_loss={val:.4f}  tactic={acc:.1%}')
        # 自己対局データにも少し混ぜて、序盤の忘却を防ぐ
        self_play.dataset = boot[: CFG.max_dataset_size // 4] + self_play.dataset

    write_log_header(CFG)

    # ---- メインループ ----
    best_score = -1.0
    for iteration in range(start_iteration, args.iterations + 1):
        t0 = time.time()

        lr = current_lr(CFG, iteration)
        trainer.set_lr(lr)

        # 1) 自己対局
        dataset = run_selfplay(self_play, CFG.num_selfplay_games,
                               quiet=not args.verbose)
        t_sp = time.time() - t0

        # 2) 学習
        t1 = time.time()
        pol, val = trainer(dataset, quiet=not args.verbose)
        t_tr = time.time() - t1

        # 3) 保存
        util.save(model, iteration)
        util.save_iteration_counter(CFG.iteration_counter_path, iteration)
        if iteration % CFG.make_check_point_frequency == 0:
            util.save_dataset(CFG.dataset_path, dataset)

        # 4) 評価
        win_rate = score_rate = acc = ''
        if iteration % CFG.eval_frequency == 0 or iteration == args.iterations:
            res = evaluate_both_sides(env, model, CFG, CFG.eval_games, seed=iteration)
            tac = tactical_test(env, model, CFG, 200, seed=TACTIC_SEED)
            win_rate, score_rate, acc = res['win_rate'], res['score_rate'], tac['accuracy']
            print(f'\n  [eval] vs random  win={res["win"]} draw={res["draw"]} '
                  f'lose={res["lose"]}  score={score_rate:.1%} '
                  f'(first {res["first"]:.0%} / second {res["second"]:.0%})  '
                  f'tactic={acc:.1%}')
            if score_rate > best_score:
                best_score = score_rate
                torch.save(model.state_dict(),
                           CFG.model_path.replace('.pt', '_best.pt'))
                print(f'  [eval] new best -> {CFG.model_path.replace(".pt", "_best.pt")}')

        print(f'iter {iteration:3d}/{args.iterations}  lr={lr:.4f}  '
              f'data={len(dataset):6d}  pi_loss={pol:.4f}  v_loss={val:.4f}  '
              f'selfplay={t_sp:.0f}s train={t_tr:.0f}s')

        append_log(CFG, [iteration, lr, len(dataset), round(pol, 5),
                         round(val, 5), win_rate, score_rate, acc,
                         round(t_sp, 1), round(t_tr, 1)])

    print('\ntraining finished. log ->', CFG.log_path)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='balanced', choices=list(PRESETS))
    ap.add_argument('--iterations', type=int, default=None)
    ap.add_argument('--sims', type=int, default=None)
    ap.add_argument('--games', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--work-dir', default='.')
    ap.add_argument('--bootstrap-epochs', type=int, default=6)
    ap.add_argument('--no-bootstrap', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    overrides = {'work_dir': args.work_dir}
    if args.sims:
        overrides['num_simulation'] = args.sims
    if args.games:
        overrides['num_selfplay_games'] = args.games
    if args.device:
        overrides['device'] = args.device
    if args.no_bootstrap:
        overrides['use_bootstrap'] = False

    CFG = get_cfg(args.preset, **overrides)
    if args.iterations is None:
        args.iterations = CFG.num_iteration

    torch.manual_seed(0)
    np.random.seed(0)

    train_loop(CFG, args)


if __name__ == '__main__':
    main()
