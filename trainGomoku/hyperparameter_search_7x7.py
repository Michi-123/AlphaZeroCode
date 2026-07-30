#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hyperparameter_search_7x7.py
============================
7x7 五目並べ用のハイパーパラメータ探索。

本番の 100 イテレーションを回す前に、短いイテレーション数で候補設定を比べる。
評価指標は「vs ランダムの勝率」だけでなく、
学習の進み具合が早く出る「戦術正解率 (tactical accuracy)」も見る。

  # 候補設定を一括比較（既定 6 候補 × 8 イテレーション）
  python hyperparameter_search_7x7.py --mode compare --iterations 8

  # 1 設定だけ試す
  python hyperparameter_search_7x7.py --mode single --preset balanced --iterations 5

  # cpuct / sims のグリッド探索（モデルは fast 固定）
  python hyperparameter_search_7x7.py --mode grid --iterations 6

結果は results/hpsearch_*.csv に保存される。
"""

import argparse
import contextlib
import copy
import io
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append('.')
sys.path.append('..')

from AlphaZeroCode import SelfPlay, Util
from AlphaZeroCode.network.AlphaZeroNetwork import AlphaZeroNetwork

from CFG_7x7 import get_cfg
from az_common import make_env, TrainFixed, evaluate_both_sides, tactical_test
from bootstrap_7x7 import generate_bootstrap_dataset


# =====================================================================
# 候補設定
# =====================================================================
CANDIDATES = [
    # name,               overrides
    ('A_small_fastsim', dict(n_residual_block=3, resnet_channels=64,
                             hidden_size=128, num_simulation=100,
                             num_selfplay_games=12, batch_size=64)),
    ('B_small_deepsim', dict(n_residual_block=3, resnet_channels=64,
                             hidden_size=128, num_simulation=200,
                             num_selfplay_games=8, batch_size=64)),
    ('C_mid_baseline', dict(n_residual_block=5, resnet_channels=128,
                            hidden_size=256, num_simulation=200,
                            num_selfplay_games=10, batch_size=128)),
    ('D_mid_lowcpuct', dict(n_residual_block=5, resnet_channels=128,
                            hidden_size=256, num_simulation=200,
                            num_selfplay_games=10, batch_size=128,
                            cpuct=1.0)),
    ('E_mid_highcpuct', dict(n_residual_block=5, resnet_channels=128,
                             hidden_size=256, num_simulation=200,
                             num_selfplay_games=10, batch_size=128,
                             cpuct=2.5)),
    ('F_mid_hist1', dict(n_residual_block=5, resnet_channels=128,
                         hidden_size=256, num_simulation=200,
                         num_selfplay_games=10, batch_size=128,
                         history_size=1)),
]

GRID = {
    'cpuct': [1.0, 1.5, 2.5],
    'num_simulation': [100, 200],
}


# =====================================================================
def run_trial(name, overrides, iterations, work_dir, bootstrap_epochs=4,
              eval_games=20, base_preset='balanced', verbose=False):
    """1 設定を iterations 回まわして最終スコアを返す。"""

    CFG = get_cfg(base_preset,
                  work_dir=os.path.join(work_dir, name),
                  model_name='trial_' + name,
                  **overrides)

    torch.manual_seed(0)
    np.random.seed(0)

    env = make_env(CFG)
    model = AlphaZeroNetwork(CFG).to(CFG.device)
    n_params = sum(p.numel() for p in model.parameters())
    util = Util(CFG)
    trainer = TrainFixed(model, CFG)
    self_play = SelfPlay(CFG, env, model)

    t_start = time.time()

    # ---- ブートストラップ ----
    boot_acc = None
    if CFG.use_bootstrap:
        boot = generate_bootstrap_dataset(CFG, CFG.bootstrap_samples, verbose=False)
        trainer.set_lr(CFG.learning_rate)
        for _ in range(bootstrap_epochs):
            trainer(boot)
        boot_acc = tactical_test(env, model, CFG, 60, seed=0)['accuracy']
        self_play.dataset = boot[: CFG.max_dataset_size // 4]

    # ---- 自己対局 + 学習 ----
    history = []
    for it in range(1, iterations + 1):
        for _ in range(CFG.num_selfplay_games):
            with contextlib.redirect_stdout(io.StringIO()):
                dataset = self_play()
        pol, val = trainer(dataset, quiet=True)
        history.append({'iter': it, 'pi_loss': round(pol, 5),
                        'v_loss': round(val, 5), 'data': len(dataset)})
        if verbose:
            print(f'    [{name}] iter {it}/{iterations} '
                  f'pi={pol:.4f} v={val:.4f}', flush=True)

    elapsed = time.time() - t_start

    # ---- 評価 ----
    res = evaluate_both_sides(env, model, CFG, eval_games, seed=7)
    tac = tactical_test(env, model, CFG, 100, seed=7)

    torch.save(model.state_dict(), CFG.model_path)

    return {
        'name': name,
        'params': n_params,
        'score_rate': round(res['score_rate'], 4),
        'win_rate': round(res['win_rate'], 4),
        'first': round(res['first'], 4),
        'second': round(res['second'], 4),
        'tactic': round(tac['accuracy'], 4),
        'boot_tactic': None if boot_acc is None else round(boot_acc, 4),
        'pi_loss': history[-1]['pi_loss'] if history else None,
        'v_loss': history[-1]['v_loss'] if history else None,
        'sec': round(elapsed, 1),
        'sec_per_iter': round(elapsed / max(1, iterations), 1),
        'overrides': overrides,
        'history': history,
    }


def save_results(results, path):
    keys = ['name', 'params', 'score_rate', 'win_rate', 'first', 'second',
            'tactic', 'boot_tactic', 'pi_loss', 'v_loss', 'sec', 'sec_per_iter']
    with open(path, 'w') as f:
        f.write(','.join(keys) + ',overrides\n')
        for r in results:
            f.write(','.join(str(r[k]) for k in keys))
            f.write(',"' + json.dumps(r['overrides']).replace('"', "'") + '"\n')
    print('saved ->', path)


def print_table(results):
    print()
    print('=' * 96)
    print(f'{"name":18} {"params":>9} {"score":>7} {"win":>7} '
          f'{"tactic":>7} {"pi_loss":>8} {"sec/it":>8}')
    print('-' * 96)
    for r in sorted(results, key=lambda x: -x['score_rate']):
        print(f'{r["name"]:18} {r["params"]/1e6:8.2f}M '
              f'{r["score_rate"]*100:6.1f}% {r["win_rate"]*100:6.1f}% '
              f'{r["tactic"]*100:6.1f}% {r["pi_loss"]:8.4f} '
              f'{r["sec_per_iter"]:8.1f}')
    print('=' * 96)
    best = max(results, key=lambda x: x['score_rate'])
    print(f'\nbest: {best["name"]}  ->  {best["overrides"]}\n')


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='compare',
                    choices=['compare', 'single', 'grid'])
    ap.add_argument('--preset', default='balanced')
    ap.add_argument('--iterations', type=int, default=8)
    ap.add_argument('--eval-games', type=int, default=20)
    ap.add_argument('--bootstrap-epochs', type=int, default=4)
    ap.add_argument('--work-dir', default='./hpsearch')
    ap.add_argument('--device', default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    common = {}
    if args.device:
        common['device'] = args.device

    if args.mode == 'single':
        trials = [(f'single_{args.preset}', dict(common))]
        base = args.preset
    elif args.mode == 'compare':
        trials = [(n, {**o, **common}) for n, o in CANDIDATES]
        base = 'balanced'
    else:  # grid
        trials = []
        keys = list(GRID)
        for combo in itertools.product(*[GRID[k] for k in keys]):
            o = dict(zip(keys, combo))
            o.update(n_residual_block=3, resnet_channels=64, hidden_size=128,
                     num_selfplay_games=10, batch_size=64)
            o.update(common)
            name = '_'.join(f'{k}{v}' for k, v in zip(keys, combo))
            trials.append((name, o))
        base = 'balanced'

    print(f'mode={args.mode}  trials={len(trials)}  iterations={args.iterations}')
    for n, o in trials:
        print(f'  - {n}: {o}')
    print()

    results = []
    for i, (name, over) in enumerate(trials, 1):
        print(f'>>> [{i}/{len(trials)}] {name} ...', flush=True)
        r = run_trial(name, over, args.iterations, args.work_dir,
                      bootstrap_epochs=args.bootstrap_epochs,
                      eval_games=args.eval_games, base_preset=base,
                      verbose=args.verbose)
        results.append(r)
        print(f'    score={r["score_rate"]:.1%} tactic={r["tactic"]:.1%} '
              f'({r["sec"]:.0f}s)', flush=True)

    print_table(results)
    save_results(results, f'./results/hpsearch_{args.mode}.csv')


if __name__ == '__main__':
    main()
