#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prove_stronger.py
=================
「自己対局だけで学習した AlphaZero が、人の考えたアルゴリズムより強い」
ことを検証して、レポートを出力するスクリプト。

    python prove_stronger.py --preset balanced --model models/xxx_best.pt \
                             --pairs 40 --levels 1 2 3

検証の手順:

  step 0  較正 : 人手アルゴリズム(L1/L2/L3) 同士とランダムを戦わせ、
                 「対照群が本当に強いのか」を先に数値で示す。
                 （弱い相手に勝っても証明にならないため）
  step 1  本番 : AZ vs 各レベルをペア方式（先後入替）で対戦。
  step 2  判定 : 引き分けを除いた勝敗に片側二項検定。
                 p < 0.05 かつスコア率 > 50% なら「有意に強い」と判定。

出力:
  results/proof_report.md    人が読むレポート
  results/proof_result.json  生データ（全局の結果つき）
  results/proof_summary.png  図（あれば）
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append('.')
sys.path.append('..')

from AlphaZeroCode.network.AlphaZeroNetwork import AlphaZeroNetwork

from CFG_7x7 import get_cfg
from az_common import make_env, evaluate_both_sides, tactical_test
from human_algorithm import HumanAlgorithm, RandomPlayer, engine_vs_engine
from arena import play_match, format_result, board_to_text, save_json


ALPHA = 0.05


# ---------------------------------------------------------------------
ARCH_KEYS = ('n_residual_block', 'resnet_channels', 'history_size', 'hidden_size')


def apply_saved_arch(CFG, model_path):
    """
    Util.save_model_info() が残す `<model>.pt.txt` からネットワーク構成を復元する。

    学習時に --preset 以外でブロック数やチャンネル数を変えていると、
    ここで合わせないと load_state_dict が size mismatch で落ちる。
    """
    for cand in (model_path + '.txt', CFG.model_path + '.txt'):
        if not os.path.exists(cand):
            continue
        arch = {}
        with open(cand) as f:
            for line in f:
                if ':' not in line:
                    continue
                k, v = line.strip().split(':', 1)
                if k in ARCH_KEYS:
                    arch[k] = int(v)
        if arch:
            for k, v in arch.items():
                setattr(CFG, k, v)
            CFG.setup()
            print(f'arch   : {cand} から復元 -> {arch}')
            return arch
    return {}


def load_model(CFG, path=None):
    path = path or CFG.model_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'モデルが見つかりません: {path}\n'
            f'先に train_gomoku_7x7.py で学習してください。')
    model = AlphaZeroNetwork(CFG).to(CFG.device)
    model.load_state_dict(torch.load(path, map_location=CFG.device))
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f'model  : {path}')
    print(f'params : {n:,} ({n/1e6:.2f}M)')
    return model, path


# ---------------------------------------------------------------------
def calibrate(CFG, levels, games, seed=0):
    """対照群（人手アルゴリズム）の強さを先に測る。"""
    print('\n' + '=' * 72)
    print('step 0: 対照群の較正 — 人の考えたアルゴリズムはどれくらい強いのか')
    print('=' * 72)

    out = {}
    rnd = RandomPlayer(width=CFG.board_width, seed=1234)

    for lv in levels:
        eng = HumanAlgorithm(width=CFG.board_width, row=CFG.row, level=lv, seed=lv)
        t0 = time.time()
        r = engine_vs_engine(eng, rnd, CFG.board_width, CFG.row, games, seed=100 + lv)
        r['seconds'] = round(time.time() - t0, 1)
        out[f'L{lv}_vs_random'] = r
        print(f'  {eng.name:38s} vs ランダム : '
              f'スコア {r["score_rate"]:6.1%}  '
              f'({r["win"]}勝 {r["draw"]}分 {r["lose"]}敗, {r["seconds"]}s)',
              flush=True)

    if len(levels) >= 2:
        lo, hi = min(levels), max(levels)
        a = HumanAlgorithm(width=CFG.board_width, row=CFG.row, level=hi, seed=hi)
        b = HumanAlgorithm(width=CFG.board_width, row=CFG.row, level=lo, seed=lo + 50)
        r = engine_vs_engine(a, b, CFG.board_width, CFG.row, games, seed=77)
        out[f'L{hi}_vs_L{lo}'] = r
        print(f'  L{hi} vs L{lo} : スコア {r["score_rate"]:6.1%} '
              f'({r["win"]}勝 {r["draw"]}分 {r["lose"]}敗)  '
              f'-> 先読みが効いていれば L{hi} が上回る', flush=True)

    return out


# ---------------------------------------------------------------------
def run_proof(CFG, model, levels, pairs, seed=0, opening_plies=2):
    print('\n' + '=' * 72)
    print('step 1: AlphaZero vs 人の考えたアルゴリズム（先後入替のペア方式）')
    print('=' * 72)

    env = make_env(CFG)
    results = {}

    for lv in levels:
        eng = HumanAlgorithm(width=CFG.board_width, row=CFG.row, level=lv,
                             seed=1000 + lv)
        print(f'\n  --- vs {eng.name} : {pairs} ペア = {pairs*2} 局 ---', flush=True)
        res = play_match(env, model, CFG, eng, n_pairs=pairs,
                         seed=seed + lv, opening_plies=opening_plies)
        print(format_result(res, ALPHA), flush=True)
        results[f'L{lv}'] = res

    return results


# ---------------------------------------------------------------------
def write_report(path, CFG, model_path, calib, results, extra, levels):
    lines = []
    A = lines.append

    A('# AlphaZero は人の考えたアルゴリズムより強くなったか')
    A('')
    A(f'- 生成日時: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    A(f'- ゲーム: {CFG.board_width}x{CFG.board_width} 盤 / {CFG.row} 目並べ')
    A(f'- モデル: `{model_path}`')
    A(f'- ネットワーク: {CFG.n_residual_block} ブロック × {CFG.resnet_channels} ch '
      f'(hidden {CFG.hidden_size}, history {CFG.history_size})')
    A(f'- 対戦時の MCTS シミュレーション回数: {CFG.num_simulation}'
      f'（ディリクレノイズは 0 に固定）')
    A(f'- device: {CFG.device}')
    A('')
    A('AlphaZero 側は **人間の棋譜も定石も一切使わず**、自己対局のみで学習した')
    A('モデルです（ルールを教える bootstrap は「勝ち手を打つ / 相手のリーチを止める」')
    A('という規則そのものだけで、戦略は含みません）。')
    A('')

    A('## 0. 対照群の較正')
    A('')
    A('弱い相手に勝っても意味がないので、まず「人の考えたアルゴリズム」自体の')
    A('強さを測ります。')
    A('')
    A('| 対戦 | スコア率 | 勝 | 分 | 敗 |')
    A('|---|---|---|---|---|')
    for k, r in calib.items():
        A(f'| {k.replace("_", " ")} | {r["score_rate"]:.1%} | '
          f'{r["win"]} | {r["draw"]} | {r["lose"]} |')
    A('')
    A('人手アルゴリズムの中身:')
    A('')
    A('1. 即勝ち手があれば必ず打つ')
    A('2. 相手の即勝ち手を必ず止める（2箇所なら受け無しと評価）')
    A('3. 長さ5の窓ごとに 4連=2000点 / 3連=120点 … と形を点数化（重みは人が決定）')
    A('4. 石の周囲2マスに候補手を絞り込み（人が決めた枝刈り）')
    A('5. 上記の評価関数を α-β 法で先読み（L1=0手, L2=2手, L3=4手）')
    A('')

    A('## 1. 対戦結果')
    A('')
    A('先手有利を打ち消すため、**同一のオープニングから先手・後手を1局ずつ**')
    A('打つペア方式です。')
    A('')
    A('| 相手 | 対局数 | 勝 | 分 | 敗 | スコア率 | 95%CI | 片側p値 | Elo差 | 判定 |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for key in [f'L{lv}' for lv in levels]:
        r = results[key]
        lo, hi = r['ci95']
        ok = r['p_value'] < ALPHA and r['score_rate'] > 0.5
        A(f'| {r["engine"]} | {r["games"]} | {r["win"]} | {r["draw"]} | {r["lose"]} | '
          f'{r["score_rate"]:.1%} | [{lo:.1%}, {hi:.1%}] | {r["p_value"]:.3g} | '
          f'{r["elo"]:+.0f} | {"**有意に強い**" if ok else "有意差なし"} |')
    A('')
    A('先手番 / 後手番の内訳:')
    A('')
    A('| 相手 | 先手 (勝/分/敗) | 後手 (勝/分/敗) |')
    A('|---|---|---|')
    for key in [f'L{lv}' for lv in levels]:
        r = results[key]
        f_, s_ = r['as_first'], r['as_second']
        A(f'| {r["engine"]} | {f_["win"]}/{f_["draw"]}/{f_["lose"]} | '
          f'{s_["win"]}/{s_["draw"]}/{s_["lose"]} |')
    A('')

    if extra:
        A('## 2. 参考指標')
        A('')
        for k, v in extra.items():
            A(f'- {k}: {v}')
        A('')

    A('## 3. 検定の中身')
    A('')
    A('- 帰無仮説 H0: AlphaZero と人手アルゴリズムの実力は同じ')
    A('  （引き分けを除いた1局の勝率が 0.5）')
    A('- 対立仮説 H1: AlphaZero の方が強い（勝率 > 0.5）')
    A('- 検定: 片側二項検定（符号検定）。有意水準 5%')
    A('- スコア率の 95% 信頼区間は Wilson 法（引き分けは 0.5 勝として計上）')
    A('')

    A('## 4. 結論')
    A('')
    proven = [lv for lv in levels
              if results[f'L{lv}']['p_value'] < ALPHA
              and results[f'L{lv}']['score_rate'] > 0.5]
    failed = [lv for lv in levels if lv not in proven]

    if proven:
        for lv in proven:
            r = results[f'L{lv}']
            A(f'- **{r["engine"]} に対して有意に強い**: '
              f'スコア率 {r["score_rate"]:.1%}, p = {r["p_value"]:.3g} '
              f'(< {ALPHA}), Elo差 {r["elo"]:+.0f}')
    if failed:
        for lv in failed:
            r = results[f'L{lv}']
            A(f'- {r["engine"]} に対しては **証明できていない**: '
              f'スコア率 {r["score_rate"]:.1%}, p = {r["p_value"]:.3g}。'
              f'学習イテレーションを増やすか、シミュレーション回数を上げてください。')
    A('')
    if proven and not failed:
        A('すべてのレベルの人手アルゴリズムを有意に上回りました。')
        A('自己対局のみで学習したモデルが、人が設計した探索＋評価関数を超えたことになります。')
    elif proven:
        A('一部のレベルは超えましたが、最上位レベルはまだ超えていません。')
    else:
        A('現時点では人手アルゴリズムを超えたと言えません。学習を継続してください。')
    A('')

    A('## 5. サンプル棋譜')
    A('')
    for key in [f'L{lv}' for lv in levels]:
        r = results[key]
        for i, g in enumerate(r.get('sample_games', [])[:1]):
            A(f'### vs {r["engine"]} — {g["result"]} '
              f'(AZ = {"先手X" if g["az_player"] == CFG.first_player else "後手O"}, '
              f'{g["moves"]}手)')
            A('')
            A('```')
            A(board_to_text(g['board']))
            A('```')
            A('')

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


# ---------------------------------------------------------------------
def make_figure(path, results, calib, levels):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:                                    # noqa: BLE001
        print('matplotlib が使えないので図は省略:', e)
        return None

    names = [results[f'L{lv}']['engine'].split(' ')[0] for lv in levels]
    scores = [results[f'L{lv}']['score_rate'] for lv in levels]
    los = [results[f'L{lv}']['score_rate'] - results[f'L{lv}']['ci95'][0] for lv in levels]
    his = [results[f'L{lv}']['ci95'][1] - results[f'L{lv}']['score_rate'] for lv in levels]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(names))
    ax.bar(x, scores, yerr=[los, his], capsize=6, color='#4c72b0')
    ax.axhline(0.5, ls='--', c='crimson', label='50% (equal strength)')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AlphaZero score rate')
    ax.set_title('AlphaZero vs hand-crafted algorithm (paired colours, 95% CI)')
    for i, s in enumerate(scores):
        ax.text(i, min(s + his[i] + 0.03, 1.02), f'{s:.0%}', ha='center')
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='balanced')
    ap.add_argument('--model', default=None, help='読み込む .pt (既定: CFG.model_path)')
    ap.add_argument('--work-dir', default='.')
    ap.add_argument('--model-name', default=None)
    ap.add_argument('--pairs', type=int, default=25, help='1レベルあたりのペア数(×2局)')
    ap.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--sims', type=int, default=None, help='対戦時のMCTS回数')
    ap.add_argument('--calib-games', type=int, default=20)
    ap.add_argument('--opening-plies', type=int, default=2)
    ap.add_argument('--device', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-calibration', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--blocks', type=int, default=None, help='n_residual_block')
    ap.add_argument('--channels', type=int, default=None, help='resnet_channels')
    ap.add_argument('--hidden', type=int, default=None, help='hidden_size')
    ap.add_argument('--history', type=int, default=None, help='history_size')
    args = ap.parse_args()

    overrides = {'work_dir': args.work_dir}
    if args.sims:
        overrides['num_simulation'] = args.sims
    if args.device:
        overrides['device'] = args.device
    if args.model_name:
        overrides['model_name'] = args.model_name

    CFG = get_cfg(args.preset, **overrides)

    # ネットワーク構成: 明示指定 > 保存された model_info > プリセット
    explicit = {'n_residual_block': args.blocks, 'resnet_channels': args.channels,
                'hidden_size': args.hidden, 'history_size': args.history}
    explicit = {k: v for k, v in explicit.items() if v is not None}
    if explicit:
        for k, v in explicit.items():
            setattr(CFG, k, v)
        CFG.setup()
        print('arch   : コマンドラインで指定 ->', explicit)
    else:
        apply_saved_arch(CFG, args.model or CFG.model_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 72)
    print('AlphaZero vs 人の考えたアルゴリズム')
    print(CFG.summary())
    print('=' * 72)

    model, model_path = load_model(CFG, args.model)
    env = make_env(CFG)

    calib = {} if args.skip_calibration else calibrate(CFG, args.levels, args.calib_games)
    results = run_proof(CFG, model, args.levels, args.pairs,
                        seed=args.seed, opening_plies=args.opening_plies)

    # 参考指標
    print('\n' + '=' * 72)
    print('step 2: 参考指標')
    print('=' * 72)
    extra = {}
    saved_eps = CFG.Dirichlet_epsilon
    CFG.Dirichlet_epsilon = 0.0
    try:
        vr = evaluate_both_sides(env, model, CFG, 20, seed=7)
        extra['AZ vs ランダム (20局)'] = (f'スコア {vr["score_rate"]:.1%} '
                                          f'({vr["win"]}勝 {vr["draw"]}分 {vr["lose"]}敗)')
        tac = tactical_test(env, model, CFG, 200, seed=999)
        extra['戦術正解率 (1手詰め 200局面)'] = (
            f'{tac["accuracy"]:.1%} '
            f'(ランダム推測なら {tac.get("random_baseline", 0.0):.1%})')
    finally:
        CFG.Dirichlet_epsilon = saved_eps
    for k, v in extra.items():
        print(f'  {k}: {v}')

    tag = ('_' + args.tag) if args.tag else ''
    rep = write_report(os.path.join(CFG.result_dir, f'proof_report{tag}.md'),
                       CFG, model_path, calib, results, extra, args.levels)
    js = save_json({'config': CFG.summary(), 'model': model_path,
                    'calibration': calib, 'results': results, 'extra': extra},
                   os.path.join(CFG.result_dir, f'proof_result{tag}.json'))
    fig = make_figure(os.path.join(CFG.result_dir, f'proof_summary{tag}.png'),
                      results, calib, args.levels)

    print('\n' + '=' * 72)
    print('出力:')
    print('  ', rep)
    print('  ', js)
    if fig:
        print('  ', fig)
    print('=' * 72)

    proven = [lv for lv in args.levels
              if results[f'L{lv}']['p_value'] < ALPHA
              and results[f'L{lv}']['score_rate'] > 0.5]
    print(f'有意に上回ったレベル: {proven if proven else "なし"}')
    return 0 if len(proven) == len(args.levels) else 1


if __name__ == '__main__':
    sys.exit(main())
