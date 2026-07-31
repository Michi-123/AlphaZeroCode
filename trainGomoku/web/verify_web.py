#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_web.py
=============
書き出した HTML が Python 版と同じ手を打つかを、実際のブラウザで確認する。

    pip install playwright
    python web/verify_web.py                      # 既定の HTML を検証
    python web/verify_web.py --html web/xxx.html

確認する内容:

  1. 自己診断      … ページ内蔵の検証（推論結果が PyTorch と一致するか）
  2. MCTS の一致   … 同じ局面・同じ探索回数で Python 版 MCTS を回し、
                      「選んだ手」と「訪問回数」まで一致するかを突き合わせる。
                      ディリクレノイズを切れば両者とも決定的なので、
                      実装が同じなら数値まで一致するはず。
  3. 1局の完走     … ランダムに着手して終局まで進み、勝敗判定が出るか

注意: 2 で「相手のリーチを止められない」ことがありますが、これは実装の
      不具合ではなくモデルが弱いためです（Python 版も同じ手を選びます）。
      このスクリプトは「両者が一致するか」だけを判定します。
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # trainGomoku/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # リポジトリ直下

from CFG_7x7 import get_cfg
from az_common import make_env
from AlphaZeroCode.MCTS import MCTS
from AlphaZeroCode.Node import Node
from AlphaZeroCode.network.AlphaZeroNetwork import AlphaZeroNetwork

# 検証に使う局面（黒 = -1 = 先手）
POSITIONS = [
    ('即勝ち(黒番)', [(8, -1), (9, -1), (10, -1), (11, -1), (22, 1), (23, 1), (24, 1)], -1),
    ('防御(白番)', [(8, -1), (9, -1), (10, -1), (11, -1), (36, 1), (38, 1)], 1),
    ('序盤(黒番)', [(24, -1), (25, 1), (17, -1), (31, 1)], -1),
]

JS_THINK = """
async ([stones, player, sims]) => {
  const b = new Int8Array(49);
  for (const [i, v] of stones) b[i] = v;
  const res = await think(b, [b], player, sims, null);
  return { action: res.action,
           visits: res.children.map(c => [c.action, c.n]).filter(x => x[1] > 0) };
}
"""


def python_mcts(CFG, model, stones, player, sims):
    W = CFG.board_width
    state = [[0] * W for _ in range(W)]
    for idx, v in stones:
        state[idx // W][idx % W] = v

    env = make_env(CFG)
    env.reset()
    env.state = [row[:] for row in state]
    env.player = player

    node = Node(CFG, state)
    node.player = player

    CFG.num_simulation = sims
    CFG.Dirichlet_epsilon = 0.0
    nxt = MCTS(env, model, CFG, train=False)(node)

    visits = [(int(c.action), int(c.n)) for c in node.child_nodes if c.n > 0]
    visits.sort(key=lambda t: (-t[1], t[0]))
    return int(nxt.action), visits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--html', default=os.path.join(HERE, 'gomoku_7x7_cpu.html'))
    ap.add_argument('--model', default=None)
    ap.add_argument('--preset', default='cpu')
    ap.add_argument('--sims', type=int, default=200)
    ap.add_argument('--chrome', default=None, help='chromium の実行ファイル')
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit('playwright が必要です:  pip install playwright')

    if not os.path.exists(args.html):
        raise SystemExit(f'HTML が見つかりません: {args.html}\n'
                         f'先に export_web.py を実行してください。')

    CFG = get_cfg(args.preset, work_dir=os.path.dirname(HERE))
    CFG.device = 'cpu'
    CFG.n_residual_block, CFG.resnet_channels = 3, 64
    CFG.history_size, CFG.hidden_size = 2, 128
    CFG.setup()

    model_path = args.model or CFG.model_path
    model = AlphaZeroNetwork(CFG).to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    fails = []
    launch = {'executable_path': args.chrome} if args.chrome else {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch)
        page = browser.new_page()
        errs = []
        page.on('pageerror', lambda e: errs.append(str(e)))
        page.goto('file://' + os.path.abspath(args.html))
        page.wait_for_selector('#selftest.ok, #selftest.ng', timeout=180000)

        # --- 1. 自己診断 ---
        cls = page.get_attribute('#selftest', 'class')
        print('1. 自己診断 :', page.inner_text('#selftest'))
        if cls != 'ok':
            fails.append('自己診断 NG')

        # --- 2. MCTS の一致 ---
        print('\n2. MCTS の一致（Python と訪問回数まで比較）')
        for label, stones, player in POSITIONS:
            js = page.evaluate(JS_THINK, [stones, player, args.sims])
            js_visits = sorted(((int(a), int(n)) for a, n in js['visits']),
                               key=lambda t: (-t[1], t[0]))
            py_action, py_visits = python_mcts(CFG, model, stones, player, args.sims)

            same_action = int(js['action']) == py_action
            same_visits = js_visits == py_visits
            ok = same_action and same_visits
            print(f'   {label:14s} JS={js["action"]:2d} Python={py_action:2d}  '
                  f'訪問一致={"はい" if same_visits else "いいえ"}  '
                  f'-> {"OK" if ok else "NG"}')
            if not ok:
                print(f'      JS    上位: {js_visits[:5]}')
                print(f'      Python上位: {py_visits[:5]}')
                fails.append(f'{label} が不一致')

        # --- 3. 1局の完走 ---
        page.select_option('#sims', '10')
        page.click('#newGame')
        page.wait_for_timeout(400)
        for _ in range(60):
            st = page.inner_text('#status')
            if ('勝ち' in st) or ('引き分け' in st):
                break
            idx = page.evaluate(
                "() => { const c=[...document.querySelectorAll('#board .cell')]"
                ".map((e,i)=>[e,i]).filter(([e])=>!e.disabled); "
                "return c.length ? c[Math.floor(Math.random()*c.length)][1] : -1; }")
            if idx < 0:
                page.wait_for_timeout(400)
                continue
            page.evaluate("i => document.querySelectorAll('#board .cell')[i].click()", idx)
            page.wait_for_function(
                "() => !document.querySelector('#status').textContent.includes('考えています')",
                timeout=180000)
            page.wait_for_timeout(100)

        final = page.inner_text('#status')
        ok3 = ('勝ち' in final) or ('引き分け' in final)
        print(f'\n3. 1局の完走: {final} -> {"OK" if ok3 else "NG"}')
        if not ok3:
            fails.append('対局が終局しない')

        if errs:
            print('\nJS エラー:', errs[:5])
            fails.append('JS エラー')

        browser.close()

    print('\n判定:', 'PASS' if not fails else 'FAIL — ' + ', '.join(fails))
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
