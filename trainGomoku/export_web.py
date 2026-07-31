#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_web.py
=============
学習済みモデルを「ブラウザだけで動く1枚のHTML」に書き出すツール。

    python export_web.py                       # models/gomoku_7x7_cpu.pt を変換
    python export_web.py --model models/xxx.pt --out web/xxx.html

出力された HTML はローカルのブラウザでそのまま開けます。
サーバーも通信も追加ライブラリも不要です（重みは HTML に埋め込まれます）。

何をしているか
--------------
1. BatchNorm を直前の畳み込みに畳み込む（fold）。
   推論時の BN は y = gamma*(x-mean)/sqrt(var+eps) + beta という
   チャンネルごとの一次式なので、bias=False の Conv と数学的に合成できる。
       W' = W * scale                 (scale = gamma/sqrt(var+eps))
       b' = beta - mean * scale
   これで JS 側は Conv と ReLU だけ実装すればよくなる。結果は厳密に一致する。

2. 全パラメータを float32 の1本のバッファに連結し、base64 で HTML に埋め込む。
   どこに何があるかは manifest(JSON) に記録する。

3. 検証用に、PyTorch 側で計算した数局面分の出力 (policy/value) も一緒に埋め込む。
   ブラウザ側は起動時にそれを再計算して突き合わせ、実装が一致しているかを
   自己診断する（ページ上部に結果が出る）。
"""

import argparse
import base64
import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.append('.')
sys.path.append('..')

from CFG_7x7 import get_cfg
from AlphaZeroCode.network.AlphaZeroNetwork import AlphaZeroNetwork

ARCH_KEYS = ('n_residual_block', 'resnet_channels', 'history_size', 'hidden_size')


def apply_saved_arch(CFG, model_path):
    """学習時に保存された <model>.pt.txt からネットワーク構成を復元する。"""
    cand = model_path + '.txt'
    if not os.path.exists(cand):
        return {}
    arch = {}
    with open(cand) as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = line.strip().split(':', 1)
            if k in ARCH_KEYS:
                arch[k] = int(v)
    for k, v in arch.items():
        setattr(CFG, k, v)
    CFG.setup()
    return arch


class Blob:
    """float32 を連結していき、どこに何を置いたかを manifest に残す。"""

    def __init__(self):
        self.parts = []
        self.offset = 0          # float 単位
        self.manifest = {}

    def add(self, name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32).reshape(-1))
        self.manifest[name] = {'offset': self.offset, 'size': int(a.size)}
        self.parts.append(a)
        self.offset += int(a.size)
        return name

    def to_base64(self):
        buf = np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
        return base64.b64encode(buf.tobytes()).decode('ascii'), int(buf.size)


def fold_bn(sd, conv_key, bn_prefix, eps=1e-5):
    """
    bias=False の Conv と、その直後の BatchNorm を1つの Conv+bias に合成する。
    戻り値: (W', b')
    """
    W = sd[conv_key].detach().cpu().numpy().astype(np.float64)
    gamma = sd[bn_prefix + '.weight'].detach().cpu().numpy().astype(np.float64)
    beta = sd[bn_prefix + '.bias'].detach().cpu().numpy().astype(np.float64)
    mean = sd[bn_prefix + '.running_mean'].detach().cpu().numpy().astype(np.float64)
    var = sd[bn_prefix + '.running_var'].detach().cpu().numpy().astype(np.float64)

    scale = gamma / np.sqrt(var + eps)
    W2 = W * scale.reshape(-1, 1, 1, 1)
    b2 = beta - mean * scale
    return W2.astype(np.float32), b2.astype(np.float32)


def build_reference(model, CFG, n_cases=6, seed=1234):
    """
    JS 実装の検証用に、ランダムな合法局面での (policy, value) を PyTorch で求める。
    """
    rng = np.random.RandomState(seed)
    A = CFG.action_size
    h = CFG.history_size
    cases = []

    model.eval()
    for _ in range(n_cases):
        # ランダムに数手進めた履歴つき局面を作る
        n_moves = int(rng.randint(0, 12))
        board = np.zeros(A, dtype=np.int8)
        history = [board.copy()]
        player = CFG.first_player
        for _ in range(n_moves):
            empty = np.where(board == 0)[0]
            if len(empty) == 0:
                break
            a = int(rng.choice(empty))
            board = board.copy()
            board[a] = player
            history.insert(0, board.copy())
            player = -player

        states = []
        for i in range(h):
            states.append(history[i] if i < len(history) else np.zeros(A, np.int8))

        feat = np.zeros((h * 2 + 1, CFG.board_width, CFG.board_width), np.float32)
        for i in range(h):
            feat[i] = (states[i] == 1).reshape(CFG.board_width, CFG.board_width)
            feat[h + i] = (states[i] == -1).reshape(CFG.board_width, CFG.board_width)
        feat[h * 2] = 1.0 if player == -1 else 0.0

        with torch.no_grad():
            p, v = model(torch.from_numpy(feat).unsqueeze(0).to(CFG.device))

        cases.append({
            'states': [s.astype(int).tolist() for s in states],
            'player': int(player),
            'policy': [round(float(x), 7) for x in p[0].cpu().numpy()],
            'value': round(float(v[0][0].cpu().numpy()), 7),
        })
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='cpu')
    ap.add_argument('--model', default=None, help='読み込む .pt (既定: CFG.model_path)')
    ap.add_argument('--work-dir', default='.')
    ap.add_argument('--out', default=None, help='出力 HTML (既定: web/<model名>.html)')
    ap.add_argument('--template', default=None)
    ap.add_argument('--sims', type=int, default=100, help='ページ初期のシミュレーション回数')
    args = ap.parse_args()

    CFG = get_cfg(args.preset, work_dir=args.work_dir)
    CFG.device = 'cpu'
    model_path = args.model or CFG.model_path
    if not os.path.exists(model_path):
        raise SystemExit(f'モデルが見つかりません: {model_path}')

    arch = apply_saved_arch(CFG, model_path)
    if arch:
        print(f'arch   : {model_path}.txt から復元 -> {arch}')

    model = AlphaZeroNetwork(CFG).to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    sd = model.state_dict()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'model  : {model_path}  ({n_params:,} params)')

    # ---- BN を畳み込みに合成して1本のバッファへ ----
    blob = Blob()

    W, b = fold_bn(sd, 'conv1.weight', 'bn1')
    blob.add('conv1.w', W)
    blob.add('conv1.b', b)

    for i in range(CFG.n_residual_block):
        W, b = fold_bn(sd, f'resnet.{i}.conv1.weight', f'resnet.{i}.batchnorm1')
        blob.add(f'res{i}.w1', W)
        blob.add(f'res{i}.b1', b)
        W, b = fold_bn(sd, f'resnet.{i}.conv2.weight', f'resnet.{i}.batchnorm2')
        blob.add(f'res{i}.w2', W)
        blob.add(f'res{i}.b2', b)

    W, b = fold_bn(sd, 'conv_policy1.weight', 'bn_policy1')
    blob.add('pol1.w', W)
    blob.add('pol1.b', b)
    W, b = fold_bn(sd, 'conv_policy2.weight', 'bn_policy2')
    blob.add('pol2.w', W)
    blob.add('pol2.b', b)

    W, b = fold_bn(sd, 'conv_value.weight', 'bn_value')
    blob.add('val.w', W)
    blob.add('val.b', b)

    blob.add('fc1.w', sd['fc_value1.weight'].detach().cpu().numpy())
    blob.add('fc1.b', sd['fc_value1.bias'].detach().cpu().numpy())
    blob.add('fc2.w', sd['fc_value2.weight'].detach().cpu().numpy())
    blob.add('fc2.b', sd['fc_value2.bias'].detach().cpu().numpy())

    b64, n_floats = blob.to_base64()
    print(f'weights: {n_floats:,} floats -> base64 {len(b64)/1e6:.2f} MB')

    # ---- 検証用の参照出力 ----
    ref = build_reference(model, CFG)
    print(f'selftest: {len(ref)} 局面の参照出力を埋め込み')

    meta = {
        'model_name': os.path.basename(model_path),
        'board_width': CFG.board_width,
        'row': CFG.row,
        'action_size': CFG.action_size,
        'history_size': CFG.history_size,
        'resnet_channels': CFG.resnet_channels,
        'n_residual_block': CFG.n_residual_block,
        'hidden_size': CFG.hidden_size,
        'in_channels': CFG.history_size * 2 + 1,
        'first_player': CFG.first_player,
        'cpuct': CFG.cpuct,
        'default_sims': args.sims,
        'n_params': int(n_params),
        'manifest': blob.manifest,
        'reference': ref,
    }

    here = os.path.dirname(os.path.abspath(__file__))
    template_path = args.template or os.path.join(here, 'web', 'template.html')
    with open(template_path, encoding='utf-8') as f:
        html = f.read()

    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        out_path = os.path.join(here, 'web', stem + '.html')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    html = html.replace('"__META_JSON__"', json.dumps(meta, ensure_ascii=False))
    html = html.replace('__WEIGHTS_B64__', b64)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'\n書き出し完了: {out_path}  ({size_mb:.2f} MB)')
    print('ブラウザでそのまま開いてください（サーバー不要）。')


if __name__ == '__main__':
    main()
