#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
human_algorithm.py
==================
「人の考えたアルゴリズム」= 人手で設計した五目並べエンジン。

AlphaZero が「人の知識なしで、自己対局だけで」人手設計のアルゴリズムを
超えられるかを検証するための *対照群* です。学習は一切しません。

設計は五目並べAIの教科書的な構成そのものです。

  1. 即勝ち手があれば必ず打つ                     （人が書いたルール）
  2. 相手に即勝ち手があれば必ず止める             （人が書いたルール）
  3. 相手の即勝ち手が2箇所（両取り）なら負けと評価（人が書いたルール）
  4. 残りは「長さ5の窓」パターン評価関数で形の良さを点数化
     （4連=2000点, 3連=120点 ... という重み付けも人が決めたもの）
  5. その評価関数を α-β 法で数手先読みする

レベル（強さ）:

  L1 : 先読みなし。ルール + 1手先の形の評価だけ。
  L2 : 2手先読み（相手の応手まで読む）。
  L3 : 4手先読み。7x7 盤ならかなり強い。

盤面表現:
  外部 API は AlphaZeroCode の Gomoku と同じ「2次元リスト、+1/-1/0」。
  内部では 1 次元リストに直して探索します（速度のため）。

使い方:
    engine = HumanAlgorithm(width=7, row=5, level=3, seed=0)
    action = engine.select(env.state, env.player)
"""

import random

WIN_SCORE = 10_000_000


class HumanAlgorithm:
    """人手設計の五目並べエンジン（学習しない）。"""

    LEVELS = {
        1: dict(depth=0, top_k=64, label='L1 ルールのみ(先読みなし)'),
        2: dict(depth=2, top_k=14, label='L2 ルール+2手読み'),
        3: dict(depth=4, top_k=10, label='L3 ルール+4手読み(α-β)'),
    }

    # 窓(長さ row)の中の石数に対する点数。人が決めた重み。
    WEIGHTS = (0, 1, 10, 120, 2000, WIN_SCORE)

    def __init__(self, width=7, row=5, level=2, seed=0, defense=1.15,
                 radius=2, depth=None, top_k=None):
        assert level in self.LEVELS, f'level must be one of {list(self.LEVELS)}'
        self.width = width
        self.row = row
        self.level = level
        self.defense = defense        # 守り重視の係数（人のチューニング）
        self.radius = radius
        self.rng = random.Random(seed)

        conf = self.LEVELS[level]
        self.depth = conf['depth'] if depth is None else depth
        self.top_k = conf['top_k'] if top_k is None else top_k
        self.label = conf['label']
        self.name = f'HumanAlgorithm-{level} ({self.label})'

        # 窓の重み（row が 5 以外でも動くように長さを合わせる）
        if row + 1 == len(self.WEIGHTS):
            W = list(self.WEIGHTS)
        else:
            W = ([0] + [10 ** i for i in range(row)] + [WIN_SCORE])[:row + 1]
        # W[n+1] を引くので、番兵をひとつ足して IndexError を構造的に潰しておく
        self.W = tuple(W + [WIN_SCORE])

        self._build_lines()
        self.nodes = 0                # 探索ノード数（統計用）

    # -----------------------------------------------------------------
    # 前処理
    # -----------------------------------------------------------------
    def _build_lines(self):
        """長さ row の窓を全部列挙して、マスごとの逆引きも作る。"""
        w, row = self.width, self.row
        lines = []
        for r in range(w):
            for c in range(w):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
                    er, ec = r + dr * (row - 1), c + dc * (row - 1)
                    if not (0 <= er < w and 0 <= ec < w):
                        continue
                    lines.append(tuple((r + dr * i) * w + (c + dc * i)
                                       for i in range(row)))
        self.lines = tuple(lines)

        through = [[] for _ in range(w * w)]
        for li, line in enumerate(self.lines):
            for idx in line:
                through[idx].append(li)
        self.lines_through = tuple(tuple(x) for x in through)

        cc = w // 2
        self.center_bonus = tuple(
            2.0 * (w - abs(i // w - cc) - abs(i % w - cc)) for i in range(w * w))

    # -----------------------------------------------------------------
    # 基本判定
    # -----------------------------------------------------------------
    def _wins_at(self, board, a, p):
        """board の a に p を置いたら row 連が成立するか（a は空でもよい）。"""
        w, row = self.width, self.row
        r0, c0 = divmod(a, w)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (-1, 1)):
            cnt = 1
            for sign in (1, -1):
                r, c = r0 + dr * sign, c0 + dc * sign
                while 0 <= r < w and 0 <= c < w and board[r * w + c] == p:
                    cnt += 1
                    r += dr * sign
                    c += dc * sign
            if cnt >= row:
                return True
        return False

    def _winning_cells(self, board, p, cells):
        return [a for a in cells if self._wins_at(board, a, p)]

    def _candidates(self, board):
        """石の周囲 radius 以内の空きマス。序盤の無駄手を人が刈り込む定石。"""
        w, rad = self.width, self.radius
        occupied = [i for i, v in enumerate(board) if v]
        if not occupied:
            c = w // 2
            return [c * w + c]

        cand = set()
        for i in occupied:
            r0, c0 = divmod(i, w)
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < w and 0 <= c < w and board[r * w + c] == 0:
                        cand.add(r * w + c)
        if not cand:
            cand = {i for i, v in enumerate(board) if v == 0}
        return sorted(cand)

    # -----------------------------------------------------------------
    # 評価関数（人が設計した部分の核）
    # -----------------------------------------------------------------
    def evaluate(self, board, p):
        """盤面全体を p の立場から点数化する。"""
        W, defense = self.W, self.defense
        s = 0.0
        for line in self.lines:
            mine = opp = 0
            for idx in line:
                v = board[idx]
                if v == p:
                    mine += 1
                elif v:
                    opp += 1
            if mine and opp:
                continue          # 両者の石が混ざった窓は死んでいる
            if mine:
                s += W[mine]
            elif opp:
                s -= defense * W[opp]
        return s

    def _move_score(self, board, a, p):
        """a に打つ価値の概算。手の並べ替えと L1 の着手選択に使う。"""
        W, defense = self.W, self.defense
        s = 0.0
        for li in self.lines_through[a]:
            mine = opp = 0
            for idx in self.lines[li]:
                v = board[idx]
                if v == p:
                    mine += 1
                elif v:
                    opp += 1
            if mine and opp:
                continue
            if opp:
                s += defense * (W[opp + 1] - W[opp])   # 相手を止める価値
            else:
                s += W[mine + 1] - W[mine]             # 自分が伸ばす価値
        return s + self.center_bonus[a]

    def _ordered(self, board, p, cands, limit):
        scored = [(self._move_score(board, a, p), a) for a in cands]
        scored.sort(key=lambda x: -x[0])
        return [a for _, a in scored[:limit]]

    # -----------------------------------------------------------------
    # α-β 探索
    # -----------------------------------------------------------------
    def _negamax(self, board, p, depth, alpha, beta, ply):
        """p の手番。p から見た評価値を返す。"""
        self.nodes += 1

        cands = self._candidates(board)
        if not cands:
            return 0.0                                  # 盤面満杯 = 引き分け

        if self._winning_cells(board, p, cands):
            return WIN_SCORE - ply                      # 自分が即勝ち

        if depth <= 0:
            return self.evaluate(board, p)

        threats = self._winning_cells(board, -p, cands)
        if len(threats) >= 2:
            return -(WIN_SCORE - ply - 1)               # 両取り = 受け無し
        moves = threats if threats else self._ordered(board, p, cands, self.top_k)

        best = -float('inf')
        for a in moves:
            board[a] = p
            v = -self._negamax(board, -p, depth - 1, -beta, -alpha, ply + 1)
            board[a] = 0
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break                                   # β カット
        return best

    # -----------------------------------------------------------------
    # 着手選択
    # -----------------------------------------------------------------
    def select(self, state, player):
        """state(2次元) で player が打つ手を返す。"""
        w = self.width
        board = [int(state[r][c]) for r in range(w) for c in range(w)]
        return self.select_flat(board, int(player))

    def select_flat(self, board, player):
        p = player
        cands = self._candidates(board)
        if not cands:
            raise ValueError('no legal move')

        # --- ルール1: 即勝ち ---
        wins = self._winning_cells(board, p, cands)
        if wins:
            return self._pick(wins, board, p)

        # --- ルール2: 相手の即勝ちを止める ---
        threats = self._winning_cells(board, -p, cands)
        if threats:
            return self._pick(threats, board, p)

        # --- L1: 先読みなし ---
        if self.depth <= 0:
            best = max(self._move_score(board, a, p) for a in cands)
            top = [a for a in cands if self._move_score(board, a, p) >= best - 1e-9]
            return self._pick(top, board, p)

        # --- L2/L3: α-β 探索 ---
        moves = self._ordered(board, p, cands, self.top_k)
        alpha, best_v, best_moves = -float('inf'), -float('inf'), []
        for a in moves:
            board[a] = p
            v = -self._negamax(board, -p, self.depth - 1, -float('inf'), -alpha, 1)
            board[a] = 0
            if v > best_v + 1e-9:
                best_v, best_moves = v, [a]
            elif v > best_v - 1e-9:
                best_moves.append(a)
            if v > alpha:
                alpha = v
        return self._pick(best_moves, board, p)

    def _pick(self, moves, board, p):
        """同点の手はランダムに選ぶ（毎局同じ将棋にならないように）。"""
        if len(moves) == 1:
            return int(moves[0])
        best = max(self._move_score(board, a, p) for a in moves)
        top = [a for a in moves if self._move_score(board, a, p) >= best - 1e-9]
        return int(self.rng.choice(top))


# ---------------------------------------------------------------------
class RandomPlayer:
    """較正用のランダムプレーヤー。"""

    name = 'Random'
    label = 'ランダム'

    def __init__(self, width=7, seed=0, **kw):
        self.width = width
        self.rng = random.Random(seed)

    def select(self, state, player):
        w = self.width
        legal = [r * w + c for r in range(w) for c in range(w) if state[r][c] == 0]
        return int(self.rng.choice(legal))


# ---------------------------------------------------------------------
def engine_vs_engine(a, b, width=7, row=5, n_games=20, seed=0, verbose=False):
    """エンジン同士を先後入れ替えて対戦させる（較正用）。a 側から見た成績。"""
    import sys
    sys.path.append('.')
    sys.path.append('..')
    from AlphaZeroCode.env.Gomoku import Gomoku

    rng = random.Random(seed)
    env = Gomoku(row=row, width=width)
    win = draw = lose = 0

    for g in range(n_games):
        env.reset()
        a_player = -1 if g % 2 == 0 else 1     # 先手(-1) を交互に持つ
        # ばらけさせるためのランダム初手
        if g >= 2:
            legal = [i for i in range(width * width) if env.state[i // width][i % width] == 0]
            env.step(rng.choice(legal))

        while not env.done:
            engine = a if env.player == a_player else b
            env.step(engine.select(env.state, env.player))

        if env.winner == a_player:
            win += 1
        elif env.winner == 0:
            draw += 1
        else:
            lose += 1
        if verbose:
            print(f'  game {g+1}/{n_games}: w{win} d{draw} l{lose}', flush=True)

    n = max(1, n_games)
    return {'win': win, 'draw': draw, 'lose': lose,
            'score_rate': (win + 0.5 * draw) / n, 'games': n_games}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='人手設計エンジンの強さ較正')
    ap.add_argument('--games', type=int, default=20)
    ap.add_argument('--width', type=int, default=7)
    ap.add_argument('--row', type=int, default=5)
    args = ap.parse_args()

    rnd = RandomPlayer(width=args.width, seed=1)
    for lv in (1, 2, 3):
        eng = HumanAlgorithm(width=args.width, row=args.row, level=lv, seed=lv)
        r = engine_vs_engine(eng, rnd, args.width, args.row, args.games, seed=lv)
        print(f'{eng.name:42s} vs Random  -> score {r["score_rate"]:.1%} '
              f'(w{r["win"]} d{r["draw"]} l{r["lose"]})', flush=True)

    l1 = HumanAlgorithm(width=args.width, row=args.row, level=1, seed=11)
    l3 = HumanAlgorithm(width=args.width, row=args.row, level=3, seed=13)
    r = engine_vs_engine(l3, l1, args.width, args.row, args.games, seed=3)
    print(f'L3 vs L1 -> score {r["score_rate"]:.1%} '
          f'(w{r["win"]} d{r["draw"]} l{r["lose"]})')
