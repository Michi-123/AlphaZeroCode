# ブラウザで動かす

Python で学習したモデルを、**HTML 1枚**に書き出してローカルのブラウザで動かすツールです。
サーバーも通信も追加ライブラリも不要で、ファイルをダブルクリックするだけで対局できます。

## 使い方

```bash
# 1. 学習済みモデルを HTML に書き出す
python export_web.py

# 出力: web/gomoku_7x7_cpu.html  (約 1.5 MB)
```

あとは出来上がった HTML をブラウザで開くだけです。

```bash
xdg-open web/gomoku_7x7_cpu.html     # Linux
open     web/gomoku_7x7_cpu.html     # macOS
start    web\gomoku_7x7_cpu.html     # Windows
```

別のモデルを変換する場合:

```bash
python export_web.py --model models/gomoku_7x7_cpu_best.pt
python export_web.py --preset balanced --model models/xxx.pt --out web/xxx.html
```

`--sims` でページを開いた直後の思考回数（既定 50）を変えられます。
画面上でも 10〜400 回の範囲で切り替えられます。

## 画面でできること

- 先手（黒）／後手（白）を選んで対局
- 思考の強さ（MCTS のシミュレーション回数）の変更
- 「待った」で1手戻す
- AI の勝率評価と、候補手の訪問回数・事前確率の表示
- 候補手を盤上に重ねて表示

## 仕組み

`export_web.py` が次の3つをやっています。

1. **BatchNorm を畳み込みに合成する（fold）**
   推論時の BN はチャンネルごとの一次式なので、`bias=False` の Conv と数学的に合成できます。

   ```
   scale = gamma / sqrt(var + eps)
   W' = W * scale
   b' = beta - mean * scale
   ```

   これで JavaScript 側は Conv と ReLU だけ実装すればよくなり、結果は厳密に一致します。

2. **重みを float32 の1本のバッファにまとめ、base64 で HTML に埋め込む**
   外部ファイルを読みに行かないので、`file://` で開いても動きます
   （`fetch` は `file://` では使えないため、埋め込みにしています）。

3. **検証用の参照出力を一緒に埋め込む**
   PyTorch で計算した数局面分の policy / value を持たせ、ページを開いた瞬間に
   JavaScript 側で再計算して突き合わせます。結果はページ上部に出ます。

推論と MCTS は `web/template.html` の中で JavaScript に再実装しています。
Python 側と同じ式（PUCT = `-Q + cpuct * p * sqrt(親の訪問回数 - 1) / (1 + 子の訪問回数)`、
評価時は温度0＝訪問回数の argmax）を使っています。

`AlphaZeroNetwork.body()` が

```python
for i in range(self.CFG.n_residual_block):
    x = self.resnet(x)
```

となっており、`self.resnet` は n 個のブロックの `Sequential` なので、
ブロックの適用回数は n 回ではなく **n×n 回**です。JavaScript 側もこれに合わせています。

## 正しく移植できているかの検証

```bash
pip install playwright
python web/verify_web.py
```

実際のブラウザで HTML を開き、次の3点を確認します。

1. **自己診断** — 推論結果が PyTorch と一致するか
2. **MCTS の一致** — 同じ局面・同じ探索回数で Python 版 MCTS を回し、
   選んだ手だけでなく**訪問回数まで**一致するかを突き合わせる
3. **1局の完走** — ランダムに着手して終局まで進み、勝敗判定が出るか

実行結果の例:

```
1. 自己診断 : 自己診断 OK — PyTorch の出力と一致しています
              (6局面, policy 最大誤差 4.9e-7, value 最大誤差 1.2e-6, 1手の推論 約70ms)

2. MCTS の一致（Python と訪問回数まで比較）
   即勝ち(黒番)        JS= 7 Python= 7  訪問一致=はい  -> OK
   防御(白番)         JS=24 Python=24  訪問一致=はい  -> OK
   序盤(黒番)         JS=45 Python=45  訪問一致=はい  -> OK

3. 1局の完走: AI の勝ちです -> OK

判定: PASS
```

ディリクレノイズを切れば両者とも決定的なので、実装が同じなら訪問回数まで一致します。
誤差が float32 の丸め（1e-6 程度）に収まっていることが移植の正しさの根拠です。

環境によっては chromium の場所を指定する必要があります。

```bash
python web/verify_web.py --chrome /path/to/chrome
```

## 注意

- **このモデルは強くありません。** 60イテレーション（自己対局600局）しか学習して
  いないため、人手アルゴリズムには全敗しています（`results/proof_report.md` 参照）。
  相手のリーチを見落とすことがありますが、これは移植の不具合ではなくモデルの弱さで、
  Python 版も同じ手を打ちます。上の検証2はその点を「両者が一致するか」だけで判定しています。
- 探索木は1手ごとに作り直します。Python 側（`SelfPlay` / `arena`）は前の手の部分木を
  再利用するため、同じシミュレーション回数でも読みの量はブラウザ版の方がやや少なくなります。
- 1回の推論は約 60〜90ms（環境依存）です。シミュレーション回数を上げるとその分だけ
  時間がかかります（100回で約5秒）。
