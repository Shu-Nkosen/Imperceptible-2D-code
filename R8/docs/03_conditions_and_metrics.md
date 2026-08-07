# 03. 実験条件・評価指標・出力

## 動画条件（ファイル名）

形式: **`r{rate}_e{exp}_f{fluoro}`**

| 項目 | 取りうる値 | 意味 |
|---|---|---|
| rate | 45, 60, 90, 120, 180 | 表示点滅レート [Hz] |
| exp | 250, 125, 60 | 露光 1/N 秒 |
| fluoro | 0, 1 | 蛍光灯なし / あり |

解析出力も同じ stem のディレクトリになる（例: `out_mid_fast_0805/r60_e250_f1/`）。

表示レートとモニタ Hz・`--interval` の関係は [`../README.md`](../README.md) の表示セクションを参照。

## 条件フォルダ（60通り）

1動画内の埋め込み条件は次の直積。

| 軸 | 値 | フォルダでの表記 |
|---|---|---|
| 画像 | rice, nagaoka_fireworks, hocho, ex | そのまま |
| チャネル | R, G, B, min, max | R / G / B / **I** / **X** |
| 強度 | 4, 8, 12 | そのまま |

例: `rice_R_4`, `hocho_B_12`, `ex_X_8`

生成時の `clip_margin`（例: 12）は埋め込み画像作成側のパラメータで、解析パイプライン本体のスイープ軸ではない。画像×チャネル調査では定数として参照される。

## スイープと採用規則

各 pass（手法）について、条件ごとに閾値・窓長・周波数・位相などを総当たりする。

### 採用の主基準

条件内で **`pixel_acc_all` が最大**のスイープを `results_<pass>.csv` に1行採用する。

### 同点時の優先（概略）

1. デコード成功を優先
2. より小さい `th`
3. より小さい窓長 `n`（accum）
4. 第一候補の周波数 / 位相

デコード失敗でも、GT があれば `pixel_acc_*` は計算でき、同じ規則で「採用スイープ」は決まる。

### 傾向分析の本命

可視化や成功率の分解は、採用1行だけでなく **`results_sweeps_<pass>.csv`**（条件×スイープ全行）を縦結合して見る。

- 条件あたり「どれか1スイープでも成功」: `groupby(...).max()` on `decode_success`
- スイープ平均成功率: `groupby(...).mean()` on `decode_success`

## 評価指標（二系統）

### A. QRデコード成功（実験の主指標）

| 場所 | 見方 |
|---|---|
| `results_sweeps_*.csv` | `decode_success` が `1` / `0` |
| `results_*.csv` | `decode_note` が空かつ `decode_decoded_text` が非空なら成功 |
| `qr_decode_all_frames_*.csv` | マップ単位の `success` など |

読取は ZXing（zxing-cpp）優先、失敗時に OpenCV。探索の広さは fast / mid / full。

### B. GT画素 accuracy（採用・副指標）

GT は動画内 QR 表示から作った `frame_QR.png`。

- 黒マスク閾値や ROI（高さ比など）で比較領域を制限
- 検出マップの「暗がQRか／明がQRか」を両方試し、合う方を採用
- `accuracy = (TP + TN) / total`（ROI内）

| 列 | 意味 |
|---|---|
| `pixel_acc_all` | そのスイープで生成した全ペア／窓の accuracy **平均** |
| `pixel_acc_ok` | デコード成功したものだけの平均 |
| `pixel_acc_best` | 全ペア／窓のうち **最大** |

## 主な出力ファイル

1動画（例: `out_mid_MMDD/r60_e250_f1/`）あたり:

| ファイル | 粒度 | 用途 |
|---|---|---|
| `results_<pass>.csv` | 条件1行＝採用スイープ | ざっくり可視化・表 |
| `results_sweeps_<pass>.csv` | 条件×スイープ | 傾向分析の本命（`adopted`, `decode_success`） |
| `qr_decode_all_frames_<pass>.csv` | マップ単位 | デコード詳細 |
| `pair_accuracy.csv` | pair・採用thのペア別 | 先頭/末尾ペアの正解率 |
| `hully_timing.txt` / `logs/` | 実行ログ | 所要時間・再現 |

一括実行時は親ディレクトリに `all_analyze_*_summary.csv` や timing CSV も出ることがある。

### `results_sweeps_*.csv` でよく使う列

| 列 | 内容 |
|---|---|
| `folder` / `image` / `channel` / `intensity` | 条件 |
| `pass` / `diff_mode` | 手法 |
| `diff_threshold` / `window_n` / `fft_target_hz` / `phase_steps` | スイープ軸 |
| `decode_success` | そのスイープで読めたか |
| `pixel_acc_all` | 採用判定に使う平均一致率 |
| `adopted` | `1` = `results_*.csv` に採用された行 |

## 差分サブディレクトリ命名（抜粋）

| 手法 | 例 |
|---|---|
| pair | `rgb_max_diff_maps_th8` |
| accum | `rgb_max_accum_n5_th16` / `..._gray` |
| stat | `rgb_max_stat_std_th8` / `rgb_max_stat_var_th1` |
| lockin | `rgb_max_lockin_f30_ps8_th8` / `..._gray` |
| fourier | `rgb_max_fourier_f30_th8` / `..._gray` |

保存方針（`--save-diff-maps`）により、全スイープを残す／上位だけ残す／残さない、が変わる。

## 既存の結果メモ（参照のみ）

画像×チャネル依存の定性的結論は、解析コード同梱の調査メモを参照:

- [`../analyze_code/out_image_channel_study/FINDINGS.md`](../analyze_code/out_image_channel_study/FINDINGS.md)

要点の一例（詳細・条件範囲は FINDINGS 側）:

- デジタル埋め込み振幅自体は揃っていても、デコード差はテクスチャと支配／従属チャネルに依存する
- 高勾配画像では pair が厳しく、lockin の相対価値が上がりうる

条件ごとの成功率の横断表やスライド用図表は、本ドキュメントのスコープ外（次段）。

## スライド用の一文まとめ

実験軸は「動画の rate/exp/fluoro」と「画像×チャネル×強度の60条件」。手法ごとに閾値等をスイープし、GT一致率最大の設定を採用しつつ、最終的な良し悪しは QR が読めたかで議論する。
