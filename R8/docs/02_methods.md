# 02. 差分マップ各手法（mid_hully の results 単位）

共通入力は隣接フレームの max-channel 差分系列 \(d(t) \in \mathbb{R}^{T-1 \times H \times W}\) である。  
各手法は \(d(t)\) から「変化が強い画素」の2Dマップを作り、それを QR としてデコードする。

この文書は **主解析 mid**（`all_analyze_mid.py` → `run_pipeline_hully.py --mid-sweeps`）で出る **`results_<pass>.csv` ごと** に整理する。

実装: [`hully_diff.py`](../analyze_code/hully_diff.py), [`time_fft.py`](../analyze_code/time_fft.py), [`gpu_ops.py`](../analyze_code/gpu_ops.py), [`run_pipeline_hully.py`](../analyze_code/run_pipeline_hully.py)

---

## mid_hully で出る results（10 pass）

| # | `results_*.csv` の pass | 中身 | 表現 |
|---|---|---|---|
| 1 | `results_pair.csv` | 隣接差分 | binary（閾値あり） |
| 2 | `results_accum.csv` | 窓積算 | binary |
| 3 | `results_stat_std.csv` | 時間標準偏差 | binary |
| 4 | `results_stat_var.csv` | 時間分散 | binary |
| 5 | `results_lockin.csv` | ロックイン振幅 | binary |
| 6 | `results_fourier.csv` | 時間FFTスコア | binary |
| 7 | `results_accum_num.csv` | 窓積算 | **gray（num）** |
| 8 | `results_stat_std_num.csv` | 時間標準偏差 | **gray（num）** |
| 9 | `results_lockin_num.csv` | ロックイン振幅 | **gray（num）** |
| 10 | `results_fourier_num.csv` | 時間FFTスコア | **gray（num）** |

対応する詳細行は `results_sweeps_<pass>.csv`。  
**`pair_num` と `stat_var_num` は無い**（pair は数値スコア経路なし、`stat_var_num` は実績ほぼゼロのため除外）。

---

## binary と gray（num）の違い

| | **binary**（`pair`, `accum`, …） | **gray / num**（`accum_num`, …） |
|---|---|---|
| スコアの扱い | 固定閾値 `th` で「変化あり／なし」に切る | 連続スコアを画像内で 0..1 正規化 |
| 見た目 | 変化あり→黒、なし→白の二値 | 高スコア＝黒の**グレースケール**（`score_to_gray_rgb`） |
| スイープの `th` | 複数候補を総当たり | 実質 `th=0` 固定（閾値スイープしない） |
| デコード側 | 二値マップを読む | 濃淡マップを読む（mid-search の gray / median_otsu などと相性が良い想定） |
| ねらい | 閾値が合えばコントラストがはっきりする | **閾値ミスで信号が消える／背景が残る**のを緩和する |

正規化はマップ全体の min–max（`normalize_score_map`）。そのため「絶対スケール」ではなく「その条件・その手法内での相対的な強さ」になる。

フォルダ名の目安:

- binary: `..._th8/` のように閾値が付く
- gray: `..._gray/`（閾値ラベルなし）

---

## mid 共通設定（全 pass）

`all_analyze_mid.py` が付け外しする既定と、`run_pipeline_hully.py` の `--mid-sweeps` 定数。

| 項目 | mid の値 | 備考 |
|---|---|---|
| プロファイル | `--mid-sweeps` | 主解析 |
| QR探索 | `--mid-search` | バリアント `gray` + `median_otsu`、median kernel **3 / 5 / 7**、拡大なし（scale=1） |
| 切り出し | 条件ブロックの 2〜4 秒 → **120フレーム** | |
| 生フレーム | `--keep-frames 0` | d(t) 後に PNG 削除 |
| 差分PNG保存 | `--save-diff-maps per_pass` | 手法ごと代表1枚 |
| 周波数候補 | `resolve_target_freqs_hard` | rate/2・半分に加え rate/4・3rate/4（エイリアス折り畳み）。2 Hz 未満は除外 |
| fourier 帯域 | `MID_FOURIER_BAND_RADIUS = 2` | 目標ビンの前後幅 |
| 採用規則 | 条件内で `pixel_acc_all` 最大 | 同点はデコード成功 → 小さい th → 小さい n → 第一候補周波数／位相 |

定数の定義場所: [`run_pipeline_hully.py`](../analyze_code/run_pipeline_hully.py) の `MID_*`。

---

## 1. `results_pair.csv` — pair（隣接差分・binary）

### 狙い

点滅の**瞬間の切り替わり**を、隣り合うフレーム差分そのもので捉えるベースライン。

### 処理

1. \(d(t)\) の各隣接ペア（実際に使うのは先頭・末尾のみ）を取る  
2. `|d| > th/255` なら黒（増減どちらも変化あり）、それ以外白  

### mid パラメータ

| 項目 | 値 |
|---|---|
| `pair_each_end` | **20**（先頭20＋末尾20、最大40ペア） |
| `th` スイープ | **4, 6, 8, 10, 12** |
| gray / num | なし |

### 出力例

- CSV: `results_pair.csv`, `results_sweeps_pair.csv`, `pair_accuracy.csv`
- マップ: `rgb_max_diff_maps_th8/`

### メモ

テクスチャが強いと背景の微変動にも反応しやすい。閾値スイープの解釈はしやすい。

---

## 2. `results_accum.csv` — accum（窓積算・binary）

### 狙い

単一ペアのノイズを抑え、**短い非重複窓で \|d\| を積算**してコントラストを稼ぐ。

### 処理

1. \(d(t)\) を長さ `n` の窓に分割（`1..n`, `n+1..2n`, …）  
2. 窓内で `|d|` を合算  
3. 積算マップを `th` で二値化  

### mid パラメータ

| 項目 | 値 |
|---|---|
| `window_n` | **3, 5** |
| `th` スイープ | **8, 12, 16, 20, 24, 32** |

### 出力例

- CSV: `results_accum.csv`, `results_sweeps_accum.csv`
- マップ: `rgb_max_accum_n5_th16/`

---

## 3. `results_stat_std.csv` — stat_std（標準偏差・binary）

### 狙い

「どのフレームか」ではなく、画素ごとの時間系列の**ばらつき（標準偏差）が大きい場所**を信号とみなす。

### std とは（var との違い）

画素 \(p\) について時間方向の差分系列を \(d_p(t)\) とする。

| | **std**（この pass） | **var**（次の pass） |
|---|---|---|
| 定義 | \(\mathrm{std}_t\, d_p(t)\) | \(\mathrm{var}_t\, d_p(t) = (\mathrm{std})^2\) |
| スケール | 差分と同じオーダー | 二乗なので数値が小さく・差が伸びる |
| mid の `th` | **4, 8, 12**（比較的大きめ） | **1, 2**（小さめ） |
| 解釈 | 「揺れの大きさ」そのもの | 同じ順位付けになりやすいが、閾値の効き方が違う |

実装はどちらも `temporal_std_var`（時間軸の std / var）。周波数は見ないので、蛍光灯など**別周期の変動**にも反応しうる。

### mid パラメータ

| 項目 | 値 |
|---|---|
| `stat_kind` | `std` |
| `th` スイープ | **4, 8, 12** |
| マップ枚数 | 条件あたり原則1枚（全120フレーム差分から1枚の統計マップ） |

### 出力例

- CSV: `results_stat_std.csv`, `results_sweeps_stat_std.csv`
- マップ: `rgb_max_stat_std_th8/`

---

## 4. `results_stat_var.csv` — stat_var（分散・binary）

### 狙い

stat_std と同じ「時間ばらつき」だが、**分散**でスコア化する。強い揺れをより強調し、弱い揺れを相対的に潰しやすい（二乗のため）。

### mid パラメータ

| 項目 | 値 |
|---|---|
| `stat_kind` | `var` |
| `th` スイープ | **1, 2** |
| gray / num | **なし**（`stat_var_num` は未採用） |

### 出力例

- CSV: `results_stat_var.csv`, `results_sweeps_stat_var.csv`
- マップ: `rgb_max_stat_var_th1/`

### メモ

binary の `stat_var` は残しているが、gray 版は実績ほぼゼロのため mid の10 pass に入れていない。std と結果が似る条件も多く、報告では「ばらつき系の別閾値スケール」として並記するとよい。

---

## 5. `results_lockin.csv` — lockin（ロックイン・binary）

### 狙い

想定点滅周波数の正弦／余弦と \(d(t)\) を掛け、**その周波数の振幅が大きい画素**を強調する。pair より周波数選択的。

### 処理（簡潔）

目標周波数 \(f\)、位相を `phase_steps` 分割。各位相 \(\phi\) で

\[
I=\sum_t d(t)\cos(2\pi ft+\phi),\quad
Q=\sum_t d(t)\sin(2\pi ft+\phi),\quad
A=\sqrt{I^2+Q^2}
\]

位相スイープの最大 \(A\) をスコアマップにし、`th` で二値化（`lockin_score_map`）。

### mid パラメータ

| 項目 | 値 |
|---|---|
| `phase_steps` | **4, 8, 16** |
| `th` スイープ | **4, 6, 8, 10, 12, 16** |
| 周波数候補 | hard セット（上記 mid 共通） |

### 出力例

- CSV: `results_lockin.csv`, `results_sweeps_lockin.csv`
- マップ: `rgb_max_lockin_f30_ps8_th8/`

### メモ

高テクスチャで pair が落ちるときに相対的に効くことがある。周波数×位相×th でスイープ量が大きい。

---

## 6. `results_fourier.csv` — fourier（時間FFT・binary）

### 狙い

画素ごとの時間系列を FFT し、**目標周波数近傍のスペクトル強度**をスコアにする。lockin と同様に周波数選択的だが、ビン近傍（帯域）で拾う。

### 処理

1. detrend、Hann 窓  
2. 時間軸 FFT  
3. 目標近傍（`band_radius=2`）の強度／ノイズ床スコア  
4. `th` で二値化  

### mid パラメータ

| 項目 | 値 |
|---|---|
| `fourier_band_radius` | **2** |
| `th` スイープ | **4, 6, 8, 10, 12, 16** |
| 周波数候補 | hard セット（rate/2, 半分, rate/4, 3rate/4 折り畳み） |

ラベル例: 30 Hz → `f30`、22.5 Hz → `f225`。

### 出力例

- CSV: `results_fourier.csv`, `results_sweeps_fourier.csv`
- マップ: `rgb_max_fourier_f30_th8/`

---

## 7. `results_accum_num.csv` — accum_num（窓積算・gray）

### 狙い

accum と同じ窓積算スコアを、**閾値二値化せずグレースケール**にしてデコードする。

### mid パラメータ

| 項目 | 値 |
|---|---|
| `window_n` | **3, 5**（binary accum と同じ） |
| `th` | 固定相当（スイープしない） |
| 表現 | gray |

### 出力例

- CSV: `results_accum_num.csv`, `results_sweeps_accum_num.csv`
- マップ: `rgb_max_accum_n5_gray/`

---

## 8. `results_stat_std_num.csv` — stat_std_num（標準偏差・gray）

### 狙い

時間 std マップを gray 化し、固定 `th` に依存せず読む。

### mid パラメータ

| 項目 | 値 |
|---|---|
| `stat_kind` | `std` |
| `th` | 固定相当 |
| 表現 | gray |
| 対応する var 版 | なし（`stat_var_num` 未採用） |

### 出力例

- CSV: `results_stat_std_num.csv`, `results_sweeps_stat_std_num.csv`
- マップ: `rgb_max_stat_std_gray/`

---

## 9. `results_lockin_num.csv` — lockin_num（ロックイン・gray）

### 狙い

ロックイン振幅マップを gray 化。位相・周波数スイープは binary lockin と同様に行い、最後の二値化だけ省略する。

### mid パラメータ

| 項目 | 値 |
|---|---|
| `phase_steps` | **4, 8, 16** |
| 周波数候補 | hard セット |
| `th` | 固定相当 |
| 表現 | gray |

### 出力例

- CSV: `results_lockin_num.csv`, `results_sweeps_lockin_num.csv`
- マップ: `rgb_max_lockin_f30_ps8_gray/`

---

## 10. `results_fourier_num.csv` — fourier_num（時間FFT・gray）

### 狙い

FFT スコアマップを gray 化。

### mid パラメータ

| 項目 | 値 |
|---|---|
| `fourier_band_radius` | **2** |
| 周波数候補 | hard セット |
| `th` | 固定相当 |
| 表現 | gray |

### 出力例

- CSV: `results_fourier_num.csv`, `results_sweeps_fourier_num.csv`
- マップ: `rgb_max_fourier_f30_gray/`

---

## mid パラメータ一覧（pass 横断表）

| pass | 表現 | 主スイープ軸（mid） |
|---|---|---|
| pair | binary | ペア±20、th = 4,6,8,10,12 |
| accum | binary | n = 3,5 × th = 8,12,16,20,24,32 |
| stat_std | binary | th = 4,8,12 |
| stat_var | binary | th = 1,2 |
| lockin | binary | freq(hard) × phase = 4,8,16 × th = 4,6,8,10,12,16 |
| fourier | binary | freq(hard) × band=2 × th = 4,6,8,10,12,16（※ mid の旧 th/255。`mid_fast` は下記） |

### mid_fast の lockin / fourier binary（正規化スコア向け）

`all_analyze_mid_fast`（`--mid-fast-sweeps`）では、壊れていた `th/255` スケールをやめ、次をスイープする。

| `diff_threshold` | 意味 | サブディレクトリ例 |
|---|---|---|
| 30 / 50 / 70 | 正規化[0,1]上の固定閾値 0.30 / 0.50 / 0.70 | `..._p50` |
| 901 | Otsu（大域・自動） | `..._otsu` |
| 902 | adaptiveThreshold（局所） | `..._adapt` |

実装: [`hully_diff.binarize_normalized_score_map`](../analyze_code/hully_diff.py)。mid/hard の 4〜16 は互換のため旧 `th/255` のまま。
| accum_num | gray | n = 3,5 |
| stat_std_num | gray | （マップ1枚系） |
| lockin_num | gray | freq(hard) × phase = 4,8,16 |
| fourier_num | gray | freq(hard) × band=2 |

---

## 手法の関係（スライド1枚用）

| pass 系統 | 時間の使い方 | 周波数 | 典型的な強み |
|---|---|---|---|
| pair | 瞬間の隣接差分 | なし | 単純・解釈しやすい |
| accum / accum_num | 短窓積算 | なし | 単発ノイズをならす |
| stat_std / stat_var / stat_std_num | 全時間のばらつき | なし | 実装が単純（std≈揺れ幅、var≈二乗強調） |
| lockin / lockin_num | 参照正弦との相関 | あり | 想定周期に選択的 |
| fourier / fourier_num | スペクトル近傍 | あり | 帯域付き周波数抽出 |
| `*_num` | 親手法と同じスコア | （親に依存） | 固定 th 二値化の失敗を緩和 |

---

## 実装関数対応表

| 内容 | 主な関数 |
|---|---|
| \(d(t)\) 生成 | `gpu_ops.max_channel_difference`, `hully_diff.build_pair_diff_stack` |
| pair | `hully_diff.generate_pair_maps` |
| accum / accum_num | `hully_diff.generate_accum_maps` |
| stat_std / stat_var / stat_std_num | `gpu_ops.temporal_std_var`, `hully_diff.generate_stat_maps` |
| lockin / lockin_num | `time_fft.lockin_score_map`, `hully_diff.generate_lockin_maps` |
| fourier / fourier_num | `time_fft.build_score_map`, `hully_diff.generate_fourier_maps` |
| gray 化 | `hully_diff.score_to_gray_rgb`, `time_fft.normalize_score_map` |
| mid スイープ組立 | `run_pipeline_hully.build_sweeps`（profile=`mid`） |
