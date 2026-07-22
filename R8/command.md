# R8 コマンド一覧（表示パターン × コマンド）

実験時にコピーして使う用。詳細説明は [`README.md`](README.md) を参照。

パスはリポジトリルートからの相対パス。表示コマンドは `R8\make_movie` で実行。

---

## 0. 事前準備（共通）

### 画像生成

```powershell
python R8/make_movie/gen_assets.py --images rice,nagaoka_fireworks,hocho,ex --intensities 4,8,12 --channels R,G,B,max,min --clip-margin 12
```

### プレゼンターのビルド

```powershell
cd R8\make_movie
gcc present_session.c -I"..\..\vcpkg\installed\x64-mingw-dynamic\include" -L"..\..\vcpkg\installed\x64-mingw-dynamic\lib" -lglfw3dll -lwinmm -lopengl32 -lgdi32 -luser32 -o present_session.exe
```

### 撮影動画の命名

`r{rate}_e{exp}_f{fluoro}.mp4`

| 記号 | 値 |
|---|---|
| rate | `45` / `60` / `90` / `120` / `180` |
| exp | `250` / `125` / `60`（1/250 など） |
| fluoro | `1`=蛍光灯あり / `0`=なし |

例: `r180_e250_f1.mp4`

---

## 1. 表示パターン（present_session）

### 1.1 OSリフレッシュと interval の対応

| 表示パターン | OSディスプレイ設定 | `--rate` | `--interval` | 実効 fps |
|---|---|---|---|---|
| 180 Hz | **180 Hz** | `180` | `1` | ≈180 |
| 120 Hz | **120 Hz**（OS切替必須） | `120` | `1` | ≈120 |
| 90 Hz | 180 Hz のまま | `90` | `2` | ≈90 |
| 60 Hz | 180 Hz のまま | `60` | `3` | ≈60 |
| 45 Hz | 180 Hz のまま | `45` | `4` | ≈45 |

撮影側で毎回合わせるもの（CLIには出ない）:

- 蛍光灯 ON/OFF → 動画名 `f1` / `f0`
- 露光 `1/250` `1/125` `1/60` → `--exp` と動画名 `e250` 等

共通オプション（下表のコマンドに既定で入れてある）:

| オプション | 既定 | 意味 |
|---|---|---|
| `--block-sec` | `6` | 1条件の秒数 |
| `--slate-sec` | `0.5` | 黒/赤スレート各秒数 |
| `--padding-sec` | `5` | 同期前後の黒余白 |
| `--qr-sec` | `3` | GT用QR表示秒数 |
| `--repeat` | `1` | normal/inv のフレーム反復 |

表示順: 黒5s → 黒/赤0.5s → 黒5s → QR3s → 条件0–29 → QR3s → 条件30–59 → QR3s

---

### 1.2 180 Hz（OS=180Hz, interval=1）

**パターン:** 180Hz / 露光1/250 / 蛍光灯あり → 動画 `r180_e250_f1.mp4`

```powershell
cd R8\make_movie
.\present_session.exe --rate 180 --exp 250 --interval 1 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r180_e250.json
```

**パターン:** 180Hz / 露光1/250 / 蛍光灯なし → `r180_e250_f0.mp4`  
（表示コマンドは同じ。撮影時に蛍光灯OFF・ファイル名だけ変える）

**パターン:** 180Hz / 露光1/125 / 蛍光灯あり → `r180_e125_f1.mp4`

```powershell
.\present_session.exe --rate 180 --exp 125 --interval 1 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r180_e125.json
```

**パターン:** 180Hz / 露光1/60 / 蛍光灯あり → `r180_e60_f1.mp4`

```powershell
.\present_session.exe --rate 180 --exp 60 --interval 1 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r180_e60.json
```

---

### 1.3 120 Hz（OSを120Hzに切替, interval=1）

**パターン:** 120Hz / 露光1/250 / 蛍光灯あり → `r120_e250_f1.mp4`

```powershell
cd R8\make_movie
.\present_session.exe --rate 120 --exp 250 --interval 1 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r120_e250.json
```

**パターン:** 120Hz / 露光1/125 → `r120_e125_f*.mp4`

```powershell
.\present_session.exe --rate 120 --exp 125 --interval 1 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r120_e125.json
```

**パターン:** 120Hz / 露光1/60 → `r120_e60_f*.mp4`

```powershell
.\present_session.exe --rate 120 --exp 60 --interval 1 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r120_e60.json
```

---

### 1.4 90 Hz（OS=180Hzのまま, interval=2）

**パターン:** 90Hz / 露光1/250 → `r90_e250_f*.mp4`

```powershell
cd R8\make_movie
.\present_session.exe --rate 90 --exp 250 --interval 2 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r90_e250.json
```

**パターン:** 90Hz / 露光1/125 → `r90_e125_f*.mp4`

```powershell
.\present_session.exe --rate 90 --exp 125 --interval 2 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r90_e125.json
```

**パターン:** 90Hz / 露光1/60 → `r90_e60_f*.mp4`

```powershell
.\present_session.exe --rate 90 --exp 60 --interval 2 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r90_e60.json
```

---

### 1.5 60 Hz（OS=180Hzのまま, interval=3）

**パターン:** 60Hz / 露光1/250 → `r60_e250_f*.mp4`

```powershell
cd R8\make_movie
.\present_session.exe --rate 60 --exp 250 --interval 3 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r60_e250.json
```

**パターン:** 60Hz / 露光1/125 → `r60_e125_f*.mp4`

```powershell
.\present_session.exe --rate 60 --exp 125 --interval 3 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r60_e125.json
```

**パターン:** 60Hz / 露光1/60 → `r60_e60_f*.mp4`

```powershell
.\present_session.exe --rate 60 --exp 60 --interval 3 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r60_e60.json
```

---

### 1.6 45 Hz（OS=180Hzのまま, interval=4）

**パターン:** 45Hz / 露光1/250 → `r45_e250_f*.mp4`

```powershell
cd R8\make_movie
.\present_session.exe --rate 45 --exp 250 --interval 4 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r45_e250.json
```

**パターン:** 45Hz / 露光1/125 → `r45_e125_f*.mp4`

```powershell
.\present_session.exe --rate 45 --exp 125 --interval 4 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r45_e125.json
```

**パターン:** 45Hz / 露光1/60 → `r45_e60_f*.mp4`

```powershell
.\present_session.exe --rate 45 --exp 60 --interval 4 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r45_e60.json
```

---

## 2. 解析パターン（run_pipeline）

リポジトリルートで実行。下表は動画例 `r180_e250_f1.mp4` / manifest `r180_e250.json`。  
他レート・露光ではパスだけ差し替える（例: `r60_e125_f0.mp4` + `manifests\r60_e125.json`）。

### 2.1 差分モードと既定スイープ

| 差分パターン | `--diff-mode` | 既定スイープ |
|---|---|---|
| 隣接ペア（既定） | `pair` | th = 4, 8, 12 |
| 窓合算 | `accum` | n = 3,5 × th = 12,16,24,32 |
| 統計（時間方向） | `stat` | std: th = 4,8,12 / var: th = 1,2,4 |
| 時間軸FFT | `fourier` | 第一候補+半分の2周波数 × th = 4,8,12 |

### 2.2 QR探索モード

| 探索パターン | フラグ | 内容 |
|---|---|---|
| fast（既定・検証用） | （なし） | gray × kernel5 × 最精度デコード1回（`detectAndDecodeMulti`） |
| mid | `--mid-search` | gray + median_otsu、kernel 3/5/7、cascade+Multi |
| full | `--full-search` | 全バリアント + 拡大1/2/3 + cascade+Multi |

### 2.3 残すフレーム枚数

| 保存パターン | `--max-frames` | 意味 |
|---|---|---|
| 全部残す（既定） | `120` | 再デコード・目視向け |
| 先頭+末尾だけ残す | `2` | まずは端点だけ確認したい（差分計算は常に120枚分） |
| 1枚だけ残す | `1` | ディスク節約（差分計算は常に120枚分） |

---

### 2.4 pair（標準）

`pair` は **先頭20＋末尾20ペア（最大40）** だけ差分生成・デコードします。保存する差分PNGはさらに **成功ペア + 最高accuracyペア + 10枚に1枚** に間引かれます。

**パターン:** pair + fast + 120枚残す（いちばん標準）

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --max-frames 120
```

**パターン:** pair + mid + 120枚残す

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --max-frames 120 --mid-search
```

**パターン:** pair + full + 120枚残す

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --max-frames 120 --full-search
```

**パターン:** pair + fast + 1枚だけ残す（本番一括・節約）

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --max-frames 1
```

**パターン:** pair + mid + 1枚だけ残す

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --max-frames 1 --mid-search
```

**パターン:** pair + fast + 先頭+末尾2枚だけ残す

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --max-frames 2
```

**パターン:** pair の閾値スイープを上書き（例: 8 と 12 だけ）

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode pair --diff-thresholds 8,12
```

---

### 2.5 accum（窓合算）

**パターン:** accum + fast + 既定スイープ（n=3,5 × th=12,16,24,32）

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode accum
```

**パターン:** accum + mid

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode accum --mid-search
```

**パターン:** accum + full

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode accum --full-search
```

**パターン:** accum + 1枚だけ残す

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode accum --max-frames 1
```

**パターン:** accum スイープを絞る（n=4 だけ × th=16,24,32）

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode accum --window-ns 4 --diff-thresholds 16,24,32
```

**パターン:** accum スイープを絞る（n=3,5 × th=12,24）

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode accum --window-ns 3,5 --diff-thresholds 12,24
```

**パターン:** accum + mid + スイープ絞り + 節約

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode accum --window-ns 4 --diff-thresholds 16,24,32 --mid-search --max-frames 1
```

---

### 2.6 stat（統計）

**パターン:** stat(std) + fast + 既定スイープ

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode stat
```

**パターン:** stat(var) + fast + 既定スイープ

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode stat --stat-kind var
```

**パターン:** stat(std) + mid

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode stat --mid-search
```

**パターン:** stat(var) + 閾値スイープ上書き

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode stat --stat-kind var --diff-thresholds 1,3,5
```

---

### 2.7 fourier（時間軸FFT）

ターゲット周波数は動画名の `rate_hz` とカメラ fps から **第一候補 + その半分** を自動算出（例: r180→30,15 Hz / r45→22.5,11.25 Hz / r120→30,15 Hz）。

**パターン:** fourier + mid + 既定スイープ（2周波数 × th=4,8,12）

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode fourier --mid-search
```

**パターン:** fourier + 先頭+末尾2枚だけ残す

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode fourier --mid-search --max-frames 2
```

**パターン:** fourier + 周波数手動指定（r90 → 15 Hz と 7.5 Hz）

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r90_e250_f1.mp4 --manifest R8/make_movie/manifests/r90_e250.json --diff-mode fourier --target-freqs 15,7.5 --mid-search
```

**パターン:** fourier + 閾値スイープ上書き

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode fourier --diff-thresholds 4,12
```

---

### 2.8 他動画への差し替え例

**パターン:** 60Hz / 露光1/125 / 蛍光灯なし / pair+fast

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r60_e125_f0.mp4 --manifest R8/make_movie/manifests/r60_e125.json --max-frames 120
```

**パターン:** 120Hz / 露光1/250 / 蛍光灯あり / accum+mid

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r120_e250_f1.mp4 --manifest R8/make_movie/manifests/r120_e250.json --diff-mode accum --mid-search
```

---

## 3. 単体ツール（切り出し済みフォルダ向け）

### 3.1 差分だけ作り直す

**パターン:** pair / th=8

```bash
python R8/analyze_code/cal-from-2frame-RGB-oute.py --base-dir R8/analyze_code/out/r180_e250_f1/rice_R_4 --diff-mode pair --threshold 8 --output-subdir rgb_max_diff_maps_th8
```

**パターン:** accum / n=4 / th=16

```bash
python R8/analyze_code/cal-from-2frame-RGB-oute.py --base-dir R8/analyze_code/out/r180_e250_f1/rice_R_4 --diff-mode accum --window-n 4 --threshold 16 --output-subdir rgb_max_accum_n4_th16
```

**パターン:** stat(std) / th=4

```bash
python R8/analyze_code/cal-from-2frame-RGB-oute.py --base-dir R8/analyze_code/out/r180_e250_f1/rice_R_4 --diff-mode stat --stat-kind std --threshold 4 --output-subdir rgb_max_stat_std_th4
```

**パターン:** stat(var) / th=1

```bash
python R8/analyze_code/cal-from-2frame-RGB-oute.py --base-dir R8/analyze_code/out/r180_e250_f1/rice_R_4 --diff-mode stat --stat-kind var --threshold 1 --output-subdir rgb_max_stat_var_th1
```

**パターン:** fourier / 30Hz / th=8

```bash
python R8/analyze_code/cal-from-2frame-RGB-oute.py --base-dir R8/analyze_code/out/r180_e250_f1/rice_R_4 --diff-mode fourier --fps 59.94 --target-freq 30 --threshold 8 --output-subdir rgb_max_fourier_f30_th8
```

### 3.2 デコードだけやり直す

**パターン:** 1条件を mid で再デコード（pair 差分 dir）

```bash
python R8/analyze_code/decode_qr_from_all_frames.py --base-dir R8/analyze_code/out/r180_e250_f1 --folder rice_R_4 --diff-subdir rgb_max_diff_maps_th8 --mid-search
```

**パターン:** accum 差分 dir を mid で再デコード

```bash
python R8/analyze_code/decode_qr_from_all_frames.py --base-dir R8/analyze_code/out/r180_e250_f1 --folder rice_R_4 --diff-subdir rgb_max_accum_n4_th16 --mid-search
```

**パターン:** full 探索

```bash
python R8/analyze_code/decode_qr_from_all_frames.py --base-dir R8/analyze_code/out/r180_e250_f1 --folder rice_R_4 --full-search
```

---

## 4. 早見表（組み合わせ）

### 表示: rate × interval

| rate | OS Hz | interval | manifest 例 |
|---|---|---|---|
| 180 | 180 | 1 | `manifests\r180_e250.json` |
| 120 | 120 | 1 | `manifests\r120_e250.json` |
| 90 | 180 | 2 | `manifests\r90_e250.json` |
| 60 | 180 | 3 | `manifests\r60_e250.json` |
| 45 | 180 | 4 | `manifests\r45_e250.json` |

`--exp` は `250` / `125` / `60`。manifest 名は `r{rate}_e{exp}.json`。

### 解析: よく使うセット

| 用途 | コマンド要点 |
|---|---|
| 標準解析 | （フラグなし）= pair + fast + max-frames 120 |
| 読み取り強化 | `--mid-search` または `--full-search`（前処理増。デコーダは ZXing 優先のまま） |
| ログ要約 | `--quiet`（対象・モード・結果のみ。一括実行では自動付与） |
| GPU | `pip install torch`（CUDA版推奨）。差分/統計/FFT が自動で GPU 利用 |
| 窓合算 | `--diff-mode accum` |
| 統計解析 | `--diff-mode stat`（`--stat-kind var` で分散） |
| 時間軸FFT | `--diff-mode fourier`（`--target-freqs 15,7.5` で周波数上書き） |
| accum を軽く | `--diff-mode accum --window-ns 4 --diff-thresholds 16,24,32` |
| ディスク節約 | `--max-frames 1` |
| 端点確認 | `--max-frames 2` |

`pair` の差分PNGは既定で間引き保存されます。デコード対象も先頭20＋末尾20ペア（最大40）に限定されています。

出力先: `R8/analyze_code/out/<動画stem>/`  
（`results.csv`, `qr_decode_all_frames.csv`, 条件フォルダ, `rgb_max_diff_maps_th*` / `rgb_max_accum_n*_th*`）

---

## 4. 一括解析（all_analyze）

`R8/movie` 内の命名規則に合う動画を順番に解析します。1本失敗しても続行します。

```bash
python R8/analyze_code/all_analyze.py
```

**既定:** 各動画について次の **全5手法** を順に実行します。

1. `pair`
2. `accum`
3. `stat`（std）
4. `stat`（var）
5. `fourier`

共通: **fast** + `--max-frames 120` + `--workers 4` + **切り出し再利用**  
動画順は rate → exp → fluoro。2手法目以降は切り出しを再利用します。  
出力は手法別に `results_pair.csv` / `results_accum.csv` などとして残ります。  
`--diff-mode ...` を付けた場合は、その手法だけ実行します。

**ログ:** ジョブごとに `target` / `pass` / `result` の3行のみ。詳細はキャプチャし、失敗時だけ末尾を表示。単体 `run_pipeline` は詳細ログ、`--quiet` で要約のみ。

**例:**

```bash
# 全手法 × 全動画（既定）
python R8/analyze_code/all_analyze.py

# 全手法だが端点2枚指定（再利用中は120枚を残す）
python R8/analyze_code/all_analyze.py --max-frames 2

# accum だけ
python R8/analyze_code/all_analyze.py --diff-mode accum

# fourier だけ
python R8/analyze_code/all_analyze.py --diff-mode fourier --max-frames 2

# mid に切り替えて全手法
python R8/analyze_code/all_analyze.py --mid-search

# 切り出しを強制やり直し
python R8/analyze_code/all_analyze.py --force-extract
```

サマリー: `R8/analyze_code/out/all_analyze_summary.csv`（`video`, `pass`, `manifest`, `status`, `exit_code`, `note`）  
所要時間: `R8/analyze_code/out/all_analyze_timing.csv`（追記蓄積。`finished_at`, `video`, `pass`, `status`, `elapsed_sec`）
