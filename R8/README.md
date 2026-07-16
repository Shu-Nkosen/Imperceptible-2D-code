# R8 sweep (表示→撮影→解析)

**コマンド早見表:** 表示パターンと解析パターンをコマンドとセットでまとめたものは [`command.md`](command.md) を参照。

## 命名規則（動画ファイル）

**`r{rate}_e{exp}_f{fluoro}.mp4`**

- `rate`: `45|60|90|120|180`
- `exp`: `250|125|60`（例: `e250` = 1/250）
- `fluoro`: `f1`=蛍光灯あり / `f0`=なし

例: `r180_e250_f1.mp4`

## 事前準備（画像生成）

`R8/make_movie` 内で実行します。事前に `ex.png` などのベース画像と **`HP_QR.png`** が必要です。

```bash
python R8/make_movie/gen_assets.py --images rice,nagaoka_fireworks,hocho,ex --intensities 4,8,12 --channels R,G,B,max,min --clip-margin 12
```

生成される主なファイル（例）:

- `rice_4_normalR.png` / `rice_4_invR.png`
- `rice_4_normalX.png` / `rice_4_invX.png`（max）
- `rice_4_normalI.png` / `rice_4_invI.png`（min）

## 表示（C/GLFW）

### ビルド

PowerShell で `R8/make_movie` に移動してビルドします（既存のGLFW/vcpkg構成に合わせる）。

```powershell
cd R8\make_movie
gcc present_session.c -I"..\..\vcpkg\installed\x64-mingw-dynamic\include" -L"..\..\vcpkg\installed\x64-mingw-dynamic\lib" -lglfw3dll -lwinmm -lopengl32 -lgdi32 -luser32 -o present_session.exe
```

### 実行

事前に **Windowsのディスプレイ設定でリフレッシュレートを下表のとおり設定**してから実行します（特に120Hzは OS 側の切替が必須）。

実効表示レートは概ね `モニタHz ÷ interval` です。`--rate` はマニフェスト／動画命名用の記録で、実際の点滅は `interval` と OS のリフレッシュで決まります。起動ログの `monitor_hz=... interval=... => fps~...` で確認してください。

#### 表示Hzごとの手動設定

| 実験レート | OSディスプレイ設定 | `--rate` | `--interval` | 実効 fps |
|---|---|---|---|---|
| **180 Hz** | **180 Hz** | `180` | `1` | ≈180 |
| **120 Hz** | **120 Hz**（ここだけ OS を切替） | `120` | `1` | ≈120 |
| **90 Hz** | 180 Hz のまま | `90` | `2` | ≈90 |
| **60 Hz** | 180 Hz のまま | `60` | `3` | ≈60 |
| **45 Hz** | 180 Hz のまま | `45` | `4` | ≈45 |

撮影側で人が毎回合わせるもの（CLIには出さない）:

- **蛍光灯** ON/OFF → 動画名の `f1` / `f0`
- **露光** `1/250` `1/125` `1/60` → `--exp` と動画名の `e250` 等（表示内容自体は変わらないが、マニフェストとファイル名を揃える）

例（180Hz）:

```powershell
.\present_session.exe --rate 180 --exp 250 --interval 1 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r180_e250.json
```

例（60Hz・180Hzモニタのまま）:

```powershell
.\present_session.exe --rate 60 --exp 250 --interval 3 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r60_e250.json
```

例（120Hz・OSを120Hzにしてから）:

```powershell
.\present_session.exe --rate 120 --exp 250 --interval 1 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --qr-sec 3 --repeat 1 --out-manifest manifests\r120_e250.json
```

- `--interval`: vsyncの間引き（上表参照）
- `--padding-sec`: 同期信号（黒→赤）の前後に入れる黒余白（秒）。PCの安定待ち用（既定: 5秒）
- `--qr-sec`: GT用QRの表示秒数（既定: 3）
- `--repeat`: normal/inv を切り替えるフレーム反復（1なら毎フレーム交互）

表示は以下の順で進みます:

1. **黒 5秒**（余白・ウォームアップ）
2. **全面黒 → 全面赤**（同期用スレート、各0.5秒）
3. **黒 5秒**（余白・安定待ち）
4. **GT用QR表示 3秒**（`HP_QR.png`・条件0の直前）
5. **条件 0〜29**（各6秒、normal/inv 交互）
6. **GT用QR表示 3秒**（中間）
7. **条件 30〜59**
8. **GT用QR表示 3秒**（末尾）

撮影した動画は命名規則に従って保存してください（例: `r180_e250_f1.mp4`）。

**注意:** QR差し込み入りの表示に更新したので、`present_session.exe` を再ビルドしてください。GT表示は埋め込みQRと同じ配置（1920×1080 中央の 1080×1080 正方形）で出します。`gen_assets.py` が作る `gt_qr_display.png` を優先し、無ければ `HP_QR.png`（330×330）から同じ配置で合成します（全面引き伸ばしはしません）。旧動画（QR区間なし・旧manifest）を解析する場合は、manifest に `gt_qr_slots` が無いと従来の連続タイムラインとして扱います。解析時は QR スロット中央付近から `frame_QR.png` を自動生成し、`pixel_acc_*` に使います。

## 解析（1コマンド）

`黒→赤` の切替を同期にして、各条件の **2秒目〜4秒目**（長さ2秒）を先頭から連続120フレーム切り出し→既存解析を実行します。
（以前の「2〜5秒から120枚間引き」だと時間幅が3秒のままだったので、連続切り出しに変更しています。）
条件ごとに動画を開き直して連続 read します（MP4 のフレーム単位シーク失敗で、2条件目以降だけ古い切り出しが残るのを防止）。
**条件ごとに差分・デコードまで回し、完了のたびに `results.csv` を上書き**します（途中まで残るのでデバッグしやすい）。
`--manifest` を指定すると `slate_sec` / `padding_sec` を自動読み取りします（未指定時は slate=0.5秒、padding=5秒）。

差分計算は常に **120フレーム分** 行います。`--diff-mode` で差分の作り方を切り替えます（既定 `pair`）。

| `--diff-mode` | 内容 | 既定スイープ |
|---|---|---|
| `pair`（既定） | 隣接フレームの max-channel 差分 | th = **4 / 8 / 12** |
| `accum` | 非重複窓 `1..n`, `n+1..2n`, … で abs(max-channel差分) の合算 | n = **3 / 5** × th = **12 / 16 / 24 / 32** |
| `stat` | 120フレーム時系列の各ピクセル統計（`std`/`var`）を二値化 | std: th = **4 / 8 / 12**、var: th = **1 / 2 / 4** |
| `fourier` | 120フレーム時系列の時間軸FFT（max-channel）で特定周波数成分を抽出 | 第一候補+半分の2周波数 × th = **4 / 8 / 12** |

デコード成功したもののうち `pixel_acc_ok` が最大の組み合わせを `results.csv` に採用します（同点なら小さい th、さらに同点なら小さい n）。
`pair` は重くなりやすいので、**decode判定は全ペアで実施したまま**、保存する差分PNGだけを **成功ペア + 最高accuracyペア + 10枚に1枚** に間引きます。全件詳細は `qr_decode_all_frames.csv` に残ります。
`--max-frames` は解析後に残す `frame_*.png` 枚数だけを切り替えます。

解析後に120枚残す（既定・pair・QR探索は fast）:

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --max-frames 120
```

窓合算（accum）で解析:

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode accum
```

accum のスイープを上書きする例:

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode accum --window-ns 4 --diff-thresholds 16,24,32
```

統計（標準偏差）で解析:

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode stat
```

統計（分散）で解析:

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode stat --stat-kind var
```

時間軸フーリエ解析（fourier）:

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --diff-mode fourier --mid-search
```

`fourier` には `scipy` が必要です: `pip install scipy`

周波数を手動指定する例（r90 → 15 Hz と 7.5 Hz）:

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r90_e250_f1.mp4 --manifest R8/make_movie/manifests/r90_e250.json --diff-mode fourier --target-freqs 15,7.5
```

解析後に1枚だけ残す（ディスク節約。差分自体は120フレーム分）:

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --max-frames 1
```

QR探索を mid（拡大なしの中間探索）にする:

```bash
python R8/analyze_code/run_pipeline.py --video R8/movie/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json --mid-search
```

- `--max-frames`: `120`（既定）/ `2`（先頭+末尾2枚）/ `1`（残すPNG枚数）。差分計算は常に120フレーム分。`--reuse-frames`（既定）時は次モード再利用のため120枚を残す
- `--reuse-frames` / `--force-extract`: 既存の条件フォルダ切り出しを再利用（既定）/ 毎回切り出し直す
- `--diff-mode`: `pair`（既定）/ `accum` / `stat` / `fourier`
- `--window-ns`: accum 時の窓長（カンマ区切り。未指定時 `3,5`）
- `--stat-kind`: stat 時の統計量（`std` 既定 / `var`）
- `--target-freqs`: fourier 時のターゲット周波数 Hz（カンマ区切り。未指定時は rate_hz+fps から第一候補+半分を自動）
- `--fourier-band-radius`: fourier 時の FFT ビン前後幅（既定 `1`）
- `--diff-thresholds`: 閾値スイープ（カンマ区切り。未指定時は mode・stat-kind ごとの既定）
- 切り出し区間の既定: `--use-start-sec 2 --use-end-sec 4`（60fpsならちょうど約120枚）
- QR探索モード（`decode_qr_from_all_frames.py` に渡る）:

| フラグ | モード | 二値化バリアント | メディアン kernel | 拡大 | デコード |
|---|---|---|---|---|---|
| （なし） | **fast**（既定） | `gray` | 5 | なし | 最精度経路を1回（`detectAndDecodeMulti`） |
| `--mid-search` | **mid** | `gray`, `median_otsu` | **3, 5, 7** | なし | cascade + Multi |
| `--full-search` | **full** | 10種（全種） | 5（既定） | 1/2/3 | cascade + Multi |

`--mid-search` はバリアント・カーネルを広げて読取を強化するモードです。両方指定した場合は `--full-search` が優先されます。`--median-kernels` を明示すると mid の既定 3/5/7 より優先されます。
検証・一括解析の既定は **fast**（gray × kernel5 × 最精度デコード1回）です。
既に切り出した条件フォルダだけ再デコードする場合の例:

```bash
python R8/analyze_code/decode_qr_from_all_frames.py --base-dir R8/analyze_code/out/r180_e250_f1 --folder rice_R_4 --mid-search
```

出力先（デフォルト）:

- `R8/analyze_code/out/r180_e250_f1/`
  - `frame_QR.png`（動画内のGT用QR表示から自動生成した正解マスク）
  - `rice_R_4/` ... `ex_X_12/`（表示順どおり: チャネル→画像→強度）
  - `rgb_max_diff_maps_th4/` … `th12/`（pair 時。各条件内・閾値ごとの差分）
  - `rgb_max_accum_n4_th16/` など（accum 時。窓長×閾値ごとの差分）
  - `rgb_max_stat_std_th4/` / `rgb_max_stat_var_th1/` など（stat 時。統計種別×閾値）
  - `rgb_max_fourier_f30_th8/` など（fourier 時。周波数×閾値）
  - `qr_decode_all_frames.csv`（全条件のデコード詳細。条件完了ごとに蓄積して上書き）
  - `results.csv`（条件完了ごとに蓄積して上書きされる集約結果）

`results.csv` の列（左が要約）:

| 列 | 内容 |
|---|---|
| `folder` | 条件フォルダ名（例: `ex_R_8`） |
| `decode_note` | 失敗理由。成功時は空（`decode_success` は出さない） |
| `decode_variant` | 成功時に効いたバリアント名（例: `median_otsu`）。失敗時は空 |
| `diff_mode` | `pair` / `accum` / `stat` / `fourier` |
| `window_n` | accum で採用した窓長。pair 時は空 |
| `stat_kind` | stat で採用した統計種別（`std` / `var`）。pair/accum/fourier 時は空 |
| `fft_target_hz` | fourier で採用したターゲット周波数（Hz）。他モード時は空 |
| `diff_threshold` | 採用した差分閾値。毎回総当たりして選ぶ |
| `pixel_acc_all` | 全差分ペア／窓の画素 accuracy 平均（GT QR から作った `frame_QR.png` があるとき） |
| `pixel_acc_ok` | デコード成功したものだけの accuracy 平均（同上） |
| `video` / `display_rate` / `exposure` / `fluorescent` / `camera_fps` | 動画共通メタ。**先頭行だけ**埋める（例: `180 Hz`, `1/250`, `蛍光灯あり`, `59.940 fps`） |
| `cond` … `intensity` | 条件の内訳 |
| `decode_decoded_text` / `decode_method` / `decode_frame_*` | 成功した代表の詳細（accum 時 `frame_*` は窓の端点） |
| `note` / `analysis_frames` / `extract_sec` | パイプライン側のメモ・切り出し健全性 |

ペア／窓単位の全詳細は `qr_decode_all_frames.csv` を参照してください。
特に `pair` は、PNG保存は間引かれていても **CSV には全ペアの判定結果**が入ります。

## 一括解析（全動画）

`R8/movie` 内の命名規則に合う `*.mp4` を順番に `run_pipeline.py` で解析します。1本失敗しても他は続行します。

```bash
python R8/analyze_code/all_analyze.py
```

既定は `pair` + **fast** + `--max-frames 120` + `--workers 4`。manifest は `R8/make_movie/manifests/r{rate}_e{exp}.json` を自動で探します。追加引数は `run_pipeline.py` にそのまま渡せます。`--workers` は実コア数を超えないよう自動上限されます。

切り出しは **再利用が既定**です。同じ動画を `pair` → `accum` → `stat` → `fourier` と回すとき、条件フォルダに `frame_*.png` が120枚あれば再切り出ししません。毎回切り直す場合は `--force-extract` を付けます。再利用中は `--max-frames 1|2` でも解析用120枚は残します（次モードのため）。

```bash
python R8/analyze_code/all_analyze.py --max-frames 2
python R8/analyze_code/all_analyze.py --diff-mode accum
python R8/analyze_code/all_analyze.py --diff-mode fourier --max-frames 2
python R8/analyze_code/all_analyze.py --mid-search --max-frames 1
```

サマリー: `R8/analyze_code/out/all_analyze_summary.csv`（`video`, `manifest`, `status`, `exit_code`, `note`）

