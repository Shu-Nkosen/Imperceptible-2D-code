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
`pair` は **先頭20＋末尾20ペア（最大40）** だけ差分生成・デコードします（`--pair-each-end 20`）。保存する差分PNGはさらに **成功ペア + 最高accuracyペア + 10枚に1枚** に間引きます。
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
- `--quiet`: INFO ログを抑制し、対象・モード・結果の要約だけ出す（一括実行では自動付与）
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
| `pixel_acc_best` | 全ペア／窓のうち最大 accuracy |
| `best_pair_frame_1` / `best_pair_frame_2` | 最高 accuracy のペア／窓端点 |
| `video` / `display_rate` / `exposure` / `fluorescent` / `camera_fps` | 動画共通メタ。**先頭行だけ**埋める（例: `180 Hz`, `1/250`, `蛍光灯あり`, `59.940 fps`） |
| `cond` … `intensity` | 条件の内訳 |
| `decode_decoded_text` / `decode_method` / `decode_frame_*` | 成功した代表の詳細（accum 時 `frame_*` は窓の端点） |
| `note` / `analysis_frames` / `extract_sec` | パイプライン側のメモ・切り出し健全性 |

ペア／窓単位の全詳細は `qr_decode_all_frames.csv` を参照してください。
**pair モード**では先頭20＋末尾20ペアだけ解析し、同じ範囲を `pair_accuracy.csv` にも出します。

---

## CSV出力の読み方

### 文字化けについて

CSV は **UTF-8（BOM付き / `utf-8-sig`）** で書き出しています。

| 開き方 | 日本語 |
|--------|--------|
| **Excel（Windows）** | ダブルクリックでだいたいOK（BOM付きのため） |
| **Excel** で文字化けする場合 | 「データ」→「テキスト/CSV」→ ファイルの元の形式 **65001: Unicode (UTF-8)** |
| **Python / pandas** | `pd.read_csv(path, encoding="utf-8-sig")` |
| **メモ帳 / VS Code** | そのまま読める |

日本語が入る主な列: `decode_note`, `fluorescent`, `note`, `gt_note`（例: `蛍光灯あり`, `QR未検出`）。

### ファイル一覧

| ファイル | 場所 | 1行の意味 |
|----------|------|-----------|
| `results.csv` | `out/<動画stem>/` | **条件1個**の要約（60行） |
| `results_<pass>.csv` | 同上 | `all_analyze` 実行時の手法別アーカイブ（例: `results_pair.csv`） |
| `pair_accuracy.csv` | 同上 | **pair モード**で採用閾値のペア正解率（**先頭20＋末尾20** / 条件） |
| `qr_decode_all_frames.csv` | 同上 | **差分1枚（ペア/窓）** ごとのデコード詳細（全閾値スイープ含む） |
| `qr_decode_all_frames_<pass>.csv` | 同上 | 手法別アーカイブ |
| `all_analyze_summary.csv` | `out/` | **動画×手法** ごとの成否サマリー |
| `all_analyze_timing.csv` | `out/` | **動画×手法** ごとの所要時間（追記・蓄積） |

1回の `run_pipeline` は **1つの `diff_mode` のみ**。モード比較は `results_pair.csv` と `results_fourier.csv` などを並べて見ます。

---

### `results.csv`（条件要約・60行/動画）

左から重要な列順です。

| 列 | 説明 |
|----|------|
| `folder` | 条件フォルダ名（例: `rice_R_4`, `ex_X_12`） |
| `decode_note` | **空 = QRデコード成功**。失敗時は理由（例: `QR未検出`, `extract失敗: 0 frames`） |
| `decode_variant` | 成功時に効いた二値化バリアント（fast では `gray` など）。失敗時は空 |
| `diff_mode` | 今回の解析手法: `pair` / `accum` / `stat` / `fourier` |
| `window_n` | accum で**採用した**窓長（例: `3`, `5`）。他モードは空 |
| `stat_kind` | stat で**採用した**統計量: `std` / `var`。他モードは空 |
| `fft_target_hz` | fourier で**採用した**ターゲット周波数（Hz）。他モードは空 |
| `diff_threshold` | **採用した**差分二値化閾値（0–255 スケール、例: `8`） |
| `pixel_acc_all` | その条件の全ペア/窓について、GT QR（`frame_QR.png`）との画素一致率の**平均**（0〜1）。GT 無しは空 |
| `pixel_acc_ok` | 上記のうち **デコード成功したものだけ** の平均。採用スコアの目安 |
| `pixel_acc_best` | 全ペア/窓のうち **最大** の画素一致率（0〜1）。GT 無しは空 |
| `best_pair_frame_1` | `pixel_acc_best` を出したペア/窓の左端フレーム |
| `best_pair_frame_2` | 同上・右端フレーム |
| `video` | 動画ファイル名。**先頭行（cond 0）だけ**埋まる |
| `display_rate` | 表示レート（例: `180 Hz`）。先頭行のみ |
| `exposure` | 露光（例: `1/250`）。先頭行のみ |
| `fluorescent` | `蛍光灯あり` / `蛍光灯なし`。先頭行のみ |
| `camera_fps` | 動画から検出した fps（例: `59.940 fps`）。先頭行のみ |
| `cond` | 条件番号 0〜59 |
| `image` | 画像名（`rice`, `ex` など） |
| `channel` | チャネル（`R`, `G`, `B`, `X`=max, `I`=min） |
| `token` | 内部トークン（CSV上の `channel` と対応） |
| `intensity` | 強度 4 / 8 / 12 |
| `decode_decoded_text` | 読み取れた QR 文字列。失敗時は空 |
| `decode_method` | 成功時の OpenCV 経路タグ（例: `gray_k5_i1`） |
| `decode_frame_1` | デコード成功した**最初の**ペア/窓の左端（代表行） |
| `decode_frame_2` | 同上・右端（`best_pair_*` とは別。最高正解率ペアは `best_pair_frame_*`） |
| `note` | パイプライン内部メモ（通常は空） |
| `analysis_frames` | 切り出して解析に使ったフレーム数（通常 120） |
| `extract_sec` | 切り出し区間の秒数（再利用時は 0） |

**採用ルール:** 各条件で閾値（＋ accum なら窓長、fourier なら周波数）を総当たりし、**デコード成功のうち `pixel_acc_ok` 最大**の組み合わせを1行に載せます。

---

### `qr_decode_all_frames.csv`（全ペア/窓の詳細）

| 列 | 説明 |
|----|------|
| `folder` | 条件フォルダ名 |
| `frame_1` | ペア左 / 窓始端のフレーム名 |
| `frame_2` | ペア右 / 窓終端のフレーム名 |
| `diff_image` | 使った差分 PNG の相対パス |
| `analysis_image` | デコード成功時の中間画像（保存設定による） |
| `decoded_text` | 読み取り文字列。失敗時は空 |
| `success` | `1` = 成功, `0` = 失敗 |
| `method` | 二値化＋デコード手法タグ |
| `note` | 失敗理由（成功時は空） |
| `recall` | GT QR との再現率（0〜1）。GT 無しは空 |
| `precision` | GT QR との精度（0〜1） |
| `accuracy` | GT QR との画素一致率（0〜1） |
| `noise` | ノイズ指標（GT 比較時） |
| `gt_note` | GT 画像が無い等のメモ |
| `diff_mode` | `run_pipeline` 実行時に付く（`pair` 等） |
| `window_n` | 試した accum 窓長 |
| `stat_kind` | 試した stat 種別 |
| `fft_target_hz` | 試した fourier 周波数 |
| `diff_threshold` | 試した閾値 |

`pair` では1条件あたり最大 **40行×閾値数**（先頭20＋末尾20ペア）。`stat` / `fourier` は閾値スイープ分のみ。

**正解率:** GT がある場合、QR デコード失敗でも差分画像（最後に試した二値化 or 生 gray）と GT を比較して `accuracy` を出します。

---

### `pair_accuracy.csv`（pair・採用閾値のペア別正解率）

`diff_mode=pair` のときだけ出力。**先頭20ペア＋末尾20ペア**（最大40行/条件。119未満なら全件）だけ載せます。

| 列 | 説明 |
|----|------|
| `folder` | 条件フォルダ名 |
| `cond` | 条件番号 0〜59 |
| `frame_1` | ペア左フレーム |
| `frame_2` | ペア右フレーム |
| `success` | QR デコード成功 `1` / 失敗 `0` |
| `accuracy` | GT との画素一致率（0〜1） |
| `recall` | GT との再現率 |
| `precision` | GT との精度 |
| `diff_threshold` | 採用した閾値 |

1条件あたり最大 **40行**（先頭20＋末尾20）。`pixel_acc_*` もこの40ペアから計算します。

---

### `all_analyze_summary.csv`（一括実行サマリー）

| 列 | 説明 |
|----|------|
| `video` | 動画ファイル名（例: `r180_e250_f1.mp4`） |
| `pass` | 解析手法: `pair`, `accum`, `stat_std`, `stat_var`, `fourier` |
| `manifest` | 使った manifest のパス |
| `status` | `OK` / `FAIL` |
| `exit_code` | `run_pipeline` の終了コード（0 = 成功） |
| `note` | 失敗理由（manifest 不在、`run_pipeline exit=1` 等） |

### `all_analyze_timing.csv`（所要時間・追記蓄積）

`out/all_analyze_timing.csv`。実行のたびに**追記**し、過去行は残します。

| 列 | 説明 |
|----|------|
| `finished_at` | ジョブ完了時刻（`YYYY-MM-DD HH:MM:SS`） |
| `video` | 動画ファイル名 |
| `pass` | 解析手法 |
| `status` | `OK` / `FAIL` |
| `elapsed_sec` | 所要秒（小数1桁） |
| `note` | 失敗理由（成功時は空） |

---

## 一括解析（全動画）

`R8/movie` 内の命名規則に合う `*.mp4` を順番に解析します。1本失敗しても他は続行します。

```bash
python R8/analyze_code/all_analyze.py
```

**既定:** 各動画について **全解析手法** を次の順で実行します。

1. `pair`
2. `accum`
3. `stat`（std）
4. `stat`（var）
5. `fourier`

共通設定は **fast** + `--max-frames 120` + `--workers 4` + **切り出し再利用**。  
手法ごとの結果は `results_<pass>.csv` / `qr_decode_all_frames_<pass>.csv` として残ります（例: `results_pair.csv`）。  
`--diff-mode accum` など手法を明示した場合は、その手法だけ実行します。

**ログ:** 一括実行はジョブ（動画×手法）ごとに次の3行だけ出します（`run_pipeline` の詳細ログはキャプチャして端末に流しません）。失敗時のみ末尾ログを短く表示します。単体の `run_pipeline.py` は従来どおり詳細表示（`--quiet` で要約のみにもできます）。

```text
[12/150] target: r180_e250_f1.mp4
[12/150] pass:   pair
[12/150] result: OK
```

```bash
python R8/analyze_code/all_analyze.py --max-frames 2
python R8/analyze_code/all_analyze.py --diff-mode accum
python R8/analyze_code/all_analyze.py --diff-mode fourier --max-frames 2
python R8/analyze_code/all_analyze.py --mid-search --max-frames 1
python R8/analyze_code/all_analyze.py --force-extract
```

サマリー: `R8/analyze_code/out/all_analyze_summary.csv`（`video`, `pass`, `manifest`, `status`, `exit_code`, `note`）  
所要時間: `R8/analyze_code/out/all_analyze_timing.csv`（追記蓄積。`finished_at`, `video`, `pass`, `status`, `elapsed_sec`, `note`）

