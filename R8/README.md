# R8 sweep (表示→撮影→解析)

## 命名規則（動画ファイル）

**`r{rate}_e{exp}_f{fluoro}.mp4`**

- `rate`: `45|60|90|120|180`
- `exp`: `250|125|60`（例: `e250` = 1/250）
- `fluoro`: `f1`=蛍光灯あり / `f0`=なし

例: `r180_e250_f1.mp4`

## 事前準備（画像生成）

`R8/make_movie` 内で実行します。事前に `ex.png` などのベース画像と `HP_QR.png` が必要です。

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

事前に **Windowsのディスプレイ設定でリフレッシュレートを target Hz に設定**してから実行します（特に120Hz）。

```powershell
.\present_session.exe --rate 180 --exp 250 --interval 1 --block-sec 6 --slate-sec 0.5 --padding-sec 5 --repeat 1 --out-manifest manifests\r180_e250.json
```

- `--interval`: vsyncの間引き（例: 180Hzモニタで 90Hz相当なら `--interval 2`）
- `--padding-sec`: 同期信号（黒→赤）の前後に入れる黒余白（秒）。PCの安定待ち用（既定: 5秒）
- `--repeat`: normal/inv を切り替えるフレーム反復（1なら毎フレーム交互）

表示は以下の順で進みます:

1. **黒 5秒**（余白・ウォームアップ）
2. **全面黒 → 全面赤**（同期用スレート、各0.5秒）
3. **黒 5秒**（余白・安定待ち）
4. **60条件 × 6秒**（各条件は normal/inv の交互表示）

撮影した動画は命名規則に従って保存してください（例: `r180_e250_f1.mp4`）。

## 解析（1コマンド）

`黒→赤` の切替を同期にして、各条件の **2秒目〜5秒目** を切り出し→既存解析を実行→`results.csv` まで出力します。
`--manifest` を指定すると `slate_sec` / `padding_sec` を自動読み取りします（未指定時は slate=0.5秒、padding=5秒）。

```bash
python R8/analyze_code/run_pipeline.py --video recordings/r180_e250_f1.mp4 --manifest R8/make_movie/manifests/r180_e250.json
```

出力先（デフォルト）:

- `R8/analyze_code/out/r180_e250_f1/`
  - `rice_R_4/` ... `ex_X_12/`（表示順どおり: チャネル→画像→強度）
  - `qr_decode_all_frames.csv`（既存スクリプト出力）
  - `results.csv`（集約結果）

