# mid_fast_0805 スライド用図

Source: `R8/analyze_code/out_mid_fast_0805`（30 stems × 60 conditions = 1800）。

## 画像ラベル（日本語）
- ex → **実験**（青）
- hocho → **工事**（赤）
- nagaoka_fireworks → **花火**（黒）
- rice → **自然**（緑）

## 指標
- **any-success**: 条件のうち、スイープのどれかで `decode_success=1` の割合
- **family**: binary と `*_num` の OR
- **adopted**: `pixel_acc_all` 最大スイープでの成功

## 注意
`re_analyze_mid_fast` により lockin/fourier の不足スイープを追記済み。
Fourier binary に旧 th/255 行が残る場合あり（系統 fourier は binary+num の OR）。

## ファイル
- `00_overview.png` … 一枚もの
- `01_family_any_success.png`
- `02_binary_vs_num.png`
- `03_by_rate.png`
- `04_by_exposure.png`
- `05_by_fluoro.png`
- `06_image_channel_heatmap.png`
- `07_image_color_strength.png` … 色の強さ（平均・支配チャネル・テクスチャ）
- `08_image_rgb_histograms.png` … 元画像 RGB ヒストグラム
- `metrics.json` / `*.csv`

## intensity クロス（追加）
- `09_intensity_by_family.png` … 強度×手法
- `10_intensity_by_rate.png` … 強度×表示レート
- `11_intensity_by_exposure.png` … 強度×露光
- `12_intensity_by_channel.png` … 強度×チャネル
- `13_intensity_by_family.png` … 画像別 2×2（強度×手法）
- `14_intensity_heatmap.png` … 4手法×3強度の画像×チャネル一枚まとめ
- `15_freq_by_rate_lockin_fourier.png` … 表示45/60/90の周波数別 lockin/fourier
- `intensity_cross.csv` / `freq_by_rate_lockin_fourier.csv` … 数値表
