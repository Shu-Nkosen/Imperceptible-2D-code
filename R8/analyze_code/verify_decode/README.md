# verify_decode — 独立検証（本体非改変）

現行の `run_pipeline_hully` / `gpu_ops` / `out_mid_fast_0805` の結果は **触らない**。  
検証コードは本ディレクトリ、結果は `../out_verify_decode/` のみ。

## 実験

| 実験 | 内容 | 出力 |
|---|---|---|
| A `exp_matched_channel` | 差分 `max` vs 埋め込みチャネル固定 `matched`（R/G/B） | `out_verify_decode/matched_channel/` |
| B `exp_fourier_score` | 時間FFT `ratio`（target/noise）vs `amp`（割りなし） | `out_verify_decode/fourier_score/` |

対象の既定: 表示 45/60/90 Hz、強度 12、token R/G/B。手法は mid_fast 相当の閾値・周波数候補。

## 前提（重要）

`out_mid_fast_0805` は解析後に `frame_*.png` を消していることが多い（`keep-frames=0`）。  
その場合は次のどちらかが必要:

1. **フレーム付きディレクトリ**を `--in-root` に渡す（stem/folder/frame_*.png）
2. 下の **合成フィクスチャ**でスモークだけ回す（本物データではない）

本物の比較には、動画から再切り出ししてフレームを残した入力が必要。

## 実行手順

作業ディレクトリは常に `R8/analyze_code`。

### 0) まず数値自己検査（数秒）

```bat
python -m verify_decode.selfcheck
```

### 0b)（任意）合成入力で端到端スモーク

合成は本物 QR ではないので decode 成功率は 0% でも、パイプラインが動く確認用。

```bat
python -m verify_decode.make_fixture --out-root out_verify_decode/_fixture
python -m verify_decode.exp_matched_channel.run --in-root out_verify_decode/_fixture --out-root out_verify_decode/matched_channel --limit-stems 1 --methods lockin,fourier
python -m verify_decode.exp_fourier_score.run --in-root out_verify_decode/_fixture --out-root out_verify_decode/fourier_score --limit-stems 1 --with-lockin
python -m verify_decode.report --root out_verify_decode
```

### 1) 実験A: チャネル一致

```bat
python -m verify_decode.exp_matched_channel.run --in-root out_mid_fast_0805 --out-root out_verify_decode/matched_channel
```

縮小例:

```bat
python -m verify_decode.exp_matched_channel.run --in-root PATH\to\frames --limit-stems 2 --limit-folders 4 --methods lockin,fourier
```

### 2) 実験B: FFT 割りなし

```bat
python -m verify_decode.exp_fourier_score.run --in-root out_mid_fast_0805 --out-root out_verify_decode/fourier_score --with-lockin
```

`hocho × B` は `by_image_channel.csv` で確認。

### 3) 要約

```bat
python -m verify_decode.report --root out_verify_decode
```

## 結果の見方

- `matched_channel/by_channel.csv` … `diff_mode=max|matched` × method × token の成功率
- `fourier_score/by_score_mode.csv` … `ratio` vs `amp` 全体
- `fourier_score/by_image_channel.csv` … 画像×チャネル（工事 B が主眼）
- いずれも発表正本（`AI_HANDOFF` / `summary_slides`）とは別数字

## 本体との境界

| やる | やらない |
|---|---|
| 本ディレクトリにコード追加 | `gpu_ops` / `hully_diff` / `run_pipeline*` の改修 |
| `out_verify_decode/` へ書く | `out_mid_fast_0805` の CSV・図・HANDOFF 更新 |
| デコードは既存 `decode_qr_from_all_frames` を import | 本番パイプラインへの自動合流 |

## 判定の目安

- A: matched が B で明確に上、R/G が同等以上 → 本番取り込みを別タスクで検討
- B: `amp` が hocho×B で上がり他を大きく壊さない → fourier スコア変更を別タスクで検討
