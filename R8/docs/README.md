# R8/docs — 解析の説明用ドキュメント

スライド報告や口頭説明向けに、`analyze_code` の**手法と流れ**をまとめたフォルダです。  
コマンド手順・運用メモは親の [`../README.md`](../README.md) と [`../command.md`](../command.md) を参照してください。

## 読む順番

1. [01_pipeline.md](01_pipeline.md) — 実験から CSV までの全体フロー
2. [02_methods.md](02_methods.md) — mid_hully の `results_*` ごと・binary/num・パラメータ詳細
3. [03_conditions_and_metrics.md](03_conditions_and_metrics.md) — 条件・命名・評価指標・出力
4. [04_status_memo.md](04_status_memo.md) — **現状整理**（re_analyze 含む。結果まとめ前のメモ）
5. [05_method_sheet.md](05_method_sheet.md) — **報告用・6手法シート**（フレーム数・閾値・条件）

## スコープ

- **含む**: パイプライン、各解析手法の意味、実験条件、成功率の定義、現状メモ、報告用手法シート
- **含まない**: 条件ごとの成功率の最終集計表・スライド本体（結果揃い後）

## 関連コード

| 役割 | パス |
|---|---|
| 現行コアパイプライン | [`../analyze_code/run_pipeline_hully.py`](../analyze_code/run_pipeline_hully.py) |
| 差分マップ生成 | [`../analyze_code/hully_diff.py`](../analyze_code/hully_diff.py) |
| FFT / lockin | [`../analyze_code/time_fft.py`](../analyze_code/time_fft.py) |
| GPU/CPU 演算 | [`../analyze_code/gpu_ops.py`](../analyze_code/gpu_ops.py) |
| QRデコード | [`../analyze_code/decode_qr_from_all_frames.py`](../analyze_code/decode_qr_from_all_frames.py) |
| 一括（主） | [`../analyze_code/all_analyze_mid.py`](../analyze_code/all_analyze_mid.py) |
| 一括（日常・軽い） | [`../analyze_code/all_analyze_mid_fast.py`](../analyze_code/all_analyze_mid_fast.py) |
| 不足スイープ追記 | [`../analyze_code/re_analyze_mid_fast.py`](../analyze_code/re_analyze_mid_fast.py) |
| 画像×チャネル調査メモ | [`../analyze_code/out_image_channel_study/FINDINGS.md`](../analyze_code/out_image_channel_study/FINDINGS.md) |
