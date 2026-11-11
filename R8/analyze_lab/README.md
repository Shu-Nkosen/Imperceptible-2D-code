# Lab 差分解析

相補フレーム（normal/inv）の差分を float32 の CIE Lab 空間で計算し、
既存の RGB 最大チャネル解析と比較する実験用スクリプトです。

R8 から実行します。

```powershell
python analyze_lab/run_lab_diff.py --video movie_rgblab/r60_e250_f1.mp4 --manifest make_movie/manifests/r60_e250_rgblab.json --mid-search
```

`out/r60_e250_f1/` に `results.csv`、`pair_accuracy.csv`、`lab_stats.csv`、
`compare.csv` を出力します。`results.csv` と `pair_accuracy.csv` は既存解析と
同じ列構成で、`lab_stats.csv` は QR の黒/白モジュール別の Lab 差分平均、
`compare.csv` は既存 `analyze_code/out_rgblab` との比較です。
