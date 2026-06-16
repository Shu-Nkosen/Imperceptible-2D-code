# R8

## Time-axis FFT detector

特定周波数で点滅する領域を動画から抽出する実装です。

### 実行例

```bash
python R8/calc-video-fft.py input.mp4 --target-freq 15 --output-dir R8_output
```

```bash
python R8/calc-video-fft.py --demo --target-freq 15 --output-dir R8_output
```

### 出力

- `*_heatmap.png`
- `*_mask.png`
- `*_summary.csv`

### 依存関係

```bash
pip install -r R8/requirements.txt
```