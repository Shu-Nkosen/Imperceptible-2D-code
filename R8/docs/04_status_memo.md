# 現状整理メモ（結果まとめ前）

報告用に結果をまとめるまでのメモ。集計表そのものはまだ書かない。  
データが揃ったら（`out_mid_fast_*` ＋ `re_analyze_mid_fast` 追記後）ここを起点にスライド化する。

関連: [手法シート](05_method_sheet.md) / [パイプライン](01_pipeline.md) / [評価指標](03_conditions_and_metrics.md)

---

## いま何をやっているか（一文）

表示点滅 QR をカメラ動画から復元する実験で、共通の差分系列 \(d(t)\) から **6系統のスコア地図**を作り、地図ごとに QR 読取を試す。候補（閾値・周波数など）は多いが、**報告は系統ごとに「いちばん良い読取結果」だけ**を出す方針。

---

## データの置き場

| もの | 場所 | 備考 |
|---|---|---|
| 日常解析の出力 | `analyze_code/out_mid_fast_MMDD/`（例: `out_mid_fast_0805`） | 動画ごと `r{rate}_e{exp}_f{fluoro}/` |
| 初回一括 | `all_analyze_mid_fast.py` | 軽い mid_fast スイープ |
| 不足スイープ追記 | `re_analyze_mid_fast.py` | 既存 CSV を消さず追記。既定は fourier/lockin の新 th |
| 追記サマリ | `out_mid_fast_*/re_analyze_mid_fast_summary.csv` | dry-run / 本実行の不足件数 |

`keep-frames=0` のため生フレームは残っていない。追記時は動画＋manifest が必要。

---

## 解析プロファイル（いまの主戦場）

**mid_fast**（`--mid-fast-sweeps`）を主に見る。

共通前処理:

1. 黒→赤スレートで同期  
2. 各条件の 2〜4 秒から **120フレーム**  
3. 隣接 max-channel 差分系列 \(d(t)\)（長さ約 119）を1回生成  
4. 各手法でスコアマップ → QRデコード（mid-search: `gray` → `median_otsu`）

実験軸（動画）: rate × exp × fluoro  
条件軸（1動画60）: 画像4 × チャネル5 × 強度3  

---

## pass の整理（報告の6系統との対応）

実装上は binary / gray（`*_num`）など複数 CSV があるが、**報告の見出しは次の6つ**にまとめる。

| # | 報告名 | 主な pass（CSV） | 備考 |
|---|---|---|---|
| 1 | 隣接フレーム差分 | `pair` | num なし |
| 2 | 複数フレーム合計差分 | `accum` / `accum_num` | 良い方（または binary 優先＋注記） |
| 3 | 時間方向 std | `stat_std` / `stat_std_num` | 同上 |
| 4 | 時間方向 var | `stat_var` | num なし |
| 5 | ロックイン | `lockin` / `lockin_num` | 新 th は binary 側を re_analyze で追記 |
| 6 | 時間 FFT | `fourier` / `fourier_num` | 同上。旧 binary th/255 はほぼ無効だった |

詳細なパラメータは [05_method_sheet.md](05_method_sheet.md)。

---

## re_analyze_mid_fast で何が増えるか

動機: mid_fast の **lockin / fourier binary** が、正規化スコアに対して壊れた `th/255`（4/8/12）を使っていた。

修正後の候補（追記対象）:

- 固定: 正規化 [0,1] 上の **0.30 / 0.50 / 0.70**（コード 30/50/70）  
- 適応: **Otsu**（901）、**adaptive**（902）

既存の旧 th=4/8/12 行は消さない（比較用）。  
`*_num`・pair/accum/stat は既定では触らない。

```bash
python R8/analyze_code/re_analyze_mid_fast.py --out-dir R8/analyze_code/out_mid_fast_0805 --dry-run
python R8/analyze_code/re_analyze_mid_fast.py --out-dir R8/analyze_code/out_mid_fast_0805
```

---

## 結果が出たあとにまとめる手順（チェックリスト）

1. **追記完了確認**  
   `re_analyze_mid_fast_summary.csv` で `missing_sweeps=0`（または意図どおり減少）を確認。

2. **6系統それぞれ**  
   条件単位で「スイープのどれかで読めたか」（any-success）と、採用行（adopted）の成功率を集計。  
   binary と num がある系統は両方出し、報告では **系統の代表1つ**（例: 成功率が高い方、または方針で固定）に絞る。

3. **切り口の例**（スライド数を抑える）  
   - 手法（6系統）の横比較  
   - rate / exp / fluoro の効果（代表手法だけでも可）  
   - 画像 × チャネル（必要なら FINDINGS と接続）

4. **fourier 特記**  
   - 旧 binary（th/255）はほぼ読めなかったこと  
   - num および新 binary（p/Otsu/adapt）でどう変わったか  

5. **数字の定義を毎回明示**  
   - 主指標: QRデコード成功  
   - 副指標: GT 画素 accuracy（`pixel_acc_*`）  
   - 採用: 条件内で `pixel_acc_all` 最大（デコード成功は同点ルール）

---

## いま分かっていること（追記前データからのメモ）

`out_mid_fast_0805`（追記前）の概略:

- accum / lockin / stat_std: binary が num と同程度〜やや優位  
- **fourier**: binary（旧 th）≈0%、num≈10% → num が圧倒（壊れた二値化が主因）  
- 高テクスチャ画像では pair が厳しく、周波数系が相対的に効く場合がある（FINDINGS）

追記後は **fourier/lockin の新 binary** を上表に必ず足す。

---

## 関連コマンド（最短）

```bash
# 日常一括（新規 out_mid_fast_MMDD）
python R8/analyze_code/all_analyze_mid_fast.py

# 既存への穴埋め
python R8/analyze_code/re_analyze_mid_fast.py --out-dir R8/analyze_code/out_mid_fast_0805
```
