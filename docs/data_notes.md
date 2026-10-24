# WM-811K 資料筆記（Data Notes）


> 對應程式：[`notebooks/01_data_inspection.ipynb`](../notebooks/01_data_inspection.ipynb)
> 最後更新：2026-10-24

---

## TL;DR

- 811,457 片晶圓，但**只有 172,950 片有 label**，其餘 638,507 片是 unlabeled。
- 有 label 的資料中：正常（`none`）**85.24%**、defect（8 類）**14.76%**，比例 **1 : 5.8**。以 anomaly detection 來說**不夠稀有**，要討論怎麼處理（見 §8）。
- `none` **不是「沒有壞 die」**，而是「壞 die 隨機散布、沒有形成空間圖樣」。anomaly 的本質是**空間 pattern**，不是壞 die 的數量。
- 每張 wafer map 的尺寸都不同（有 label 的資料共 346 種），**模型輸入前一定要統一尺寸**。
- 原作者的 train/test 切分是**以 lot 為單位**，但兩邊的 defect 比例差很多（32.4% vs 6.7%）。
- 讀原始 pickle 需要一行相容性修正（見 §2）。之後請直接讀 `data/LSWMD_clean.pkl`。

---

## 1. 資料來源

| 項目 | 內容 |
|---|---|
| 名稱 | WM-811K wafer map dataset |
| 下載 | <https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map> |
| 原始論文 | Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets*. IEEE Transactions on Semiconductor Manufacturing, 28(1), 1–12. |
| 資料性質 | 真實晶圓廠的晶圓測試結果，label 由專家標註 |
| 作業分類 | **Category 3：Image**（每個樣本是一張 2D 圖） |

> paper 引用前請到 IEEE Xplore 再確認一次書目資訊（作業規定引用必須真實且可驗證）。

---

## 2. 如何讀取

### 檔案位置

```
hw2/
├── data/
│   ├── LSWMD.pkl          # 原始檔（~2.0 GB，從 Kaggle zip 解壓）
│   └── LSWMD_clean.pkl    # 整理後版本（~2.0 GB，由 notebook 產生）← 平常請用這個
├── notebooks/
│   └── 01_data_inspection.ipynb
├── src/
│   └── make_split.py      # 產生 split 檔
├── splits/
│   └── split_v1.csv       # train/val/test 切分（見 docs/split.md）
└── docs/
    ├── data_notes.md      # 本文件
    └── split.md
```

> 兩個 pkl 都超過 GitHub 單檔 100MB 的上限，**不會放進 repo**。組員需要自己從 Kaggle 下載，再跑一次 notebook 產生 clean 版。

### 讀整理後的版本（推薦）

```python
import pandas as pd
df = pd.read_pickle("data/LSWMD_clean.pkl")
```

### 讀原始檔（只有第一次需要）

原始 pickle 是用很舊的 pandas（0.x）存的，裡面引用到已經不存在的模組路徑 `pandas.indexes`。用 pandas 2.x 直接讀會報錯：`ModuleNotFoundError: No module named 'pandas.indexes'`。解法是先建立別名：

```python
import sys
import pandas as pd
import pandas.core.indexes
sys.modules["pandas.indexes"] = pandas.core.indexes

df = pd.read_pickle("data/LSWMD.pkl")
```

測試環境：Python 3.12.4、pandas 2.2.3、numpy 2.1.2。讀檔需要數十秒到幾分鐘，建議記憶體至少 8GB。

### `LSWMD_clean.pkl` 和原始檔的差別

| 項目 | 原始檔 | clean 版 |
|---|---|---|
| split 欄位名稱 | `trianTestLabel`（原作者拼錯） | `trainTestLabel` |
| `failureType` 格式 | `array([['none']])` | `'none'`（字串） |
| `trainTestLabel` 格式 | `array([['Training']])` | `'Training'`（字串） |
| 沒有 label 的樣本 | 空 array `[]` | `'unlabeled'` |
| 額外欄位 | 無 | `mapH`、`mapW`（wafer map 高、寬） |
| 資料筆數 | 811,457 | 811,457（**沒有刪除任何資料**） |

---

## 3. 欄位說明

每一列（row）代表**一片晶圓**。

| 欄位 | 型態 | 意義 | 用途 |
|---|---|---|---|
| `waferMap` | `np.ndarray`（2D, `uint8`） | 晶圓測試結果圖 | **模型輸入 x** |
| `failureType` | str | 專家標註的 defect 類型 | **ground truth y** |
| `lotName` | str | 批號，例如 `lot1` | **split 分組依據**（見 §7） |
| `waferIndex` | float（1–25） | 在該 lot 中是第幾片 | 輔助資訊 |
| `dieSize` | float | 該晶圓上 die 的總數 | 輔助資訊；可以當 Tier 1 特徵 |
| `trainTestLabel` | str | 原作者的 train/test 標記 | 參考用（見 §6） |
| `mapH`, `mapW` | int | wafer map 的高、寬 | clean 版才有 |

### `waferMap` 像素值

| 值 | 意義 |
|---|---|
| `0` | 背景（在圓形晶圓外面，沒有 die） |
| `1` | 良品 die（pass） |
| `2` | 不良 die（fail） |

- 已用前 5,000 張隨機樣本確認，只會出現 0、1、2 這三個值。
- 這三個值是**類別**，不是連續數值。preprocessing 時請注意，例如可以 one-hot 成 3 個 channel，或只取 `== 2` 的二值圖。
- 已驗證：有 label 的資料中 **100%** 符合 `dieSize == (waferMap > 0).sum()`，也就是 `dieSize` 剛好等於「1 的數量 + 2 的數量」。

### 坑：`info()` 顯示沒有缺值，其實有

`df.info()` 顯示每個欄位都是 811,457 non-null，**但這是假象**。沒有 label 的樣本是存成空 array `array([], shape=(0, 0))`，不是 NaN，pandas 不會把它當成缺值。clean 版已經改成 `'unlabeled'` 字串了。

---

## 4. Label 分布

### 整體

| 項目 | 數量 | 比例 |
|---|---:|---:|
| 總筆數 | 811,457 | 100% |
| Unlabeled | 638,507 | 78.7% |
| **有 label** | **172,950** | 21.3% |

### 有 label 的 172,950 筆

| `failureType` | 數量 | 佔有 label 資料 |
|---|---:|---:|
| `none`（正常） | 147,431 | 85.24% |
| Edge-Ring | 9,680 | 5.60% |
| Edge-Loc | 5,189 | 3.00% |
| Center | 4,294 | 2.48% |
| Loc | 3,593 | 2.08% |
| Scratch | 1,193 | 0.69% |
| Random | 866 | 0.50% |
| Donut | 555 | 0.32% |
| Near-full | 149 | 0.09% |
| **Defect 合計** | **25,519** | **14.76%** |

- **Imbalance ratio：defect : normal = 1 : 5.8**
- 8 類 defect 之間也很不平衡：Edge-Ring 9,680 筆，Near-full 只有 149 筆，差了 65 倍。

---

## 5. `none` 的真正意義：空間 pattern，不是數量

**`none` 晶圓上幾乎都有壞 die**（99.999% 的 `none` 晶圓至少有一個像素值 = 2）。

各類別的 **fail rate**（= 不良 die 數 / die 總數）：

| `failureType` | 平均 | 中位數 | 最大 |
|---|---:|---:|---:|
| none | 0.106 | 0.098 | 0.471 |
| Scratch | 0.101 | 0.092 | 0.393 |
| Edge-Ring | 0.151 | 0.140 | 0.451 |
| Loc | 0.150 | 0.142 | 0.698 |
| Edge-Loc | 0.183 | 0.156 | 0.746 |
| Center | 0.232 | 0.250 | 0.727 |
| Donut | 0.277 | 0.258 | 0.678 |
| Random | 0.481 | 0.477 | 0.771 |
| Near-full | 0.877 | 0.878 | 1.000 |

**解讀：**
- **正常（`none`）**：有大約 10% 的壞 die，但**隨機散布**，屬於正常的良率損失。
- **異常（defect）**：壞 die **聚集成特定形狀**，例如外圈、中心、刮痕，代表某個製程步驟出了問題，需要工程師處理。
- **Scratch 的 fail rate 跟 `none` 幾乎一樣**，Edge-Ring 和 Loc 也只高一點點。只看壞 die 數量**沒辦法**分辨大部分 defect，一定要看空間分布。

> ⚠️ 上面的統計是用**全部** label 資料算的，屬於資料集描述。paper §4 的正式 EDA 規定**只能用 training portion**，split 決定之後要在 train set 上重算。

---

## 6. 原作者的 train/test 切分

`failureType` × `trainTestLabel` 交叉表：

| `failureType` | Training | Test | 合計 |
|---|---:|---:|---:|
| none | 36,730 | 110,701 | 147,431 |
| Edge-Ring | 8,554 | 1,126 | 9,680 |
| Edge-Loc | 2,417 | 2,772 | 5,189 |
| Center | 3,462 | 832 | 4,294 |
| Loc | 1,620 | 1,973 | 3,593 |
| Scratch | 500 | 693 | 1,193 |
| Random | 609 | 257 | 866 |
| Donut | 409 | 146 | 555 |
| Near-full | 54 | 95 | 149 |
| **合計** | **54,355** | **118,595** | **172,950** |
| **defect 比例** | **32.4%** | **6.7%** | 14.8% |

**觀察：**
1. **原始切分是以 lot 為單位**：10,762 個有 label 的 lot 中，**沒有任何一個** lot 同時出現在 Training 和 Test。這是好的，沒有 lot-level leakage。
2. **但兩邊分布差很多**：
   - Test 的資料量是 Training 的 2.2 倍，比例反常
   - defect 比例：Training 32.4%，Test 只有 6.7%
   - Edge-Ring 88% 在 Training；Near-full、Scratch、Loc 反而是 Test 比較多
3. 原始切分是為了**監督式分類**設計的，不一定適合我們的 anomaly detection 設定。

---

## 7. Lot 結構（對 split 很重要）

| 項目 | 數值 |
|---|---:|
| lot 總數 | 46,293 |
| 有 label 的 lot | 10,762 |
| 至少含一片 defect 的 lot | 8,047 |
| 全部晶圓都有 label 的 lot | 4,685 |
| 部分有 label 的 lot | 6,077 |
| 完全沒有 label 的 lot | 35,531 |
| 每個 lot 的晶圓數 | 1–25（中位數 24） |

**為什麼要以 lot 為單位切分：** 同一個 lot 的晶圓是同一批、同一組機台做出來的，defect pattern 往往很相似。如果同一個 lot 的晶圓同時出現在 train 和 test，模型等於「看過考題」，分數會虛高，也就是 **data leakage**。建議用 `GroupShuffleSplit` 或 `StratifiedGroupKFold`，以 `lotName` 當 group。這部分寫進 paper §6 的 leakage prevention。

> 注意：同一個 lot 裡面可能有些晶圓有 label、有些沒有（例如 lot2 第 11 片是 unlabeled，但第 10、12 片有 label）。

---

## 8. Wafer map 尺寸

| 範圍 | 不同尺寸數 | 高（H） | 寬（W） |
|---|---:|---|---|
| 全部 811K | 632 | 6 – 300 | 3 – 205 |
| 有 label 的 173K | 346 | 15 – 212 | 3 – 204 |

- 有 label 的資料：高、寬的中位數都是 33，平均約 35，1st–99th percentile 約 25–76。
- 全部資料中最常見的尺寸：32×29（108,687）、25×27（64,083）、49×39（39,323）、26×26（30,078）、30×34（29,513）。
- **影響：** CNN 和 Autoencoder 需要固定大小的輸入，preprocessing 要決定 resize 或 pad。因為像素值是類別，resize 時要用 **nearest-neighbor** 插值，不能用 bilinear，否則會產生 0.5、1.5 這種不存在的值。

---

## 9. 待決定的事項

| # | 問題 | 選項 / 備註 |
|---|---|---|
| D1 | **anomaly 怎麼定義** | (a) 任何 defect pattern 都算 anomaly，test set 降採樣到 1–2%；(b) 維持原本 14.76% 的比例；(c) 只把少數幾類當 anomaly |
| D2 | **split 方式** | 沿用原始切分，或自己以 lot 為單位重新切（train / val / test）。固定 seed，三個 tier 共用同一份 index |
| D3 | **unlabeled 資料要不要用** | 不能直接當「正常」資料訓練（裡面可能混著 defect）。可以做成 contamination ablation |
| D4 | **輸入表示法** | 尺寸統一方式（resize / pad、目標大小）、channel 表示法（one-hot 3 channel / 二值 fail map） |

---

## 10. 已知坑一覽

| 坑 | 說明 | 處理 |
|---|---|---|
| pickle 讀不進來 | 舊版 pandas 存檔 | §2 的 `sys.modules` 別名 |
| 欄名拼錯 | `trianTestLabel` | clean 版已改成 `trainTestLabel` |
| label 是巢狀 array | `[['none']]` 不能直接比對字串 | clean 版已攤平 |
| 假的 non-null | unlabeled 存成空 array，不是 NaN | clean 版已改為 `'unlabeled'` |
| `none` ≠ 零壞 die | `none` 平均有約 10% 壞 die | 用空間 pattern 定義 anomaly |
| 尺寸不一 | 346 種尺寸 | resize 時用 nearest-neighbor |
| lot leakage | 同 lot 晶圓高度相似 | 以 `lotName` 做 group split |
| EDA 用到 test | 作業規定只能用 train | split 後在 train set 重算統計 |
