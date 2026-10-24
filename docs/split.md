# Train / Val / Test 切分（Split）

> **三個 tier 都必須使用 `splits/split_v1.csv`**，不要重新切。作業規定所有方法要用同一份 split。
>
> 對應程式：[`src/make_split.py`](../src/make_split.py)
> 最後更新：2026-10-24

---

## TL;DR

- 檔案：`splits/split_v1.csv`（172,950 列，約 5.6 MB，**會放進 repo**）
- 比例：train / val / test ≈ **60 / 20 / 20**
- **以 `lotName` 分組**：同一個 lot 只會出現在其中一份，避免 leakage
- **以 9 類 `failureType` 分層**：三份的 defect 比例都約 14.8%，8 類 defect 在每一份都有
- `label`：`0` = 正常（`none`），`1` = anomaly（8 類 defect 任一）
- seed = 42，可重現（同 seed 重跑，檔案的 MD5 完全相同）

---

## 1. 如何重現

```bash
python src/make_split.py
# 等同於
python src/make_split.py --data data/LSWMD_clean.pkl --out splits/split_v1.csv --seed 42
```

前提：`data/LSWMD_clean.pkl` 已存在（由 `notebooks/01_data_inspection.ipynb` 產生，見 [data_notes.md](data_notes.md)）。

程式最後會自動檢查以下三點，任何一點不成立就報錯：
1. 每一筆都有被分配到某一份
2. **沒有任何 lot 同時出現在兩份**
3. 每一類 `failureType` 在三份裡都至少有 1 筆

---

## 2. 檔案格式

```
index,lotName,waferIndex,failureType,label,split
0,lot1,1,none,0,val
1,lot1,2,none,0,val
...
```

| 欄位 | 說明 |
|---|---|
| `index` | `LSWMD_clean.pkl` 的原始 index，**用它對回 `waferMap`** |
| `lotName` | 批號（分組依據） |
| `waferIndex` | 在該 lot 中是第幾片（1–25） |
| `failureType` | 9 類原始 label（`none` + 8 類 defect） |
| `label` | 二元 label：0 = normal，1 = anomaly |
| `split` | `train` / `val` / `test` |

CSV 只存 index，**不存 waferMap**，所以檔案很小。

---

## 3. 切分結果

### 總覽

| split | 晶圓數 | 佔比 | lot 數 | normal | defect | defect 比例 |
|---|---:|---:|---:|---:|---:|---:|
| train | 103,312 | 59.7% | 6,452 | 87,970 | 15,342 | 14.85% |
| val | 35,145 | 20.3% | 2,149 | 30,064 | 5,081 | 14.46% |
| test | 34,493 | 19.9% | 2,161 | 29,397 | 5,096 | 14.77% |
| **合計** | **172,950** | 100% | 10,762 | 147,431 | 25,519 | 14.76% |

比例不是剛好 60/20/20，因為整個 lot 只能放在同一邊，屬於正常誤差。

### 各類別分布

| failureType | train | val | test |
|---|---:|---:|---:|
| none | 87,970 | 30,064 | 29,397 |
| Edge-Ring | 5,833 | 1,960 | 1,887 |
| Edge-Loc | 3,140 | 1,058 | 991 |
| Center | 2,654 | 741 | 899 |
| Loc | 2,128 | 739 | 726 |
| Scratch | 693 | 249 | 251 |
| Random | 451 | 204 | 211 |
| Donut | 353 | 102 | 100 |
| Near-full | 90 | 28 | 31 |

---

## 4. 設計理由（paper §6 可直接引用）

| 決定 | 理由 |
|---|---|
| 只用有 label 的資料 | 作業規定 test set 必須有 ground truth；unlabeled 不能保證是正常資料 |
| 不沿用原作者的切分 | 原始切分的 defect 比例不一致（Training 32.4% vs Test 6.7%），各類分布也偏差很大（見 data_notes.md §6） |
| 以 lot 分組 | 同一個 lot 的晶圓來自同一批製程，pattern 高度相似。拆到 train 和 test 兩邊會造成 leakage，分數虛高 |
| 以 9 類分層，而不是 0/1 | 確保稀有類別（Near-full、Donut）在每一份都有，§8 的 per-type 錯誤分析才做得了 |
| test 維持自然比例（14.8%） | 不丟棄稀有類別樣本；PR-AUC 本身就能反映 imbalance |
| `StratifiedGroupKFold` 兩階段 | 先取 1/5 當 test，剩下的再取 1/4 當 val，結果是 60/20/20 |

---

## 5. 使用規則

| 用途 | 可以用的資料 |
|---|---|
| **Tier 2 one-class / reconstruction 模型訓練** | **只能**用 `split == "train" and label == 0` |
| Tier 3 監督式模型訓練 | `split == "train"`（normal + defect 都可以用） |
| 調 threshold τ、選超參數、early stopping | `split == "val"`（可以用 defect label） |
| 最終結果 | `split == "test"`，**只在最後評估一次**，不能拿來調任何東西 |
| EDA 統計（paper §4） | **只用 `split == "train"`** |
| Preprocessing 的 fit（例如 scaler 的 mean/std） | **只用 `split == "train"`** |
| Contamination ablation | 從 `split == "train" and label == 1` 依比例混進訓練集 |

---

## 6. 讀取範例

```python
import pandas as pd

df = pd.read_pickle("data/LSWMD_clean.pkl")
split = pd.read_csv("splits/split_v1.csv", index_col="index")

data = df.loc[split.index].join(split[["label", "split"]])

train_normal = data[(data["split"] == "train") & (data["label"] == 0)]   # Tier 2 訓練用
val          = data[data["split"] == "val"]
test         = data[data["split"] == "test"]
```

`df.loc[split.index]` 會用原始 index 對回 waferMap。`failureType` 和 `lotName` 兩邊都有，所以 join 時只取 `label` 和 `split`。

---

## 7. 版本紀錄

| 版本 | 日期 | 說明 |
|---|---|---|
| v1 | 2026-10-24 | 初版：labeled only、lot-grouped、9 類分層、60/20/20、seed=42 |

> 如果之後要改切分方式，請產生 `split_v2.csv`，**不要覆蓋 v1**，並在這裡記錄原因。
