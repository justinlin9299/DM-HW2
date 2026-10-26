# Wafer Map Anomaly Detection on WM-811K

NYCU Data Mining HW2 — Anomaly Detection｜Team `<TEAM_ID>`

偵測半導體晶圓測試圖（wafer map）中的**異常空間圖樣**（defect pattern），並以三層方法比較（Tier 1 Naive Baseline / Tier 2 Primary Method / Tier 3 Strong Reference）。

| 成員 | 學號 | 負責 |
|---|---|---|
| `林哲寧` | `315554018` | 資料檢查、split、EDA、preprocessing、Tier 1 baseline、評估模組、repo |
| `賴育德` | `315551152` | Tier 2（合成 defect CNN、Conv AE ablation、和 Tier 1 的比較分析） |
| `<NAME>` | `<ID>` | Tier 3 |

---

## 任務定義

| 項目 | 內容 |
|---|---|
| 資料類別 | **Image** |
| 偵測單位 | 一片晶圓的 wafer map（2D 陣列，0 = 背景、1 = 良品 die、2 = 不良 die） |
| Normal | `failureType == "none"`：有不良 die，但**隨機散布**（正常良率損失） |
| Anomaly | 8 類 defect pattern 任一：Center、Donut、Edge-Loc、Edge-Ring、Loc、Near-full、Random、Scratch。不良 die **聚集成特定空間形狀**，代表製程異常 |
| 資料量 | 172,950 片有 label 的晶圓（147,431 normal / 25,519 anomaly，**1 : 5.8**） |
| 主要指標 | PR-AUC（主要）、ROC-AUC、F1、Recall、FPR；另報 per-type recall |

資料來源：[WM-811K（Kaggle）](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)。Wu, Jang & Chen, *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets*, IEEE Trans. Semiconductor Manufacturing, 2015.

---

## 目前結果

| Tier | 方法 | PR-AUC | ROC-AUC | F1 | Recall | FPR |
|---|---|---:|---:|---:|---:|---:|
| 1 | Predict-all-normal | 0.148 | 0.500 | 0.000 | 0.000 | 0.000 |
| 1 | Fail-rate threshold（τ = val best-F1） | 0.474 | 0.748 | 0.423 | 0.407 | 0.089 |
| 2 | CNN＋合成 defect（只用 train normal） | **0.818 ± 0.014** | **0.934 ± 0.009** | **0.748 ± 0.008** | **0.735 ± 0.013** | **0.040 ± 0.005** |
| 2 | ablation：Conv Autoencoder | 0.462 ± 0.048 | 0.703 ± 0.047 | 0.428 ± 0.049 | 0.339 ± 0.059 | 0.041 ± 0.009 |
| 3 | TODO | | | | | |

Test set：34,493 片（29,397 normal / 5,096 anomaly）。Tier 2 為 3 個 seed 的 mean ± std，τ 一律在 val 上以 best-F1 選擇。詳見 [docs/tier1.md](docs/tier1.md)、[docs/tier2.md](docs/tier2.md)。

---

## Quick start

`data/processed/*.npz`（已 resize 成 64×64，約 38 MB）**已經在 repo 裡**，clone 下來就能直接訓練，不需要下載原始資料。

```bash
git clone https://github.com/justinlin9299/DM-HW2.git
cd DM-HW2
pip install -r requirements.txt
```

```python
import sys; sys.path.append("src")
from data import load_split, to_onehot
from evaluate import evaluate, per_type_recall, find_best_f1_threshold

X_train, y_train, ft_train, idx_train = load_split("train")   # (103312, 64, 64) uint8
X_val,   y_val,   ft_val,   idx_val   = load_split("val")
X_test,  y_test,  ft_test,  idx_test  = load_split("test")

X_train_normal = to_onehot(X_train[y_train == 0])   # one-class 模型只能用 train normal

# ... 訓練模型，得到 scores_val、scores_test（越高越異常）...

tau = find_best_f1_threshold(y_val, scores_val)     # τ 只能在 val 上選
print(evaluate(y_test, scores_test, tau))           # test 只評估一次
print(per_type_recall(ft_test, scores_test, tau))
```

**必讀規則**（細節見 [docs/split.md](docs/split.md) §5）：

| 用途 | 可以用的資料 |
|---|---|
| one-class / reconstruction 模型訓練 | **只能**用 train 且 `y == 0` |
| 監督式模型訓練 | train（normal + anomaly） |
| 選 τ、調超參數、early stopping | val |
| 最終結果 | test，**只評估一次** |
| 需要 fit 的前處理（PCA、scaler） | **只在 train 上 fit** |

---

## 資料夾結構

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── LSWMD.pkl              # 原始資料（不在 repo，見「從零重現」步驟 1）
│   ├── LSWMD_clean.pkl        # 攤平 label 後的版本（不在 repo，步驟 2 產生）
│   └── processed/             # 64×64 模型輸入（在 repo 中）
│       ├── train.npz
│       ├── val.npz
│       └── test.npz
├── splits/
│   └── split_v1.csv           # 三個 tier 共用的 train/val/test 切分
├── src/
│   ├── make_split.py          # 產生 split
│   ├── preprocess.py          # resize → npz
│   ├── data.py                # load_split / to_onehot / to_fail_mask
│   ├── evaluate.py            # 共用評估函式（三個 tier 都要用）
│   ├── tier1_baseline.py      # Tier 1
│   └── CNN.py                 # Tier 2（在 Colab GPU 上訓練）
├── scripts/
│   └── download_data.py       # 從 Kaggle 下載原始資料
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   └── 02_eda.ipynb
├── figures/                   # paper 用圖（PNG + PDF）
│   ├── eda/
│   ├── preprocessing/
│   └── tier2/
├── results/                   # 各步驟的數值結果（CSV）
│   ├── eda/
│   ├── tier1/
│   └── tier2/                 # synth/（主方法）、ae/（ablation），含模型權重
└── docs/                      # 各步驟的詳細說明
```

---

## 文件導覽

建議閱讀順序：

| 文件 | 內容 | paper 對應章節 |
|---|---|---|
| [docs/data_notes.md](docs/data_notes.md) | 資料結構、欄位、label 分布、已知坑 | §3 |
| [docs/split.md](docs/split.md) | 切分方式、設計理由、**使用規則** | §6 |
| [docs/eda.md](docs/eda.md) | 分布對比、分離度、空間 pattern、假設 H1–H4 | §4 |
| [docs/tier1.md](docs/tier1.md) | Tier 1 設計與結果、**`evaluate.py` 用法** | §5–§7 |
| [docs/preprocessing.md](docs/preprocessing.md) | resize 方式、品質檢查、**載入方式與表示法選擇** | §5–§6 |
| [docs/tier2.md](docs/tier2.md) | Tier 2 設計（合成 defect）、結果、和 Tier 1 的比較、AE ablation | §5–§8 |

---

## 主要發現（截至目前）

1. **Accuracy 會誤導**：什麼都不做的 predict-all-normal 有 85.2% accuracy，但 recall = 0。
2. **只看「壞多少」不夠**：fail rate 單一特徵的 PR-AUC 只有 0.474。它抓得到 Near-full、Random（≥ 99%），但 **Scratch 的偵測率只有 9.2%，和 normal 的誤報率（8.9%）相同**。
3. **anomaly 是空間 pattern**：加入外圈 / 中心位置的資訊後，Edge-Ring 的 AUROC 從 0.73 升到 0.93，Center 從 0.85 升到 0.95（train split）。
4. **split 以 lot 分組**：同一批次（lot）的晶圓不會同時出現在 train 和 test，避免 leakage。
5. **看形狀的 CNN 大幅勝出**：Tier 2 只用正常晶圓加上合成 defect 訓練，PR-AUC 就從 0.474 升到 **0.818**（bootstrap 95% CI 的差距為 [0.348, 0.377]）。Edge-Ring 從 27% 升到 89%，Scratch 從 9% 升到 63%，誤報率則從 8.9% 降到 4.0%。
6. **重建式 Autoencoder 沒有用**：PR-AUC 0.462，和 Tier 1 差不多。normal 晶圓本身就有隨機壞 die，重建誤差會被這些雜訊淹沒。

---
