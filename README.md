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
| 2 | CNN＋合成 defect（只用 train normal） | 0.818 ± 0.014 | 0.934 ± 0.009 | 0.748 ± 0.008 | 0.735 ± 0.013 | 0.040 ± 0.005 |
| 2 | ablation：Conv Autoencoder | 0.462 ± 0.048 | 0.703 ± 0.047 | 0.428 ± 0.049 | 0.339 ± 0.059 | 0.041 ± 0.009 |
| 3 | PatchCore（one-class SOTA，只用 train normal） | 0.812 ± 0.005 | 0.932 ± 0.002 | 0.725 ± 0.005 | 0.693 ± 0.023 | 0.038 ± 0.005 |
| 3 | ResNet-18 監督式（真實 9 類 label，上限） | **0.991 ± 0.000** | **0.998 ± 0.000** | **0.953 ± 0.001** | **0.946 ± 0.002** | **0.007 ± 0.000** |
| 3 | control：Tier 2 的 CNN 改用監督式 | 0.988 ± 0.000 | 0.997 ± 0.000 | 0.944 ± 0.002 | 0.929 ± 0.004 | 0.007 ± 0.000 |

Test set：34,493 片（29,397 normal / 5,096 anomaly）。Tier 2 / 3 為 3 個 seed 的 mean ± std，τ 一律在 val 上以 best-F1 選擇。T2 vs PatchCore：+0.006（bootstrap 95% CI [−0.003, +0.015]，p = 0.62，無顯著差異）；監督式 ResNet-18 − T2：+0.173（CI [0.164, 0.181]，p = 0.003）。詳見 [docs/tier1.md](docs/tier1.md)、[docs/tier2.md](docs/tier2.md)、[docs/tier3.md](docs/tier3.md)；ablation 見 [docs/ablation.md](docs/ablation.md)。

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
│   ├── CNN.py                 # Tier 2 / Tier 3 監督式 / ablation 的訓練（GPU）
│   ├── patchcore.py           # Tier 3 PatchCore（one-class）
│   ├── compare.py             # 三個 tier 的主結果表 + 顯著性檢定
│   └── ablation.py            # ablation 總表、threshold ablation、圖
├── scripts/
│   ├── download_data.py       # 從 Kaggle 下載原始資料
│   └── run_tier3_ablation.sh  # Tier 3 + 所有 retrained ablation
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   └── 02_eda.ipynb
├── figures/                   # paper 用圖（PNG + PDF）
│   ├── eda/
│   ├── preprocessing/
│   ├── tier2/
│   └── ablation/
├── results/                   # 各步驟的數值結果（CSV）
│   ├── eda/
│   ├── tier1/
│   ├── tier2/                 # synth/（主方法）、ae/（ablation），含模型權重
│   ├── tier3/                 # patchcore/、resnet/、sup/（control）；模型權重不在 repo
│   ├── ablation/              # 每個設定一個資料夾（不含權重）+ summary / leave_one_out / threshold.csv
│   └── compare/               # 主結果表、顯著性、per-type recall / AUROC
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
| [docs/tier3.md](docs/tier3.md) | Tier 3 PatchCore ＋ 監督式 ResNet-18、三個 tier 主結果表、顯著性、比較解讀 | §2、§5–§7 |
| [docs/ablation.md](docs/ablation.md) | 4 個 controlled ablation：污染、leave-one-type-out、head、threshold | §7–§9 |

---

## 主要發現（截至目前）

1. **Accuracy 會誤導**：什麼都不做的 predict-all-normal 有 85.2% accuracy，但 recall = 0。
2. **只看「壞多少」不夠**：fail rate 單一特徵的 PR-AUC 只有 0.474。它抓得到 Near-full、Random（≥ 99%），但 **Scratch 的偵測率只有 9.2%，和 normal 的誤報率（8.9%）相同**。
3. **anomaly 是空間 pattern**：加入外圈 / 中心位置的資訊後，Edge-Ring 的 AUROC 從 0.73 升到 0.93，Center 從 0.85 升到 0.95（train split）。
4. **split 以 lot 分組**：同一批次（lot）的晶圓不會同時出現在 train 和 test，避免 leakage。
5. **看形狀的 CNN 大幅勝出**：Tier 2 只用正常晶圓加上合成 defect 訓練，PR-AUC 就從 0.474 升到 **0.818**（bootstrap 95% CI 的差距為 [0.348, 0.377]）。Edge-Ring 從 27% 升到 89%，Scratch 從 9% 升到 63%，誤報率則從 8.9% 降到 4.0%。
6. **重建式 Autoencoder 沒有用**：PR-AUC 0.462，和 Tier 1 差不多。normal 晶圓本身就有隨機壞 die，重建誤差會被這些雜訊淹沒。
7. **和 one-class SOTA PatchCore 打平**（0.818 vs 0.812，差距不顯著），但推論快約 60 倍。兩者強項不同：T2 擅長 Edge-Ring（全域形狀），PatchCore 擅長 Loc / Scratch（局部異常）。
8. **監督式上限是 0.991，T2 顯著輸給它**（差 0.173）。但同一個小 CNN 改用真實 label 就有 0.988，所以差距來自**合成 defect 和真實 defect 的分布差異**，不是模型容量。
9. **Normal pool 必須乾淨**：只要混入 1% 未標註的真實 defect，T2 的 PR-AUC 就從 0.818 掉到 0.638。
10. **能偵測沒合成過的類別**：8 類中有 6 類不合成時 ROC-AUC 只掉 < 0.02；唯一的盲點是 Scratch（0.918 → 0.792）。但拿掉合成 Scratch 會讓誤報率從 4.1% 降到 2.3%，整體 PR-AUC 反而 +0.063。

---
