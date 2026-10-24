# Tier 1：Naive Baseline ＋ 共用評估模組

> 1. **第 5 節是 `src/evaluate.py` 的使用方式，Tier 2/3 請用這個模組算 metric**，確保三個 tier 的計算方式完全相同。
> 2. 第 3 節的數字是主結果表中 Tier 1 的那一列。
>
> 對應程式：[`src/tier1_baseline.py`](../src/tier1_baseline.py)、[`src/evaluate.py`](../src/evaluate.py)
> 最後更新：2026-10-24

---

## TL;DR

| 方法 | PR-AUC | ROC-AUC | F1 | Recall | FPR | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| T1a Predict-all-normal | 0.148 | 0.500 | 0.000 | 0.000 | 0.000 | **0.852** |
| T1b Fail-rate（τ = 3σ） | 0.474 | 0.748 | 0.293 | 0.176 | 0.004 | 0.875 |
| T1b Fail-rate（τ = val best-F1） | 0.474 | 0.748 | **0.423** | **0.407** | 0.089 | 0.836 |

- **Accuracy 陷阱**：什麼都不做的 T1a，accuracy 有 **85.2%**，但一個 defect 都沒抓到。所以 **accuracy 不能當主要指標**。
- **Tier 1 的效能下限**：PR-AUC **0.474**、ROC-AUC **0.748**。Tier 2/3 至少要超過這個數字。
- **H1 證實**：Near-full / Random 幾乎 100% 抓到，但 **Scratch 只有 9.2%**，跟 normal 被誤報的比例（8.9%）一樣，等於隨機。

---

## 1. 如何重現

```bash
python src/tier1_baseline.py
```

前提：`data/LSWMD_clean.pkl`、`splits/split_v1.csv` 已存在。執行約 1–2 分鐘（大部分是讀 pkl）。結果是確定性的，不需要多個 seed，std = 0。

輸出（`results/tier1/`）：

| 檔案 | 內容 |
|---|---|
| `metrics.csv` | 主結果：每個方法一列，包含所有 metric、τ、混淆矩陣、ms/sample |
| `per_type_recall.csv` | 各 failureType 的偵測率（`none` 那列 = 誤報率） |
| `scores_test.csv` | 每筆 test 樣本的 index、failureType、label、score、兩種 τ 的預測結果。**§8 挑 FP/FN 案例時用** |

---

## 2. 方法設計

### T1a：Predict-all-normal
- score = 0（所有晶圓都一樣），τ = ∞，所以永遠判為正常。
- 作用：展示 class imbalance 下的 **accuracy 陷阱**，也就是作業投影片說的「exposes class-imbalance pitfalls」。
- 它的 PR-AUC = 0.148，剛好等於 test 的 defect 比例，這是 PR-AUC 的「隨機基準線」。

### T1b：Fail-rate threshold
- **Anomaly score**：

  $$s(x) = \frac{\#\{\text{dies with value } 2\}}{\#\{\text{dies with value} > 0\}}$$

  也就是不良 die 佔全部 die 的比例。
- **Decision**：$s(x) \ge \tau$ 就判定為 anomaly。
- **為什麼選這個**：EDA 顯示它是單一特徵中最簡單、最直觀的（見 [eda.md](eda.md) Table 1），也最接近工程師的直覺：「壞太多就有問題」。
- **兩種 τ 策略**（都不使用 test）：

| 策略 | 怎麼決定 | τ 值 | 用到的資料 |
|---|---|---:|---|
| 3σ | train 正常樣本的 mean + 3 × std | **0.2652** | train normal（mean 0.1052、std 0.0533） |
| best-F1 | 在 val 上找讓 F1 最大的 τ | **0.1842** | val（含 defect label） |

符合作業規定：「validation set of labeled anomalies may be used strictly for threshold tuning」。

---

## 3. 結果（test set，34,493 片：29,397 normal / 5,096 defect）

### 主結果

| 方法 | PR-AUC | ROC-AUC | F1 | Precision | Recall | FPR | Accuracy | τ | ms / sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1a all-normal | 0.148 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.852 | ∞ | 0 |
| T1b 3σ | 0.474 | 0.748 | 0.293 | 0.885 | 0.176 | 0.004 | 0.875 | 0.265 | 0.011 |
| T1b best-F1 | 0.474 | 0.748 | 0.423 | 0.441 | 0.407 | 0.089 | 0.836 | 0.184 | 0.011 |

### 混淆矩陣

| 方法 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| T1a | 0 | 0 | 5,096 | 29,397 |
| T1b 3σ | 895 | 116 | 4,201 | 29,281 |
| T1b best-F1 | 2,073 | 2,627 | 3,023 | 26,770 |

計算成本：τ 的計算（算 train + val 的 fail rate）約 1.5 秒；推論約 **0.011 ms/sample**（CPU，單執行緒 pandas apply）。

### 各類別偵測率（驗證 H1）

| failureType | n (test) | T1b 3σ | T1b best-F1 |
|---|---:|---:|---:|
| none（= 誤報率） | 29,397 | 0.004 | 0.089 |
| Scratch | 251 | 0.020 | **0.092** |
| Edge-Ring | 1,887 | 0.028 | 0.267 |
| Loc | 726 | 0.088 | 0.333 |
| Edge-Loc | 991 | 0.146 | 0.380 |
| Center | 899 | 0.389 | 0.681 |
| Donut | 100 | 0.390 | 0.750 |
| Random | 211 | 0.991 | 0.995 |
| Near-full | 31 | 1.000 | 1.000 |

---

## 4. 解讀

1. **Accuracy 陷阱**：T1a 的 accuracy（85.2%）和 T1b 3σ（87.5%）只差 2.3 個百分點，但 recall 從 0 變成 17.6%。best-F1 的 recall 最高，accuracy 反而最低（83.6%）。**accuracy 跟真正要的偵測能力方向相反**，所以主要指標用 PR-AUC。
2. **τ 決定了 FP / FN 的取捨**，而 PR-AUC 和 ROC-AUC 不受 τ 影響（兩種 T1b 的分數相同）：
   - **3σ**：保守，precision 0.885，每 9 個警報約 8 個是真的，但漏抓 82% 的 defect。適合「工程師人力很少、警報一定要準」的情境。
   - **best-F1**：較積極，抓到 41% 的 defect，但 normal 有 8.9% 被誤報（2,627 片）。
   - 這兩種 τ 可以直接當作 **threshold strategy ablation** 的一部分。作業列的 AD-specific ablation 主題就包括「percentile vs best-F1 vs budget」。
3. **H1 證實**（[eda.md](eda.md) §6）：
   - Near-full、Random：99–100%。這些類別「壞很多」，fail rate 就夠了。
   - **Scratch：9.2%，和 normal 的誤報率 8.9% 幾乎相同**，代表 fail rate 對 Scratch 完全沒有辨識力，抓到的只是剛好壞比較多的幾片。
   - **Edge-Ring 只有 26.7%**，而它是數量最多的 defect（1,887 片），所以整體 recall 被拖低。EDA 顯示 `edge_fail_rate` 對 Edge-Ring 的 AUROC 有 0.93，支持 H2：需要位置資訊。
4. **EDA 預測準確**：test 的 ROC-AUC 0.748 和 train 上算的 0.744（eda.md Table 1）幾乎一樣，說明 split 之間沒有分布偏移，切分是健康的。

---

## 5. `src/evaluate.py` 使用方式（給 Tier 2 / Tier 3）

**約定：** label 1 = anomaly，0 = normal；**score 越高越異常**；判定規則為 `score >= tau`。

```python
import sys; sys.path.append("src")   # notebook 中使用時
from evaluate import evaluate, per_type_recall, find_best_f1_threshold

# 1. 在 VAL 上選 tau（不能用 test）
tau = find_best_f1_threshold(y_val, scores_val)

# 2. 在 TEST 上評估一次
metrics = evaluate(y_test, scores_test, tau)          # dict
recalls = per_type_recall(types_test, scores_test, tau)  # DataFrame: n, detected, rate
```

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `evaluate(y_true, scores, tau)` | label、score、τ | dict：`pr_auc`, `roc_auc`, `f1`, `precision`, `recall`, `fpr`, `accuracy`, `tau`, `tp`, `fp`, `fn`, `tn` |
| `per_type_recall(failure_types, scores, tau)` | 9 類 failureType、score、τ | 各類的 n、偵測數、偵測率 |
| `find_best_f1_threshold(y_val, scores_val)` | val 的 label 和 score | 讓 F1 最大的 τ |
| `predict(scores, tau)` | score、τ | 0/1 預測 |

**注意：**
- Autoencoder 的重建誤差本來就是越大越異常，可以直接用。如果模型輸出的是「正常程度」（例如 One-Class SVM 的 `decision_function`，越大越正常），要**先取負號**。
- 多個 seed 時，對每個 seed 各呼叫一次 `evaluate`，再算 mean ± std。
- 主結果表建議至少報告：**PR-AUC（主要）、ROC-AUC、F1、Recall、FPR、ms/sample**。

---

## 6. 給 paper 的素材

- **§5 Method / §6 Setup**：T1b 的 s(x) 公式和兩種 τ 的定義（第 2 節）。
- **§6 Metrics 選擇理由**：用 T1a 說明 accuracy 不可靠（85% accuracy、0 recall）；PR-AUC 的隨機基準 = 0.148（defect 比例）。
- **§7 Main results**：第 3 節主結果表的 Tier 1 列。
- **§7 Ablation**：3σ vs best-F1 是 threshold strategy ablation 的現成結果。
- **§8 Error analysis**：`scores_test.csv` 可以直接篩出 FP / FN 案例，例如 Scratch 被漏抓、高 fail rate 的 normal 被誤報。
