# 比較方法：Isolation Forest（手工空間特徵，one-class）

> 對應程式：[`src/iforest.py`](../src/iforest.py)｜主表與顯著性：[`src/compare.py`](../src/compare.py) → `results/compare/`
> 最後更新：2026-11-02

---

## TL;DR

| 方法 | PR-AUC | ROC-AUC | F1 | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| T1b Fail-rate（best-F1） | 0.474 | 0.748 | 0.423 | 0.407 | 0.089 |
| **Isolation Forest（手工特徵）** | **0.685 ± 0.012** | **0.907 ± 0.006** | **0.611 ± 0.013** | **0.637 ± 0.007** | **0.078 ± 0.010** |
| T2 CNN（合成 defect） | 0.818 ± 0.014 | 0.934 ± 0.009 | 0.748 ± 0.008 | 0.735 ± 0.013 | 0.040 ± 0.005 |

（3 個 seed 的 mean ± std，test set）

- **傳統 data mining 的 one-class 方法**：只用 train normal 訓練，和 T2 一樣不使用任何真實 defect label。
- **介於 Tier 1 和 T2 之間**：
  - 比 fail-rate 高 +0.211（bootstrap 95% CI [0.204, 0.217]），表示人工設計的空間特徵有用。
  - 比 T2 低 −0.133（CI [0.121, 0.144]，Welch p = 0.0006），表示 CNN 學到了手工特徵抓不到的形狀資訊。
- **最大差距在 Edge-Ring 和 Scratch**：
  - Edge-Ring：IF 57% vs T2 89%。
  - Scratch：IF 28% vs T2 63%。
  - Center（78%）、Edge-Loc（60%）、Loc（66%）反而略高於 T2，但 IF 的誤報率是 7.8%，約為 T2（4.0%）的兩倍，所以這三類的比較不在同一個操作點上。看 threshold-free 的 per-type ROC-AUC，兩者在這三類大致相當（0.89–0.93）。

---

## 1. 如何重現

```bash
python src/iforest.py                  # 3 seeds，約 8 分鐘（CPU，主要是特徵抽取）
python src/iforest.py --limit 3000     # smoke test（結果不能拿來報告）
python src/compare.py                  # 更新主表與顯著性
```

只需要 numpy、scipy、scikit-learn（requirements.txt 已包含）。輸出在 `results/iforest/`，格式和 `results/tier2/` 一致：`metrics.csv`、`per_type_recall.csv`、`scores_test.csv`。

## 2. 方法設計

### 2.1 特徵（每片 61 維，在 64×64 的 processed map 上計算）

半徑 r 的定義和 [eda.md](eda.md) §2 相同：以晶圓自己的中心計算，最外圈 die 的 r = 1。

| 群組 | 維度 | 內容 | 針對的 pattern |
|---|---:|---|---|
| 全域 | 2 | fail rate；3×3 median filter 後的 fail rate（濾掉孤立的壞 die） | Near-full、Random |
| 分區密度 | 13 | 參考 Wu et al. 2015：中心圓（r < 0.33）1 區、中圈（0.33–0.75）4 個象限、外圈（r ≥ 0.75）8 個扇區 | Center、Donut、Edge-Ring |
| 外圈摘要 | 3 | 8 個外圈扇區密度的 max / min / std（不受旋轉影響） | Edge-Loc |
| Radon | 32 | fail mask 在 12 個角度的投影；各投影位置跨角度的 mean 和 std，平均池化成 16 bins | Scratch（線狀）、環狀 |
| 幾何 | 11 | median filter 後的 8-連通 component：數量、最大 component 的面積 / die 數、佔全部 fail 的比例、質心半徑、離心率、長短軸、bounding-box 填滿率、半徑 mean / std、第二大 component 的面積 | Loc、Scratch |

和 Wu et al. 的差異：Radon 投影改用平均池化（原文用 cubic interpolation 到 20 點）；幾何特徵用 component 的二階矩估計，沒有算 convex hull（solidity）。

### 2.2 模型、score 與 τ

| 項目 | 值 |
|---|---|
| 模型 | `sklearn.ensemble.IsolationForest`，300 棵樹，其餘為預設值（`max_samples = 256`） |
| 訓練資料 | train normal（87,970 片） |
| Score | `−score_samples(x)`，越高越異常 |
| τ | val 上的 best-F1（`find_best_f1_threshold`） |
| Seeds | 0、1、2（`random_state`；特徵是 deterministic） |

特徵和超參數都沒有在 val 上調過。

## 3. 結果（test set）

### 各類別偵測率（at τ，seed 平均）與 ROC-AUC（vs 全部 test normal）

| failureType | n | IF recall | T2 recall | IF ROC-AUC | T2 ROC-AUC |
|---|---:|---:|---:|---:|---:|
| none（誤報率） | 29,397 | 0.078 | 0.040 | — | — |
| Scratch | 251 | 0.280 | 0.627 | 0.771 | 0.897 |
| Edge-Ring | 1,887 | 0.567 | 0.892 | 0.909 | 0.975 |
| Edge-Loc | 991 | 0.598 | 0.553 | 0.891 | 0.892 |
| Loc | 726 | 0.655 | 0.624 | 0.903 | 0.896 |
| Center | 899 | 0.782 | 0.646 | 0.929 | 0.911 |
| Donut | 100 | 0.943 | 0.930 | 0.981 | 0.985 |
| Random | 211 | 1.000 | 0.945 | 1.000 | 0.990 |
| Near-full | 31 | 1.000 | 1.000 | 1.000 | 0.999 |

### 其他

| | IF | T2 |
|---|---:|---:|
| Recall @ 5% FPR | 0.556 | 0.763 |
| 推論時間（ms / 片，含特徵抽取） | 1.75 | 0.014（GPU） |

## 4. 解讀

1. **空間特徵確實有效**：同樣不看任何 defect label，只是從「壞多少」改成「壞在哪、長什麼形狀」，PR-AUC 就從 0.474 升到 0.685。Edge-Ring 的 ROC-AUC 從 0.74 升到 0.91，Scratch 從 0.42 升到 0.77。
2. **但手工特徵有上限**：
   - Edge-Ring 只有在外圈「幾乎整圈都壞」時才會讓 13 區密度明顯偏離；較細、較淡的 ring 和 normal 外圈的隨機壞 die 重疊。
   - Scratch 很細，經過 64×64 resize 和 median filter 後容易斷裂，Radon 投影也會被背景的隨機壞 die 稀釋。
   - T2 的 CNN 直接從像素學到這兩類的形狀，是它領先的主要來源。
3. **Isolation Forest 的 one-class 假設**：它只知道「normal 長什麼樣」，所以只要是罕見的特徵組合都會被當成異常。normal 晶圓本身的 fail rate 變異很大，高 fail rate 的 normal 晶圓就成了主要的誤報來源，誤報率因此偏高（7.8%）。以 seed 0 為例：誤報晶圓的 fail rate 中位數是 0.205，判對的 normal 只有 0.096；84% 的誤報晶圓，fail rate 高於判對 normal 的第 90 百分位。T2 用合成 defect 明確告訴模型「哪些偏離才算異常」，誤報率只有 4.0%。

## References

- Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). Wafer map failure pattern recognition and similarity ranking for large-scale data sets. *IEEE Trans. Semiconductor Manufacturing*, 28(1), 1–12.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *ICDM 2008*, 413–422.
