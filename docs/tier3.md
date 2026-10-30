# Tier 3：Strong / SOTA Reference（PatchCore ＋ 監督式 ResNet-18）

> 1. 第 3 節的表就是 paper §7 的主結果表（三個 tier 同一個 test set）。
> 2. 第 4 節是顯著性檢定，第 5、6 節是「T2 為什麼和 PatchCore 打平、為什麼輸給監督式」的解讀。
>
> 對應程式：[`src/patchcore.py`](../src/patchcore.py)、[`src/CNN.py`](../src/CNN.py)（`--method resnet` / `--method sup`）、[`src/compare.py`](../src/compare.py)｜執行腳本：[`scripts/run_tier3_ablation.sh`](../scripts/run_tier3_ablation.sh)
> 最後更新：2026-10-30

---

## TL;DR

我們用**兩個** Tier 3，分別回答兩個問題：

| Tier 3 | 角色 | 回答的問題 |
|---|---|---|
| **PatchCore**（Roth et al., CVPR 2022） | **同 paradigm 的 SOTA**：one-class，只用 train normal | T2 有沒有贏過不用 label 的 SOTA？ |
| **ResNet-18 監督式** | **上限（ceiling）**：用真實 9 類 label | T2 離「有 label」的上限還有多遠？ |

| 方法 | 訓練用的 label | PR-AUC | ROC-AUC | F1 | Recall | FPR | ms / sample |
|---|---|---:|---:|---:|---:|---:|---:|
| T1 Fail-rate（best-F1） | 無 | 0.474 | 0.748 | 0.423 | 0.407 | 0.089 | 0.013 (CPU) |
| **T2 CNN＋合成 defect** | **無**（只用 train normal） | **0.818 ± 0.014** | **0.934 ± 0.009** | **0.748 ± 0.008** | **0.735 ± 0.013** | 0.040 ± 0.005 | 0.014 |
| **T3 PatchCore** | **無**（只用 train normal） | 0.812 ± 0.005 | 0.932 ± 0.002 | 0.725 ± 0.005 | 0.693 ± 0.023 | **0.038 ± 0.005** | 0.839 |
| T3 ResNet-18（監督式） | 真實 9 類 label | 0.991 ± 0.000 | 0.998 ± 0.000 | 0.953 ± 0.001 | 0.946 ± 0.002 | 0.007 ± 0.000 | 0.021 |
| T3 control：T2 的 CNN 改監督式 | 真實 9 類 label | 0.988 ± 0.000 | 0.997 ± 0.000 | 0.944 ± 0.002 | 0.929 ± 0.004 | 0.007 ± 0.000 | 0.003 |

（3 個 seed 的 mean ± std，test set 34,493 片；Tier 1 是 deterministic，只有一次。粗體 = 不用 label 的方法中最好的）

- **T2 和 PatchCore 打平**：PR-AUC 差 +0.006，**bootstrap 95% CI [−0.003, +0.015] 包含 0，Welch p = 0.62**。所以只能寫「comparable」，不能寫「better」。
- **兩者擅長的類別不同**：T2 在數量最多的 Edge-Ring 明顯較好（ROC-AUC 0.975 vs 0.940）；PatchCore 在 Loc、Scratch 較好（0.929 / 0.939 vs 0.896 / 0.897）。
- **T2 便宜很多**：推論快約 60 倍（0.014 vs 0.84 ms / 片），模型 98k 參數 vs WideResNet-50（69M）＋ 45k × 1536 的 memory bank。
- **監督式上限 0.991**，T2 顯著輸給它（+0.173，CI [0.164, 0.181]，p = 0.003）。同一個小 CNN 改用真實 label 就有 0.988，所以**差距來自 label，不是模型容量**。

---

## 1. 如何重現

```bash
pip install torch torchvision                  # ImageNet 權重會自動下載
python -u src/patchcore.py                     # T3 PatchCore，3 seeds → results/tier3/patchcore/（每個 seed 約 6 分鐘）
python -u src/CNN.py --method resnet           # T3 ResNet-18，3 seeds → results/tier3/resnet/
python -u src/CNN.py --method sup              # T3 control → results/tier3/sup/
python src/compare.py                          # 主結果表 + 顯著性 → results/compare/
# 或一次跑完 Tier 3 + 所有 ablation：
bash scripts/run_tier3_ablation.sh
```

輸出格式和 `results/tier2/` 完全相同（`metrics.csv`、`per_type_recall.csv`、`scores_test.csv`；CNN 類另有 `history_seed*.csv`、`config.json`）。
ResNet-18 的權重每個 45 MB、PatchCore 的 memory bank 每個 138 MB，**都沒有放進 repo**，重跑即可。

硬體：RTX 4090，PyTorch 2.6.0 + CUDA 12.4。Tier 2 原本的數字是在 Colab GPU 上跑的；我們在 4090 上用相同指令重跑 Tier 2（`results/ablation/baseline`）得到 PR-AUC 0.818 ± 0.022，和 Colab 的 0.818 ± 0.014 一致。

---

## 2. 方法設計

### 2.1 為什麼選這兩個

- **PatchCore**（Roth et al., CVPR 2022）是 one-class 工業瑕疵偵測的代表性 SOTA（MVTec AD image-level AUROC 最高 99.6%），也是 anomalib 等公開套件的標準方法。它和 T2 **同一個 paradigm**（只看正常樣本），所以是最公平的「強對手」。
- **監督式 CNN** 是 WM-811K 上的主流做法（Nakazawa & Kulkarni 2018；Kyeong & Kim 2018）。它使用 T2 刻意不用的真實 defect label，所以代表的是**上限**，而不是同條件的競爭者。ResNet-18（He et al., 2016）是標準 backbone。
- 作業規定 Tier 3 可以使用 public package 與 pretrained weights（需引用）。

### 2.2 PatchCore 設定（`src/patchcore.py`）

| 步驟 | 設定 | 和原論文的差異 |
|---|---|---|
| 輸入 | 64×64 one-hot 3 channel，ImageNet 正規化，nearest 放大到 128×128 | 原論文 224（自然影像）；wafer 只有 64×64，放大到 224 不會多資訊 |
| Backbone | torchvision WideResNet-50-2，ImageNet 權重，凍結 | 相同 |
| Patch 特徵 | layer2 + layer3，各自 3×3 平均（locally aware），layer3 放大到 layer2 的 16×16 → 每片 256 個 1536 維 patch | 相同 |
| Memory bank | **全部** 87,970 片 train normal 的 22.5M 個 patch → greedy coreset 0.2% = 45,041 個 | 原論文 coreset 1–10%（MVTec 每類只有幾百張）；我們的 patch 數多 100 倍以上 |
| Coreset 距離 | 隨機投影到 128 維（Johnson-Lindenstrauss）再做 greedy k-center；bank 保留完整 1536 維 | 相同 |
| Score | $s(x) = \max_{p \in x} \min_{m \in \mathcal{M}} \lVert f_p - m \rVert_2$（patch 到 bank 最近鄰距離的最大值） | 沒有使用原論文可選的 re-weighting |
| τ | val best-F1（和所有 tier 相同） | — |
| Seed | 決定隨機投影與 coreset 起點，3 個 seed | — |

**沒有任何訓練**（backbone 凍結），fit 只是抽特徵＋選 coreset，每個 seed 351 秒。沒有調超參數（size、coreset 比例都是依記憶體 / 時間預先決定，沒有看 val 或 test 結果）。

### 2.3 監督式 ResNet-18 設定（`src/CNN.py --method resnet`）

| 項目 | T3 ResNet-18 | T3 control（sup CNN） |
|---|---|---|
| 訓練資料 | train 全部 103,312 片（87,970 normal + 15,342 defect），真實 9 類 label | 同左 |
| 模型 | torchvision ResNet-18，ImageNet 權重，fc 換成 9 類，11.2M 參數 | T2 的 `WaferCNN`（16-32-64-128），98k 參數 |
| 輸入 | 64×64 one-hot，nearest 放大到 128×128 | 64×64 one-hot |
| Score | $s(x) = \log\sum_{k=1}^{8} e^{z_k} - z_0$（**和 T2 相同**） | 同左 |
| 其他 | 和 T2 相同：AdamW lr 1e-3、wd 1e-4、cosine、batch 256、dihedral 增強、最多 30 epoch、val PR-AUC early stopping（patience 5）、τ = val best-F1、seeds 0/1/2 | 同左 |

**Method-specific preprocessing（paper 要交代）**：PatchCore 和 ResNet-18 會把輸入放大到 128×128（讓 stride-32 的 ImageNet backbone 不會把 64×64 壓得太小），PatchCore 另外做 ImageNet 正規化。T2 和 T3 control 直接用 64×64。split、τ 規則、`evaluate.py` 全部相同。

---

## 3. 主結果（paper §7 主表）

來源：`results/compare/main_table.csv`

| 方法 | PR-AUC | ROC-AUC | F1 | Recall | FPR | Recall@5%FPR | ms / sample | fit 時間 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 Fail-rate（best-F1） | 0.474 | 0.748 | 0.423 | 0.407 | 0.089 | 0.332 | 0.013 (CPU) | 2 s |
| **T2 CNN＋合成 defect** | **0.818 ± 0.014** | **0.934 ± 0.009** | **0.748 ± 0.008** | **0.735 ± 0.013** | 0.040 ± 0.005 | **0.763** | 0.014 | 86 s (Colab) |
| T3 PatchCore | 0.812 ± 0.005 | 0.932 ± 0.002 | 0.725 ± 0.005 | 0.693 ± 0.023 | **0.038 ± 0.005** | 0.731 | 0.839 | 351 s |
| T3 ResNet-18（監督式） | 0.991 ± 0.000 | 0.998 ± 0.000 | 0.953 ± 0.001 | 0.946 ± 0.002 | 0.007 ± 0.000 | 0.992 | 0.021 | 188 s |
| T3 control（sup CNN） | 0.988 ± 0.000 | 0.997 ± 0.000 | 0.944 ± 0.002 | 0.929 ± 0.004 | 0.007 ± 0.000 | 0.988 | 0.003 | 32 s |

- 粗體 = 不使用 defect label 的方法（T1、T2、PatchCore）中最好的。監督式方法是上限，不參與粗體。
- ms / sample 都是 GPU batch 推論（T1 為 CPU），只能看數量級。PatchCore 慢是因為每個 patch 都要和 45k 個 bank patch 算距離。
- PatchCore 的 val PR-AUC 0.800 / 0.815 / 0.805，和 test（0.809 / 0.819 / 0.808）一致；seed 波動（0.005）比 T2（0.014）小，因為沒有 SGD 訓練。

### 各類別偵測率（at τ，seed 平均）

來源：`results/compare/per_type_recall.csv`

| failureType | n (test) | T1 | T2 | PatchCore | ResNet-18 |
|---|---:|---:|---:|---:|---:|
| none（= 誤報率） | 29,397 | 0.089 | 0.040 | 0.038 | 0.007 |
| Edge-Loc | 991 | 0.380 | 0.553 | 0.646 | 0.900 |
| Loc | 726 | 0.333 | 0.624 | 0.725 | 0.886 |
| Scratch | 251 | 0.092 | 0.627 | 0.737 | 0.875 |
| Center | 899 | 0.681 | 0.646 | 0.719 | 0.956 |
| Edge-Ring | 1,887 | 0.267 | **0.892** | 0.639 | 0.992 |
| Donut | 100 | 0.750 | 0.930 | 0.943 | 0.993 |
| Random | 211 | 0.995 | 0.945 | 0.956 | 0.965 |
| Near-full | 31 | 1.000 | 1.000 | 1.000 | 1.000 |

### 各類別 ROC-AUC（vs 全部 test normal，不受 τ 影響）

來源：`results/compare/per_type_auroc.csv`

| failureType | T1 | T2 | PatchCore | ResNet-18 |
|---|---:|---:|---:|---:|
| Edge-Loc | 0.751 | 0.892 | **0.901** | 0.996 |
| Loc | 0.666 | 0.896 | **0.929** | 0.995 |
| Scratch | 0.422 | 0.897 | **0.939** | 0.992 |
| Center | 0.831 | 0.911 | **0.922** | 0.999 |
| Edge-Ring | 0.740 | **0.975** | 0.940 | 1.000 |
| Donut | 0.927 | 0.985 | **0.993** | 1.000 |
| Random | 0.999 | 0.990 | **0.992** | 0.999 |
| Near-full | 1.000 | 0.999 | **1.000** | 1.000 |

（粗體 = T2 和 PatchCore 中較好的）

---

## 4. 顯著性檢定

來源：`results/compare/significance.csv`。兩種檢定：
- **Welch t-test**：兩個方法各 3 個 seed 的 test PR-AUC。
- **Paired bootstrap**：對 test 晶圓做 1000 次重抽樣，每次算「兩方法各自 seed 平均 PR-AUC」的差，取 2.5 / 97.5 百分位。同時反映 test set 的抽樣誤差。

| 比較（a − b） | PR-AUC 差 | bootstrap 95% CI | Welch p | 結論 |
|---|---:|---|---:|---|
| T2 − T1 | +0.344 | [+0.329, +0.357] | —（T1 只有 1 次） | T2 顯著優於 T1 |
| **T2 − PatchCore** | **+0.006** | **[−0.003, +0.015]** | **0.62** | **無顯著差異：comparable** |
| ResNet-18 − T2 | +0.173 | [+0.164, +0.181] | 0.003 | 監督式顯著優於 T2 |
| T3 control − T2 | +0.170 | [+0.162, +0.178] | 0.003 | 只換 label 就顯著優於 T2 |
| ResNet-18 − T3 control | +0.002 | [+0.002, +0.003] | < 0.001 | 統計上顯著，但實際差距可忽略 |

**paper 的寫法**：T2 明顯優於 T1；和同 paradigm 的 SOTA PatchCore **表現相當**（差距不顯著），但成本低很多；明顯輸給使用真實 label 的監督式上限。

---

## 5. T2 vs PatchCore：打平，但強項不同

1. **整體打平，但錯在不同地方**。PatchCore 在局部、形狀不規則的 defect（Loc、Scratch、Edge-Loc、Center）較好；T2 在 Edge-Ring 明顯較好（0.975 vs 0.940）。Edge-Ring 是數量最多的 defect（佔 test defect 的 37%），所以整體 PR-AUC 拉平。
2. **為什麼 PatchCore 抓 Loc / Scratch 比較好**：PatchCore 不需要事先知道 defect 長什麼樣，任何「normal 裡沒出現過的局部 patch」都會被標出來。T2 只認得手工合成的形狀，真實 Loc / Scratch 和合成版不像的部分就會漏掉（[ablation.md](ablation.md) A2：拿掉合成 Scratch，Scratch ROC-AUC 掉 0.126）。
3. **為什麼 T2 抓 Edge-Ring 比較好**（推測，尚未驗證）：Edge-Ring 是**整圈外緣**都壞。normal 晶圓外緣本來就比較容易有壞 die（[eda.md](eda.md) 中位數 `edge_fail_rate` 0.157），所以每個外緣 patch 在 bank 裡都能找到「有點像」的 normal patch，PatchCore 的局部距離不大。T2 用 GlobalMaxPool 前的 receptive field 加上合成的 Edge-Ring，學到的是「整圈都壞」這個全域形狀。**PatchCore 是局部方法，看不到全域結構**。
4. **成本**：T2 推論快約 60 倍（0.014 vs 0.84 ms / 片），只需要存 98k 參數；PatchCore 需要 69M 參數的 backbone 加 138 MB 的 bank。兩者在 WM-811K 的產量下都夠快，但 T2 更適合放在邊緣設備上。
5. **互補 → future work**：兩者錯在不同類別，把兩個分數做 rank 平均的 ensemble 可能會超過任一者（尚未驗證，**不能放進結果**，只能寫在 future work）。

## 6. T2 vs 監督式：為什麼輸

1. **差距來自 supervision，不是 architecture**。T3 control 只換訓練資料與 label（模型和 T2 完全相同），PR-AUC 就從 0.818 → 0.988。ResNet-18（120 倍參數）只再多 0.002。**T2 要變好，該改進的是合成 defect，而不是加大模型。**
2. **合成 defect 和真實 defect 的分布不一致（sim-to-real gap）**。差距最大的四類（Edge-Loc、Loc、Scratch、Center）剛好是形狀變化最大、手工規則最難模仿的類別。Donut、Edge-Ring、Near-full 形狀規則簡單，差距就很小。
3. **監督式也學到了「normal 的邊界」**：看過 15,342 片真實 defect，誤報率壓到 0.7%；T2 和 PatchCore 都只看過 normal，誤報率約 4%。
4. **代價**：監督式需要 15,342 片**專家標註**的 defect，而且只能偵測**看過的 8 類**。T2 和 PatchCore 都不需要 defect 標註；[ablation.md](ablation.md) A2 顯示 T2 對大部分沒合成過的類別仍能偵測。新產線、新 defect 類型、標註成本高的情況下，one-class 方法才可行。

---

## 7. 給 paper 的素材

- **§2 Related Work**：PatchCore（one-class SOTA）、監督式 CNN（WM-811K 主流）、ResNet；說明我們選兩個 Tier 3 的理由（同 paradigm SOTA ＋ 監督式上限）。
- **§5 Method**：Tier 3 只需要文字說明（第 2.2、2.3 節的表）。
- **§6 Setup**：PatchCore 只用 train normal；監督式用 train 全部；128×128 放大、ImageNet 正規化屬於 method-specific preprocessing。
- **§7 Main results**：第 3 節的主表＋第 4 節的 CI 與 p-value。「T2 vs PatchCore：comparable，not better」。
- **§7 解讀**：第 5 節第 1–4 點（局部 vs 全域；成本），第 6 節第 1–2 點（supervision vs architecture；sim-to-real gap）。
- **§9 Future work**：T2 + PatchCore ensemble（第 5 節第 5 點）、更真實的合成 Scratch / Loc。

### References（BibTeX 要自行到出版商頁面核對）

- K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler. *Towards Total Recall in Industrial Anomaly Detection*. CVPR 2022.
- S. Zagoruyko, N. Komodakis. *Wide Residual Networks*. BMVC 2016.
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016.
- T. Nakazawa, D. V. Kulkarni. *Wafer Map Defect Pattern Classification and Image Retrieval Using Convolutional Neural Network*. IEEE Trans. Semiconductor Manufacturing, 31(2), 2018.
- K. Kyeong, H. Kim. *Classification of Mixed-Type Defect Patterns in Wafer Bin Maps Using Convolutional Neural Networks*. IEEE Trans. Semiconductor Manufacturing, 31(3), 2018.
- M.-J. Wu, J.-S. R. Jang, J.-L. Chen. *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets*. IEEE Trans. Semiconductor Manufacturing, 28(1), 2015.
