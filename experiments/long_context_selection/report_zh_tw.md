# DeepCASE 固定 10 筆資訊感知上下文選擇：完整實驗報告

## 0. 先說最終結論

原先要驗證的主張是：

> 從過去 100 個候選中，使用不讀取測試標籤的 selector 選出固定 10 個 context，能在不顯著增加錯誤升級的條件下，比 DeepCASE 原始 Last10 取得更高的嚴格事件召回率。

這個完整主張在 AIT-ADS 六個獨立確認情境上是**確定失敗**，不是「還沒跑完」：

- `association_abs10` 的平均嚴格事件召回率增加 `+0.051787`，情境層級 bootstrap 95% CI 為 `[0.021142, 0.077110]`，六個情境中五個為正。
- 但匯總每 1,000 個真實 non-incident 的錯誤自動升級，從 `17.6815` 增加到 `95.5035`，相對增幅 `440.13%`，遠高於事前凍結的 `5%` 安全界限。
- 因此只能說「召回率上有正向訊號」，不能說「整體比前人好」或「實務上更安全」。

失敗後開發的 `anchor5_assoc5`（固定保留最新 5 筆，再從更舊候選選 5 筆）在原設計情境有較好的安全性，但它是看到主實驗失敗後才開發，沒有新的獨立 AIT 情境可供確認，所以只是未來方法候選，不能回頭拯救原主張。

HDFS 公開資料上的數值檢定形式上通過，但有三個無法忽略的效度問題：基準送審工作量已是 `99.9137%`、安全上限被算成 `1006.18/1000`（超過物理上限），且最終分割有 `88.42%` 序列的 event pattern 與官方訓練集完全相同。排除這些重疊序列後，召回增益仍在，但安全性 non-inferiority 反而失敗。因此 HDFS 只能作為 context-selection 召回訊號的次要證據，不能驗證 SOC 告警分流主張。

## 1. 這個題目到底在研究什麼

DeepCASE 原始做法把目前 event 之前最近 10 個 events 交給 ContextBuilder。但 attention 只能在已經取進來的 10 個位置中加權；第 11 個以前的資訊若在 retrieval 階段被切掉，attention 無法救回。

因此研究問題不是「100 筆是否比 10 筆多」這種必然問題，而是：

1. Last10 是否實際排擠了可量化的有效資訊？
2. 在 downstream 模型仍只收 10 個位置時，last-100 候選池內是否存在比 Last10 更好的子集？
3. 能否用只從訓練資料學到的規則，在線上不讀目標標籤地選出這 10 筆？
4. 如果召回率上升，是否用錯誤升級、Reject 或分析師工作量換來？

主要可驗證假說為：

- H1：Last10 存在實質的標籤前驅排擠。
- H2：固定 10 個位置的 oracle 子集可優於 Last10，表示候選池有改善上限。
- H3：不使用測試標籤的 selector 可逼近該上限。
- H4：召回增益在獨立情境中成立，且錯誤升級的相對增加不超過 5%。

實驗結果是 H1 在「已標記 incident 前驅」的特定機制上不成立；H2 成立；H3 對召回率成立；H4 因安全代價失敗。

## 2. 為什麼不用 Last100 直接跑 RNN

這是研究邏輯上不需要，而不只是「電腦跑不動」。

### 2.1 直接比 Last100 會混淆什麼

Last10 RNN 與 Last100 RNN 不是同一資源預算：後者有 10 倍時間步、不同的梯度傳播距離、記憶壓力、訓練難度與推論成本。若 Last100 較好，無法知道是候選資訊好，還是只因為輸入與計算預算變大；若較差，也無法知道是候選沒資訊，還是 RNN 無法處理長序列。

工程 smoke test 中，連 hidden size 16、10 個訓練 step 的 length-100 CPU cache 都約需 40 分鐘，並逼近記憶體上限。這不是一個適合作為固定預算主實驗的 baseline。

### 2.2 正確的上限問題

本題目的 estimand 是：

`max performance(S), S ⊆ last100, |S| = 10`

因此正確作法是讓 Last10、oracle10 與 deployable selector 都交給完全相同的 DeepCASE 模型 10 個位置。Last100 只作為前處理候選池，不作為 RNN 長度。這同時降低計算成本並排除長度與模型容量的 confounding。

## 3. 資料集與正確使用方式

### 3.1 AIT-ADS：主實驗

- 來源：AIT Austrian Institute of Technology 公開的 AIT Alert Dataset，共 8 個攻擊情境、`2,655,821` 筆 alerts。
- 標籤：主結果使用 `event_incident`，incident group 使用 `event_incident_group`；不用容易擴張正例範圍的 `window_incident` 取代。
- 時間關係：只容許同 machine、目標時間之前、且不超過一天的 alerts。同秒 alerts 不能相互當 context，避免用檔案列順序偽造因果先後。
- 多重性：不預先去除重複 alerts，因為重複本身是 Last10 可能浪費名額的研究對象。
- 情境角色：`fox`、`russellmitchell` 只用於開發；`harrison`、`santos`、`shaw`、`wardbeck`、`wheeler`、`wilson` 為六個完整獨立確認單位。
- 不可做 random row split；同一情境的模式高度相關，亂切 rows 會把幾乎相同的 attack pattern 分到 train/test，導致虛高成績。

### 3.2 HDFS：公開外部次要實驗

- 使用 DeepCASE 官方資料頁指向 DeepLog 的 `hdfs_train`、`hdfs_test_normal`、`hdfs_test_abnormal`三個檔案。
- 一行視為一個 block sequence，context 絕不跨行。
- 用移除 CR/LF 後的整行 SHA-256 modulo 10 切分官方 test：0–3 為 development，4–5 為 validation，6–9 為 blind final。完全相同的行因此必然落在同一 partition，不會跨 validation/final 洩漏。
- `hdfs_train + development` 可擬合 vocabulary、association、ContextBuilder、cluster policy；validation label 只選取安全可行的 operating point；final label 在 runner 中以 `-9` 遮蔽，只有最後 analyzer 解盲。
- HDFS 沒有時間戳，且 anomaly 是 sequence-level label；所以它只能測「位置預算下的 context selection」，不能測 AIT 的一天 timeout 與 alert-level incident triage。

## 4. 實驗方法

### 4.1 比較方法

| 方法 | 作用 | 是否可部署 | 是否可用來支持最終主張 |
| --- | --- | --- | --- |
| `last10` | 取最近 10 筆，原始 baseline | 是 | 是 |
| `predecessor_oracle10` | 候選本身有 incident label 者優先，不讀目標 label | 否，線上不會先知前驅 label | 否，只估上限 |
| `class_oracle10` | 用目標真實 class 決定要選正或負 evidence | 否 | 否，只估上限 |
| `association_abs10` | 用訓練期 target×candidate×recency 關聯的絕對證據強度取 10 | 是 | 是，為原凍結 selector |
| `anchor5_assoc5` | 最新 5 筆 + 更舊候選中 association 最強 5 筆 | 是 | 本次否，因為是失敗後開發 |

### 4.2 `association_abs10` 如何從 100 選 10

對每個當前已觀察到的 target event type `t`、候選 event type `c`、以及候選的 recency band `b`，只用開發訓練情境估計：

`w(t,b,c) = logit P(feature present | incident,t) - logit P(feature present | non-incident,t)`

實作中使用 smoothing，對未同時出現在兩類的 target 回退到 global candidate/band 表，並限制絕對 log-odds 以避免稀疏計數產生無限大分數。推論時：

1. 用當前 event type、候選 event type 與位置取出 `|w(t,b,c)|`。
2. 從有效 last-100 候選中取分數最高 10 筆；同分時取較新者。
3. 選完後恢復時間順序，再交給 DeepCASE，避免把關聯分數順序偽裝成時間順序。
4. 選擇函式沒有 label 參數；回執、tie rule 與輸出長度都有自動測試。

「不使用測試標籤」不等於「完全不用任何標籤」。此 selector 是 supervised training-only selector：可用訓練標籤學 evidence，但不可使用 held-out 情境的 label 擬合或選超參數。

### 4.3 相同的 downstream 條件

- 所有決定性方法都只輸入 10 個 context 位置。
- 使用相同 hidden size 64、3 epochs、500 training steps、batch size 512 與 seeds `1729, 2718, 31415`。
- 用相同的 7 組 Interpreter operating grid，Last10 baseline 為 `eps=.1, threshold=.2`。
- 方法比較的 operating point 不看測試 label；依測試輸入可觀察的 review workload，選最靠近 baseline workload 的凍結 grid point，同分時優先低 workload 與較早登錄的 grid point。

## 5. 指標、比較與統計單位

### 5.1 主要指標

- 嚴格自動事件召回率 = `正確自動判為 Incident / 所有真實 Incident alerts`。Reject 不算抓到。
- 錯誤升級/1000 = `真實 Non-Incident 中被自動判為 Incident 的數量 / Non-Incident 數量 × 1000`。
- 分析工作量 = `(automatic Incident + Reject) / 所有 alerts`。只有自動 Non-Incident dismissal 才真正減少人工工作。
- 次要指標：strict macro-F1、incident-group recall、accepted-decision error、Reject 細分率、前處理與推論時間。

### 5.2 比較公平性

召回率必須在近似相同 review workload 比較，否則方法可以把幾乎所有東西送給人，虛假獲得高 recall。不過本實驗只有 7 個離散 grid points，所以仍有 workload mismatch。事後用相鄰 grid point 的隨機插值做 sensitivity，平均 recall gain 為 `+0.052072`、CI 仍大於 0；但這只能說 recall 不是完全來自 workload mismatch，無法反轉安全失敗。

### 5.3 正確統計單位

三個 neural seeds 只是技術重複，不是三個新的真實世界。AIT 最終推論先對 seed 平均，再以六個完整攻擊情境做 paired scenario bootstrap 20,000 次；不把幾百萬個高度相關 alerts 當成幾百萬個獨立樣本。HDFS 則以整個 block sequence 為 bootstrap 單位。

## 6. 關卡實驗結果

### Gate A：Last10 排擠了什麼

| 統計 | 結果 | 正確解讀 |
| --- | ---: | --- |
| Incident targets | 1,817,250 | 數據量大不等於獨立樣本多 |
| Last10 重複位置率 | 78.905% | 平均 10 個位置只有 2.1085 種 event type，顯示強烈冗餘 |
| Any-incident displacement@10 | 530 / 1,817,013 = 0.02917% | 幾乎不存在「benign flood 把有 label 的 attack predecessor 擠出 Last10」 |
| Same-group displacement@10 | 522 / 1,815,197 = 0.02876% | 同一 incident group 的結論一樣 |
| Any predecessor coverage | Last10 99.9578% → Last100 99.9870% | 長池在這個 label-defined coverage 上只多 0.0292 個百分點 |

所以 H1 的具體故事應被拒絕：冗餘很高，但「已標記 attack 前驅被擠掉」不是主要現象。不能因為重複率高就偷換成「一定排擠攻擊資訊」。

### Gate B：固定 10 個名額中是否有改善上限

| 診斷上限 | fox recall gain | russellmitchell recall gain | 結果 |
| --- | ---: | ---: | --- |
| predecessor-oracle10 | +0.02162 | +0.05811 | 通過預先界限 |
| class-oracle10 | +0.06588 | +0.08082 | 通過預先界限 |

這證明的是「last-100 中存在某些更好的固定 10 筆子集」，不是「oracle 可以上線」，也不是「有 label 的 predecessor 排擠是原因」。結合 Gate A，較合理的推論是：長池的改善空間可能來自沒有 attack label 但對 target/risk 有條件預測性的 event，或來自減少重複。這只是從結果得出的機制假說，不是已證明的因果機制。

### Gate C：無測試標籤 selector 的設計可行性

`association_abs10` 在兩個設計情境的平均 strict recall gain 分別為：

- fox：`+0.066609`
- russellmitchell：`+0.067080`

兩者都通過事前凍結的 advancement threshold，且 receipt 確認 retrieval 不讀取 held-out labels、輸出恰好 10 個位置。因此才進入六情境確認。

### Final confirmation：召回變好，但安全性失敗

| held-out 情境 | Last10 recall | Selector recall | recall gain | Last10 false/1000 | Selector false/1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| harrison | 0.91565 | 0.98589 | +0.07024 | 0.339 | 54.217 |
| santos | 0.94209 | 0.99020 | +0.04811 | 0.028 | 64.214 |
| shaw | 0.88841 | 0.87411 | -0.01430 | 0.082 | 83.373 |
| wardbeck | 0.77062 | 0.85566 | +0.08504 | 0.309 | 0.294 |
| wheeler | 0.90468 | 0.99179 | +0.08712 | 71.075 | 231.267 |
| wilson | 0.90899 | 0.94351 | +0.03452 | 0.038 | 53.728 |

這張表顯示不能只報 recall：在 harrison、santos、shaw、wilson，baseline 的 false escalation 幾乎為零，selector 卻上升到每千個 53–83 個；wheeler 則從 71 上升到 231。只有 wardbeck 同時提高 recall 且不增加 false escalation。

![AIT 確認實驗的召回增益與錯誤升級代價](figures/ait_confirmation_tradeoff.png)

## 7. 失敗後的可行方法探索

### 7.1 為什麼改成 `anchor5_assoc5`

`association_abs10` 可能為了稀有、強關聯的舊 event，同時把太多即時狀態捨棄；而 `|w|` 也會把強烈的 non-incident evidence 選進 context，不保證 downstream cluster 的錯誤升級不增加。

`anchor5_assoc5` 使用固定 10 個名額：

- 5 個給最新 context，保留局部狀態與 baseline 穩定性。
- 5 個給更舊高關聯 context，保留長池的增益機會。
- 選完後仍保持時間順序，不使用 test label。

### 7.2 探索結果與不可越界的解讀

純相對 `1.05 × baseline FPR` 的 v1 安全規則，因多個 validation baseline FPR 恰好為 0，導致六個 scenario×seed 中只有 2 個存在可行 operating point。這不是方法成功，而是暴露了純相對 margin 在零分母附近的退化。

失敗後登錄的 v2 hybrid margin 為：當 baseline < 1/1000 時允許絕對 +1/1000，否則使用相對 +5%。在原兩個設計情境上 6/6 seeds 有可行點，fox 與 russellmitchell 的平均 recall gain 約為 `+0.00963` 與 `+0.06814`，pooled false escalation 為 `0.13293 → 0.20509/1000`。

但這是 post-failure development，而且 margin 也是看到 v1 問題後修正。它可以產生新假說，不是獨立確認。

## 8. HDFS 外部實驗與有效性審計

### 8.1 凍結數值結果

- final 共 `269,290` sequences、`5,215,529` event rows；其中 `6,774` anomalous 與 `262,516` normal sequences。
- Last10 recall `0.832931`，anchor5_assoc5 recall `0.860252`，增益 `+0.027321`。
- sequence bootstrap 95% CI `[0.026330, 0.028322]`。
- frozen analyzer 依事前 v2 公式判為 numerical pass。

### 8.2 為什麼 numerical pass 不能當成研究主張成功

1. **安全界限不再有約束力。** Baseline false escalation 已是 `958.269/1000`，相對 +5% 得到 `1006.182/1000`，任何實際系統都不可能超過 1000，所以這個 pass 是必然的。
2. **Baseline 本身幾乎沒有分流功能。** 平均 review workload `0.999137`，等於 99.91% events 仍需送審；即使 recall 更高，也不能說是可用的 SOC automation。
3. **模式重疊過高。** final 有 `238,115/269,290 = 88.42%` sequences 與 official train 的 event pattern 完全相同。排除這些後剩 `31,175` sequences，recall gain CI 仍為正，但 safety-violation CI 為 `[+3.520, +4.588]`，表示在非重疊部分安全性其實失敗。
4. **標籤層級不相容。** HDFS 的整條 block anomaly 標籤被繼承給每個 event row，不是 SOC alert-level incident lineage。
5. **沒有 timestamp。** 無法驗證同 machine、一天內、strictly-past 候選規則。

事後 validity audit 不改寫 frozen analyzer 的數值決定；兩者同時保留，前者回答「有沒有照公式通過」，後者回答「這個公式在當前資料上還有沒有實質意義」。

## 9. 已排除或顯式揭露的漏洞與謬誤

| 漏洞/謬誤 | 本實驗的處理 | 仍剩風險 |
| --- | --- | --- |
| 「重複高，所以 attack 必然被擠掉」 | Gate A 直接算 displacement，得到只有 0.029% | 無 label 的 precursor 價值仍無法由 Gate A 觀察確定 |
| 用 oracle 成績宣稱方法成功 | Oracle 只是 Gate B 上限，明確不可部署 | Oracle 與 deployable selector 的 gap 需獨立分析 |
| Last100 較好就說 retrieval 較好 | 決定性比較都固定 10 位，避免資源與長度 confounding | 沒有提供長序列模型的性能結論 |
| 測試 label leakage | 情境分離、selector receipt、HDFS masked final labels、freeze hashes | AIT 資料曾被此 repository 早期工作使用，所以只能稱 internal pre-outcome freeze，不是外部 preregistration |
| 把 seeds 當獨立樣本 | 統計先對 seed 平均，再重抽情境/序列 | AIT 只有 6 個確認情境，CI 的外推範圍有限 |
| 忽略 Reject | Reject 不算正確抓到，並計入人工 workload | Reject 的實際組織成本沒有真實金額資料 |
| 只挑有利 threshold/子集 | 7 點 grid、tie rule、gates、seeds 與確認情境先凍結，不刪 adverse shaw | 離散 grid 使 workload 無法完全一樣 |
| 只報 recall，不報代價 | 事前要求 false escalation +5% 上限，因此判定失敗 | +5% relative 在 baseline 近 0 或近 1000 都可退化，已由 v1/HDFS 暴露 |
| 數值檢定通過就有實務意義 | HDFS 保留 formal pass，另做 non-binding limit、workload、overlap 審計 | 找不到第二個具有相容 alert lineage 的公開資料 |
| 看到失敗後改 margin 再宣稱成功 | anchor v2 明確標記 post-failure development | 需全新資料/情境做確認 |
| 與論文原數字直接比 | 主結果是同一 code/data/budget 內的 paired Last10 比較 | 不能宣稱重現了原論文的所有絕對指標 |

### 已知 protocol deviation

原 protocol 要求 Gate C 還應完成 `random10` 20 retrieval seeds、`unique_recent10`、`rarity10` 與 fixed relevance-diversity control。本次 `association_abs10` 在這些 controls 尚未完成前就進入確認，這是真實的 protocol deviation，不應隱瞞。

它會限制「改善到底是 association、去重複、稀有性還是隨機換樣本所造成」的機制歸因。但它不會把本來成功的主張錯判成失敗：主 selector 已經在獨立六情境上因預先定義的安全條件失敗，再補這些 controls 也不能反轉該結論。因此現在停止大量 control-model 訓練是依 fail-closed stop rule 節省計算，不是把未知當成成功。若論文要主張 association 是「最好 selector」，這些 controls 就必須在新的 preregistered 實驗中補齊。

## 10. 尚需決定的所有研究點

下表可直接當作新一輪實驗的 decision register。「本次」是現有實驗已凍結的選擇；「下次」是若要對 anchor 做真正獨立確認時必須在看結果前決定的事。

| 類別 | 必須決定的點 | 本次決定 | 下次建議 |
| --- | --- | --- | --- |
| 主張 | 要優化 recall、F1、cost 還是 safety-constrained recall | 同 workload 下的 strict recall，且 false escalation +5% 上限 | 使用同時具有絕對與相對上限的複合 safety margin |
| Label | event label、window label、incident group 何者是主要 | `event_incident` | 若新資料 lineage 不同，須先建 label mapping，不可臨時改 |
| 單位 | 統計獨立單位 | AIT scenario；HDFS sequence | 至少 10–20 個獨立 attack deployments 或 sites，不是更多 rows |
| 分割 | random、time、scenario、site split | scenario-disjoint；HDFS hash split | 首選 future-time + site-disjoint，並防 exact-pattern duplicates 跨 split |
| 候選實體 | same host、IP、user 或 graph neighbor | same machine | 若做 cross-host，entity edge 必須是事件當下已知資訊 |
| 時間範圍 | 1h、6h、1d、7d 或位置範圍 | strictly earlier 1 day | 用 validation 或 preregistered sensitivity，不可用 test label 挑最佳窗 |
| 候選池 | 20、50、100 或 adaptive | 主要 100，coverage 報 10/20/50/100 | 依 candidate preprocessing latency 凍結上限，不要依測試成績選 |
| 名額 | 是否恰好 10、是否允許 padding | 恰好 10 | 若改 adaptive-k，必須改研究問題並交代不同計算預算 |
| 特徵 | event ID、recency、frequency、source、entity | target/candidate event + recency band | 只允許線上當時可得特徵，列出 prohibited features |
| 關聯估計 | count、PMI、log-odds、neural scorer | smoothed clipped log-odds | 在 design data 比較 calibration 與 minimum support，不可在 confirmation 改 |
| 分數方向 | signed evidence 或 absolute evidence | absolute | 分別測試 incident-positive、nonincident-positive 與雙通道設計；這可能是 false escalation 根源 |
| Recency | 保留幾個最新 anchors | 主方法 0；事後方法 5 | 預先比 3/7、5/5、7/3，只能用 development 選一個 |
| Diversity | 是否懲罰重複 event type | 主方法未明示懲罰 | 加 MMR/submodular objective，並用 unique-recent 當 control |
| 未知 event | drop、UNK、global fallback | UNK + global fallback | 分開報告 known/unknown performance |
| Tie/padding | 同分、不足 10 筆如何處理 | 同分取較新，固定 padding | 繼續預先寫死並加測試 |
| Baselines | 需要哪些 control | Last10、oracles；其他 controls 未完成 | Last10、random10 平均、unique-recent、rarity、MMR、anchor |
| 訓練 | architecture、steps、batch、seeds | 完全相同 64/500/512/3 seeds | 當計算允許時增加 seeds，但仍不把 seeds 當獨立單位 |
| Cluster labels | 可用多少 train labels 定 policy | 全部 train labels，共同寬鬆上限 | 加 label-budget sensitivity，但所有方法同 budget |
| Operating point | 固定點、同 workload、同 FPR 或 cost | 凍結 grid 的最近 workload | 增加 grid 密度或先定 randomized interpolation，減少 mismatch |
| Safety | relative margin 在 0 與 1000 附近的行為 | AIT +5%；post-failure v2 hybrid | 使用 `min(b+δ_abs, (1+r)b, 1000-ε)` 或直接事前定義絕對 clinically/operationally meaningful margin |
| 統計 | CI 方法、bootstrap 單位、iterations | scenario/sequence paired bootstrap 20k | 預先定義同時性、multiple selectors 的 multiplicity control |
| 成功 | 最低 gain、CI、多少情境為正、safety | CI>0、4/6 正、false +5%、exact10 | 只能在新資料解盲前設定，失敗不可改 endpoint 拯救 |
| 效率 | candidate scoring time、RAM、inference latency | Last100 RNN 不可行，selector 仍輸入 10 | 報 p50/p95 per-alert latency、peak RAM、candidate cache size |
| 外推 | 需幾個資料集、標籤是否相容 | AIT 主實驗 + HDFS 限制性次要 | 優先尋找 alert-level incident lineage + timestamp + entity 的公開資料，否則誠實限制 claim |

## 11. 可以寫成什麼論文

### 11.1 不能寫的結論

- 不能寫「我們的 selector 整體優於 DeepCASE」。
- 不能寫「Last10 經常把 attack alerts 擠出去」；資料直接否定這個具體機制。
- 不能用 class-oracle、post-failure anchor 或 HDFS 無約束的 safety pass 取代 AIT 確認失敗。
- 不能把「本實作中的 Last10 paired baseline」稱為完整重現或擊敗原論文所有結果。

### 11.2 可以防守的論文主張

> 在 AIT-ADS 的固定 10 筆 DeepCASE 上下文預算下，last-100 候選池存在可提升嚴格事件召回率的子集；訓練期關聯 selector 在六個 held-out 情境的平均召回率顯著提升，但錯誤升級的巨幅增加使預先定義的 safe-superiority 假說失敗。公開 HDFS 重現召回增益，但其送審工作量、安全界限退化、重疊與標籤層級使其不能構成 SOC 安全主張的外部確認。

這是完整的 negative-result thesis，不是沒成果。它對前人的增量貢獻是：

1. 分開「labelled predecessor displacement」、「fixed-budget information upper bound」與「deployable selector」三個不同命題。
2. 在固定 10 個名額、不使用測試標籤的條件下實際驗證 selector。
3. 證明「提高 recall」可能同時破壞 false-escalation safety，因此不能用單一 F1/recall 宣稱優越。
4. 暴露 relative safety margin 在 baseline 接近 0 與接近物理上限時的兩種退化。
5. 提出「時效 anchor + 長池證據」的下一個可檢驗假說，並明確不把它當作事後救援。

### 11.3 建議論文題目

**固定預算下的 DeepCASE 長候選池上下文選擇：召回增益、安全性失敗與公開資料有效性審計**

英文可用：

**Fixed-Budget Long-Pool Context Selection for DeepCASE: Recall Gains, Safety Failure, and a Validity Audit of Public External Data**

### 11.4 如果導師一定要「比前人好」

可以誠實主張的只有受限定義：

> 在同一實作、同一資料、同一計算與 context 預算下，`association_abs10` 對 Last10 的 strict automatic incident recall 平均高 5.18 個百分點。

但這句後面必須立即寫安全失敗；不可把「recall 較高」縮寫成「整體較好」。若要得到真正的 overall better-than-prior 結論，必須用新的、未見過的、具有 timestamp 與 alert-level incident lineage 的資料，事前凍結 `anchor5_assoc5` 或其後續版本，補齊 random/unique/rarity/diversity controls，再同時通過 recall CI 與有約束力的絕對+相對 safety margin。在只有目前公開資料的限制下，無法誠實保證這個正向結論。

## 12. 可重現性與證據位置

- 主實驗事前凍結規範：[`protocol.md`](protocol.md)
- AIT retrieval audit：[`artifacts/retrieval_audit.md`](artifacts/retrieval_audit.md)
- 固定 10 上限：[`results/gate_b/gate_b_full_decision.md`](results/gate_b/gate_b_full_decision.md)
- Selector design gate：[`results/gate_c/gate_c_design_decision.md`](results/gate_c/gate_c_design_decision.md)
- 六情境確認決定：[`results/confirmation_decision.md`](results/confirmation_decision.md)
- 失敗後 anchor 開發：[`results/anchor_design_decision.md`](results/anchor_design_decision.md)
- HDFS 事前凍結規範：[`post_failure_external_protocol_v2.md`](post_failure_external_protocol_v2.md)
- HDFS frozen 數值決定：[`results/hdfs_external/final_decision.md`](results/hdfs_external/final_decision.md)
- HDFS post-unblinding 效度審計：[`results/hdfs_external/validity_audit.md`](results/hdfs_external/validity_audit.md)
- 最終 fail-closed machine decision：[`results/overall_decision.json`](results/overall_decision.json)
- 圖表原始碼：[`make_figures.py`](make_figures.py)

主要公開來源：

- [DeepCASE official repository](https://github.com/Thijsvanede/DeepCASE)
- [DeepCASE data page](https://github.com/Thijsvanede/DeepCASE/tree/main/data)
- [DeepCASE paper](https://noah-de.github.io/files/publications/vanede2022deepcase.pdf)
- [AIT Alert Dataset on Zenodo](https://zenodo.org/records/8263181)
- [AIT-ADS paper](https://www.skopik.at/ait/2024_cset.pdf)

## 13. 結論

last-100 候選池有使用價值，但不是因為 Last10 大量擠掉已標記攻擊前驅；當前的絕對關聯 selector 能提高召回，卻把過多 non-incident 升級。這不是計算失敗，而是一個已經被獨立情境確認的方法限制。在現有公開資料條件下，最負責任也最有研究價值的專題，是把這個 recall–safety trade-off、relative safety margin 的退化與外部資料不相容完整報告，而不是透過改 endpoint 或挑有利資料製造虛假的「較前人好」。
