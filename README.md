# DeepCASE 資安事件脈絡分析

本專題以 DeepCASE 為基礎，檢查其他主機的事件、較早的事件，以及少量攻擊標記能否改善資安告警判斷。DeepCASE 會結合歷史事件，將相似脈絡分群，協助分析人員集中處理相關告警。

實驗保留 DeepCASE 的主要架構，分別比較三個延伸方向。結果顯示長時間脈絡有召回與誤判的取捨，跨主機及訓練標記不足的方法則未展現穩定改善。

## 研究結果

召回率是實際攻擊被找回的比例。模型無法自動判斷、需交由人工審查的事件稱為 Reject。

| 方向 | 方法 | 最終結果 |
| --- | --- | --- |
| 跨主機關聯 | 分別編碼本機與其他主機的歷史，再調整兩者的資訊比重 | 兩個設計情境、單一種子的對照中，自動辨識出的攻擊減少，更多攻擊轉為 Reject，未擴大實驗。 |
| 長時間脈絡 | 從前 100 筆事件選出 10 筆，維持相同輸入數量 | 嚴格自動召回提高 5.18 個百分點，但每千筆正常事件的錯誤自動升級由 17.68 增至 95.50，超過事前設定的限制。 |
| 訓練標記不足 | 加入攻擊辨識的輔助訓練目標與平衡抽樣 | 送審召回提高 0.59 個百分點，95% 信賴區間為 −0.39 至 +1.64 個百分點，包含零，尚不能認定有提升。 |

長時間實驗只計算自動判對的攻擊，標記不足實驗也計入 Reject 後送審的攻擊，兩者的增益不能直接比較。標記不足指訓練時可見的攻擊標記少，AIT-ADS 本身的攻擊事件並不稀少。

完整的比較條件與限制見[最終報告摘要](docs/final_report_summary.md)。

## 專案導覽

| 位置 | 內容 |
| --- | --- |
| `main.py`、`main_research.py`、`src/`、`configs/` | 初步探索的基準流程與三項延伸方法 |
| [長時間脈絡實驗](experiments/long_context_selection/README.md) | 固定 10 筆輸入的事件選擇、AIT-ADS 確認實驗與 HDFS 外部檢查 |
| [訓練標記不足實驗](experiments/imbalance/README.md) | 輔助訓練、標記比例與送審量的配對比較 |
| `tests/` | 共用流程與初步延伸方法的測試 |
| [早期探索報告](docs/controlled_variants_report.md) | 保留初步方法與結果，與最終確認結果分開解讀 |

## 環境與基本檢查

最終報告使用 Python 3.12.14，套件版本列於 `requirements.txt`。下載專案並建立環境：

```powershell
git clone https://github.com/NSYSUTim/deepcase-controlled-extensions.git
cd deepcase-controlled-extensions
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

這項測試檢查共用流程。完整確認實驗另需資料集、前處理與實驗設定，且目前實驗程式仍引用整理前的目錄名稱，尚不能直接由此版本重跑，詳見各實驗說明。

初步探索的指令範例如下。執行前須在設定檔中填入資料路徑，完成基準流程後，再將研究設定的 `baseline_run_id` 改為新產生的執行識別值：

```powershell
.\.venv\Scripts\python.exe main.py deepcase-run --config configs/deepcase_ait_ads_cpu.json
.\.venv\Scripts\python.exe main_research.py run --config configs/research/cross_host_report.json
```

`configs/research/` 另含階層式脈絡與標記平衡設定。這些是初步探索入口，不會產生最終報告的六情境、三種子確認結果。

## 資料與適用範圍

AIT-ADS 請由[官方資料頁](https://doi.org/10.5281/zenodo.8263181)取得，HDFS 的資料來源見 [DeepCASE 官方專案](https://github.com/Thijsvanede/DeepCASE)。本專案未附資料集、模型權重及完整執行產物。

AIT-ADS 來自模擬環境，HDFS 則只有序列級標記，結果不能直接代表企業資安營運中心的部署成效。送審比例也只是人工工作量的代理值，未實測分析人員工時。

DeepCASE 的版本與授權說明見[第三方專案說明](THIRD_PARTY_NOTICES.md)。
