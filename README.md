# DeepCASE Controlled Extensions

本 repository 整理我以 DeepCASE 為基礎進行的受控延伸實驗，聚焦於三個問題：跨主機事件關聯、長時間攻擊脈絡，以及資料與標籤不平衡。程式碼保留 DeepCASE 的 GRU、注意力機制與後續分群流程，每次只修改一項因素，目的是確認結果是否真的來自欲處理的問題，而不是全面更換模型架構後產生的差異。

這份程式碼是研究過程的可檢查紀錄，不宣稱提出的方法全面優於原始 DeepCASE。較嚴格的多次初始化、情境分離與誤判檢查顯示，部分改善只出現在特定指標或特定設定，尚不足以支持穩定提升的結論。

## 研究問題與實作對應

| 研究問題 | 採用方法 | 程式碼位置 | 選擇原因 |
| --- | --- | --- | --- |
| 跨主機事件關聯 | 雙 GRU 分別編碼本機與其他主機事件，再以 gated fusion 調整兩類資訊的比重 | `src/my_capstone_research/cross_host/` | 直接將所有主機事件混入同一序列，容易讓大量無關事件蓋過本機脈絡；分流編碼可保留來源差異，gated fusion 則讓模型依事件決定是否採用其他主機資訊。 |
| 長時間攻擊脈絡 | 分別編碼近期事件與較早的代表事件；另測試從較長時間範圍選出固定數量的高關聯事件 | `src/my_capstone_research/hierarchical_context/`、`experiments/long_context_selection/` | 直接增加序列長度會同時增加運算量與無關事件，無法分辨改善來自「時間範圍」或「更多輸入」。固定輸入數量能在相同輸入預算下比較事件選取方式。 |
| 資料與標籤不平衡 | weighted BCE、Focal Loss、類別與事件群組平衡抽樣 | `src/my_capstone_research/rebalanced/`、`experiments/imbalance/` | 加權損失直接提高少數攻擊事件的訓練權重；Focal Loss 進一步降低大量容易樣本的影響；群組平衡抽樣避免樣本多的單一事件群組主導訓練。測試資料仍維持原始分布。 |

### 為什麼沒有直接改用 Transformer 或圖神經網路？

這兩類方法並非不能使用，而是不適合當時「一次只檢驗一個問題」的受控比較。改用 Transformer 會同時更換主要編碼架構、參數量及訓練條件；圖神經網路還需要額外定義主機、使用者、程序或事件之間的節點與連線方式，引入新的資料需求與建模假設。保留 DeepCASE 主體，才能較明確地判斷跨主機資訊、長時間事件選取或不平衡處理本身是否有效。

## 程式碼結構

```text
.
├── main.py                         # 原始 DeepCASE 基準流程入口
├── main_research.py                # 三項延伸方法的統一入口
├── compare_results.py              # 基準與延伸方法的結果比較
├── configs/                        # AIT-ADS 與研究方法範例設定
├── src/
│   ├── my_capstone/                # 資料、執行與評估共用程式
│   └── my_capstone_research/
│       ├── cross_host/             # 雙 GRU 與 gated fusion
│       ├── hierarchical_context/   # 短期脈絡與長期記憶
│       └── rebalanced/             # weighted BCE、Focal Loss 等
├── experiments/
│   ├── imbalance/                  # 不平衡學習的受控比較
│   └── long_context_selection/     # 固定輸入數量的長時間事件選取
└── tests/                          # 共用流程與三項方法的單元測試
```

## 資料集與可重現範圍

本 repository 不重新散布資料集。AIT-ADS 可由其[官方 Zenodo 頁面](https://doi.org/10.5281/zenodo.8263181)取得；HDFS 的 DeepCASE 前處理資料與原始執行方式可參考[官方 DeepCASE repository](https://github.com/Thijsvanede/DeepCASE)。下載後需依各實驗目錄的 README 或設定檔調整資料路徑。

公開資料仍無法完整替代實際企業 SOC 日誌：其蒐集期間、攻擊設計、標註對象與判定標準，以及主機與服務配置均不相同。因此，本研究結果只用來比較在現有資料條件下的相對變化，不外推為真實企業環境的部署成效。

為避免把本機路徑與舊實驗產物誤當成可重現資料，下列內容未上傳：

- 高雄市政府或其他未獲授權的場域資料；
- 公開資料集的副本、模型權重、快取及完整輸出結果；
- 本機虛擬環境與暫存檔；
- 與這三項 DeepCASE 研究問題無關的後續專題。

`configs/research/*.json` 保留實驗設定，其中的 `baseline_run_id` 是原研究執行時的識別值。重新執行延伸方法前，請先完成自己的基準流程，並把該欄位改成新產生的 run ID。

## 環境與檢查

建議使用 Python 3.11 建立獨立環境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

先執行不需要資料集與模型權重的測試：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pytest experiments\imbalance\tests -q
.\.venv\Scripts\python.exe -m pytest experiments\long_context_selection\tests -q
```

執行基準流程與研究方法的指令格式如下：

```powershell
python main.py deepcase-run --config configs/deepcase_ait_ads_cpu.json
python main_research.py run --config configs/research/cross_host_report.json
python main_research.py run --config configs/research/hierarchical_context_report.json
python main_research.py run --config configs/research/rebalanced_report.json
```

長時間事件選取與不平衡學習的完整控制條件、資料切分方式及停止條件，分別記錄於兩個 `experiments/` 子目錄。這些腳本需要先下載資料，部分完整實驗也需要較長的 CPU／GPU 執行時間。

## 結果解讀

- 跨主機模型只有有限的辨識改善，未證明在不同設定下皆穩定有效。
- 階層式脈絡在部分指標上改善，但也使事件分散到更多群組。
- 延伸事件搜尋時間後，可找出更多攻擊相關事件，但正常事件被誤判為攻擊的情況也增加。
- 不平衡方法必須同時檢查少數類別辨識、誤判及人工審查負擔，不能只依單一 F1 或召回率下結論。

這些負面與混合結果仍予以保留，因為它們直接反映方法的限制，也說明後續研究設計為何需要更完整的資料與評估條件。

## 第三方專案

本專案以 DeepCASE 的公開實作為依賴，沒有把 DeepCASE 原始 repository 或資料集副本納入本 repository。版本與授權說明見 `THIRD_PARTY_NOTICES.md`。
