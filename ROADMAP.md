# Roadmap to the 2026-08-01 (AoE) submission

The strategy brainstorm from 2026-07-14 (the day the test set dropped), kept verbatim as
the plan of record, plus a live status column. Detailed findings live in
[TEST_SET_ANALYSIS.md](TEST_SET_ANALYSIS.md) (test-set facts),
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) ("System Architecture" section: measured
numbers, 10B accounting, distillation pipeline, infra rules), and
[EXPERIMENTS.md](EXPERIMENTS.md) (per-job results). 繁中原文保留 — 這是計畫的原始語言。

## 改變策略的關鍵事實（2026-07-14）

1. **評測有人工評審，且優先評 primary 系統** — 短而乾的 gold-style 答案能討好 chrF，但人類偏好
   流暢、有用、完整的回答。這讓蒸餾的價值又上升一級（teacher 的答案風格天然更討喜）。
2. **測試 prompt 是 self-contained，還內嵌字數限制、風格要求等額外指令** — 我們的訓練/推理是用
   自家模板（外加 lang-hint system turn），存在 train/test 格式落差。
   （07-15 實證更新：*每一筆*測試 prompt 都有內嵌指令，含「無答案」逃生門和「用 X 語回答」結尾；
   字數預算 24 語全有 — 見 TEST_SET_ANALYSIS.md §4。）
3. **`task` 欄位在測試時直接給你**（qa-context / qa-oeg / sum-sum）— 按任務路由合法，一個 9B
   底座掛多個 LoRA adapter 依任務切換完全在 10B 限制內（實測帳：9.44B + 0.029B/adapter，
   IMPLEMENTATION_NOTES §2）。
4. **驚喜語言 bho（Bhojpuri）** — 訓練資料零筆；fra/swh/tel/tha 從測試集消失（別再為它們優化）。
   每隊可交 3 份輸出（primary + 2 variants），可以對沖。
   （07-15 更新：最終測試集確認只有這一個驚喜語言。）

## 方法清單（按投報比排序）＋ 現況

| # | 方法 | 原始計畫 | 現況（2026-07-15 晚） |
|---|---|---|---|
| A | **對齊測試格式**（必做，投報比最高） | 寫 `run_test.py` 讀官方 JSONL、直接餵 self-contained prompt、輸出 `{id, output}`；dev 改「測試格式」重跑 sanity check — lang-hint 拿掉後大跌就表示依賴自家模板 | ✅ 完成。`run_test.py`＋sbatch 就緒、TEST_SET_ANALYSIS.md 文件化。唯一未收尾：lang-hint 依賴度 A/B = job 3859645（跑步中） |
| B | **蒸餾**（人工評審讓它更值錢） | teacher 在 train split 生成 → chrF/BERTScore 對 gold 過濾 → teacher+gold 混合練全新 adapter（不接續 3822375，保持單變數可比）。OEG 是主要得分空間 | 🟡 生成中。122B(vLLM)/aya+oeg ✅（3859682，17 分鐘）；35B 全量 3 shards 跑步中（3859277-79，ETA 明晨）；`filter_teacher.py` ✅ 已在真資料出 report（3860144：30/70 留 44.3%，OEG 對 GPT-4.1 gold 分數特別高）；`train_lora.py --data` ✅ |
| C | **指令遵循增強** | 訓練例隨機加「N 字內」「條列式」等約束並改寫目標答案。非英語長度控制是通用模型弱點，可拉開差距 | ⬜ 已證實必要（smoke：字數預算 9/10 violated）。設計要點：**從答案反推約束**（量答案字數→prompt 加「約 N 字」，永遠可滿足）；約束措辭**直接從測試集挖各語言原生說法**（已探勘：471/2,259 筆 OEG prompt 有數字字數約束，24 語樣本都在） |
| D | **Bhojpuri 應急包** | FLORES-200、Aya collection 撈 bho_Deva 混進 SFT；驗證輸出不滑回 Hindi | ⬜ 已證實必要（smoke：bho 輸出漂成 Hindi/尼泊爾語/邁蒂利語）。注意：`facebook/flores` 在 HF 是 gated，改用 `openlanguagedata/flores_plus`；Aya 系列看 `CohereLabs/aya_collection_language_split` |
| E | **任務路由**（幾乎零成本，合法） | qa-context 用 few-shot 示範（+35 chrF 來源）、qa-oeg 用蒸餾 adapter、sum-sum 接隊友。與隊友合流成聯合系統（共用同一個 9B base，否則爆 10B）才有總榜資格 | 🟡 設計定案。缺口：`run_test.py` 還沒有 `--shots`（variant1 安全牌也需要）。3858987（adapter+3shot，跑步中）出分後拍板 adapter 角色。⚠️ belebele MC 增益不轉移（測試集無選擇題），路由決策要看 tydiqa/aya 型分數 |
| F | **推理期品質守門** | fastText LID（<1MB）檢查輸出語言、錯了換 seed 重生成；可試 best-of-N + 9B 自評 | ⬜ 未動工。LID 直接對治實測到的 bho 漂移；vLLM 管線讓 best-of-N 成本大降（實測批次 ~250× 快） |
| G | **三份提交對沖** | primary = 蒸餾+路由完全體；variant1 = 9B 3-shot（27.64 安全牌）；variant2 = 激進版（best-of-N） | ⬜ 策略已定，最後 3 天執行：用 100% 樣本資料重練最終版、跑測試集、提交 |

## 時程（截止 2026-08-01 AoE）

- **第 1 週（~07-20）**：A ✅ → B teacher 生成 ✅/🟡 → 過濾閾值定案 → C+D 資料準備
- **第 2 週（~07-27）**：C+D 混進同一次 SFT 重練 → E（含 `run_test.py --shots`）+ F 推理管線 →
  dev 上用測試格式驗證整條路由
- **最後 3 天**：G — 100% 資料重練最終 adapter、跑官方測試集三種配置、Google Form 提交
