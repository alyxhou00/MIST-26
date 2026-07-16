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
   字數預算**每個語言都有、但只在 21% 的 qa-oeg 列上**（各語言精確 21/100，因為 qa-oeg 是
   100 個 prompt 翻 24 語的平行語料；471/2,359 — 2026-07-15 對官方檔案實測。先前寫「24 語全有」
   本身沒錯，但易被誤讀成「每列都有」）— 見 TEST_SET_ANALYSIS.md §4。）
3. **`task` 欄位在測試時直接給你**（qa-context / qa-oeg / sum-sum）— 按任務路由合法，一個 9B
   底座掛多個 LoRA adapter 依任務切換完全在 10B 限制內（實測帳：9.44B + 0.029B/adapter，
   IMPLEMENTATION_NOTES §2）。
4. **別再看 dev 的整體分數做決定 —— 但也別過度修剪 dev**（2026-07-16 修訂）。
   dev 是我們的「模擬考」，但題型分佈跟真正的考試不同，所以**總分是假的**。要**分 sub-task 看**，
   而且每個 sub-task 用對的指標。

   | source | dev 列數 | 代表什麼 |
   |---|---|---|
   | `facebook/belebele` | 1,123 | ❌ **選擇題。測試集一題都沒有。38% 的 dev，零預測力。** |
   | `copenlu/answerable_tydiqa` | 615 | ✅ **qa-context** —— 且要用 **EM/F1** 看，不是 chrF |
   | `FBK-MT/MCIF` | 165 | ✅ qa-context |
   | `wmt25-mist-oeg-gpt-4.1` | 97 | ✅ **qa-oeg 的長篇端**（gold 中位 175 詞）≈ 87% 的 qa-oeg prompt |
   | `CohereLabs/aya_dataset` | 978 | ✅ **qa-oeg 的短答端**（gold 中位 24 詞）≈ 13% 的 qa-oeg prompt |

   實務規則：
   - **qa-context 看 tydiqa + MCIF 的 EM/F1**；**qa-oeg 把 oeg 和 aya 當兩個獨立欄位看**
     （它們量的是同一任務的兩端，**絕不可平均**）；**belebele 不計分**。
   - ⚠️ **dev 的權重是反的**：aya 有 978 列卻只對應 ~13% 的 qa-oeg；oeg 只有 97 列卻對應 ~87%。
   - **「overall chrF」永遠不要拿來比較系統**（例：3859645 整體只掉 1.67 看似輕微，但那 1.67
     幾乎全是 belebele 崩 20 分造成的，而 belebele 根本不重要；反過來也會發生）。

   ⚠️ **這一條本身就是踩過的坑，而且踩了兩次**：
   - 第一次：把 README 的 sub-task 分類表當成 proxy 對應表用。
   - 第二次（更糟）：**先前這裡寫「aya 不代表任何測試任務、71% 是雜訊」——那是錯的，已撤回。**
     理由是「qa-oeg 要 120-180 詞、aya 只有 24 詞」，但**只有 ~20% 的 qa-oeg 帶字數預算**；
     整個任務是光譜，100 個 unique prompt 裡約 13% 是短答／清單／機智問答（「說出一個名字裡
     沒有母音字母的國家」、「首都五大景點」）——**正是 aya 的形態**。錯在用 20% 的特徵定義
     100% 的任務。
   - **教訓：qa-oeg 只有 100 個 unique prompt**（平行語料翻 24 語）——要下關於它的結論，
     就把 100 個全部讀完，別抽樣外推。細節見 TEST_SET_ANALYSIS §5b。
5. **驚喜語言 bho（Bhojpuri）** — 訓練資料零筆；fra/swh/tel/tha 從測試集消失（別再為它們優化）。
   每隊可交 3 份輸出（primary + 2 variants），可以對沖。
   （07-15 更新：最終測試集確認只有這一個驚喜語言。）

## 方法清單（按投報比排序）＋ 現況

| # | 方法 | 原始計畫 | 現況（2026-07-15 晚） |
|---|---|---|---|
| A | **對齊測試格式**（必做，投報比最高） | 寫 `run_test.py` 讀官方 JSONL、直接餵 self-contained prompt、輸出 `{id, output}`；dev 改「測試格式」重跑 sanity check — lang-hint 拿掉後大跌就表示依賴自家模板 | ✅ **完成並收尾**。`run_test.py`＋sbatch 就緒、TEST_SET_ANALYSIS.md 文件化。**lang-hint 依賴度 A/B 已出（3859645）：拿掉幾乎不痛 → 可直接用無 hint 的測試格式，不必重練。** 25.97 vs 27.64（−1.67），但這個平均是誤導的：損失幾乎全在 belebele（52.70→32.42），而測試集無選擇題、不轉移；測試集真正有的 source 只掉 1 分內（tydiqa −4.55、MCIF −0.81、aya −0.34、OEG **+0.09**）。附帶觀察：hint 撐的是 MC 的「格式」，與 few-shot 先前被歸功的效果冗餘 —— 拿掉一個就崩。⚠️ 別試圖算「排除 belebele 後的整體 chrF」：整體是 corpus-level 聚合、非 per-source 加權平均（n 加權得 38.61，與實際 27.64 對不上）。⚠️ **新發現（commit df12b0a）**：官方檔案是**雙重跳脫**的——全部 8,640 筆 qa-context prompt 帶的是字面上的 `\` `n` 兩個字元（不是換行），位置正好在「文章／問題／指令」的段落邊界。目前 verbatim 餵法讓模型在 **79% 的 qa 列**讀到字面 `\n\n`。TEST_SET_ANALYSIS §2 原本寫反了（用 `'\n' in prompt` 測，那測的是真換行）。`run_test.py --unescape` 可還原，**預設關**（會動到官方輸入，且 dev 沒有對應樣本可 A/B）→ 建議先 qualitative smoke，並列為 variant 提交的候選軸 |
| B | **蒸餾**（人工評審讓它更值錢） | teacher 在 train split 生成 → chrF/BERTScore 對 gold 過濾 → teacher+gold 混合練全新 adapter（不接續 3822375，保持單變數可比）。OEG 是主要得分空間 | 🟡 生成中。⚠️ **資料形狀已查證（TEST_SET_ANALYSIS §5c）**：122B 那 4,126 列 = aya 3,763 + **oeg 僅 363**，也就是 **91% 是 aya** → **不是**現成的 qa-oeg 訓練集，不能靠它單獨練 qa-oeg adapter；35B 獨有的 7,789 列裡 4,577 列是 belebele（格式不轉移），真正有用的獨有部分是 tydiqa 2,497 + MCIF 715。**qa-oeg 是全盤最薄的一環：2,359 測試列，只有 97 dev proxy + 363 訓練列。** 122B(vLLM)/aya+oeg ✅（3859682，17 分鐘）；35B 全量 3 shards 跑步中（3859277-79，實測 ~215-265 列/hr → 各 15-18h，24h 限內；**先前寫的「ETA 明晨」是低估**）；`filter_teacher.py` ✅ 已在真資料出 report（3860144：30/70 留 44.3%，OEG 對 GPT-4.1 gold 分數特別高）；`train_lora.py --data` ✅ |
| C | **指令遵循增強** | 訓練例隨機加「N 字內」「條列式」等約束並改寫目標答案。非英語長度控制是通用模型弱點，可拉開差距 | ✅ 程式就緒（`constraint_bank.py` + `augment_constraints.py`，commit db1addc）。**改良**：約束措辭不用手寫翻譯，直接從測試集**提取**——每個語言的 qa-context 尾巴全 360 列一字不差，可原樣取用；`--selftest` 對 tests.jsonl 驗證每條主張。**兩個易錯點**：jpn/zho 的預算單位是「字」（字元）不是詞，且有換算（150 words → zho 250字 → jpn 300字）；數字字形是**逐語言**而非逐文字系統（ben ১০০、mar १००、ckb ١٠٠、pes ۱۰۰，但 arb/hin/bho 都用 ASCII 100）。副產品：拿到各語言**確切的拒答字串**（"not answerable"/"无法回答"/…），可直接對治 smoke 的假拒答。待跑：等 filter_teacher 產出後套用 |
| D | **Bhojpuri 應急包** | FLORES-200、Aya collection 撈 bho_Deva 混進 SFT；驗證輸出不滑回 Hindi | ✅ 資料已產出：**8,009 列** `data/sft-bho.jsonl`（在叢集，commit ab5aad3 + 修正）。⚠️ **原計畫的兩個資料源都是死路**（已對 HF API 查證）：`openlanguagedata/flores_plus` **也是 gated**（正是為了避開 gated 才選它）；`CohereLabs/aya_collection_language_split` 132 個語言 config **完全沒有 bho**。改用：`HuggingFaceFW/fineweb-2` config `bho_Deva`（18,666 篇原生網頁文，唯一有量的來源）→ 續寫任務 6,000 列；`CohereLabs/xP3x` config `bho_Deva`（**未 gated** 地拿到 FLORES 的 bho）→ hin→bho 翻譯 2,009 列。注意 xP3x 的 1.22M 列其實只有 **2,009 句** unique（200+ 來源語 × 3 template 展開），列數不等於資料量。品質閘：`bho_lid.py`（功能詞判別 bho/hin/mai/npi）。⚠️ **更正**：本文件先前寫「fineweb 的 bho 子集混了 167 篇 Hindi/80 篇 Nepali/16 篇 Maithili」——**那是錯的**，經抽樣查證，那些多半是分類器自己把真正的博傑普爾語誤判（原因見下）。實測 fineweb bho_Deva **約 96% 是真 bho，只有約 1% 判為 hin/npi/mai，子集基本乾淨**；這個閘的價值是在邊緣棄權，不是攔截大量污染 |
| E | **任務路由**（幾乎零成本，合法） | qa-context 用 few-shot 示範（+35 chrF 來源）、qa-oeg 用蒸餾 adapter、sum-sum 接隊友。與隊友合流成聯合系統（共用同一個 9B base，否則爆 10B）才有總榜資格 | 🟡 **adapter 角色已拍板：原始設計不變，但兩者絕不疊加。** 3858987（adapter+3shot）= **21.64**，比純 3-shot（27.64）和 adapter 0-shot（26.56）**都差**，且除 OEG 外每個 source 都是三者最差（belebele 52.70/85.82→26.66、MCIF 34.61/49.26→20.98、tydiqa 38.94/19.53→**14.46**、aya 24.19/21.95→19.94；只有 OEG 25.55/29.06→29.62 撐住，n=97）。原假設「demo 救回 tydiqa 同時保住 adapter 增益」完全失敗 —— demo 反把 tydiqa 壓到比 adapter-only 更低。解釋：adapter 在 **0-shot 格式**微調（`train_lora.py` 無 demo），few-shot prompt 對它是 OOD。**推論（與 A 列合起來）：蒸餾 adapter 應直接用測試格式（無 hint、無自家模板）訓練並 0-shot 評測** —— 訓練/推理格式必須一致。**路由表（依據 = 唯一忠實的 proxy，見 TEST_SET_ANALYSIS §5b）：`qa-oeg`（2,359 列）→ adapter（OEG chrF 29.06 vs 3-shot 25.55、BERTScore 72.89 vs 69.38，一致）；`sum-sum`（1,776 列）→ 隊友；`qa-context`（8,640 列）→ ✅ **定案：adapter，0-shot**（2026-07-16，jobs 3864996-99 per-source 重算）。**那場「chrF vs EM 誰對」的爭議根本不存在 —— 是我們把兩個不同的任務混在同一個欄位裡算。** 拆開後，**唯一忠實的 proxy（MCIF，跨語言，n=165）上 adapter 四個指標全勝**：EM 21.82 vs 0.61（36 倍）、F1 57.92 vs 28.15、chrF 49.26 vs 34.61、BERTScore 86.41 vs 74.38 —— **沒有任何指標有異議**。先前寫的「EM 16.92 vs 6.54、F1 打平、只有 chrF 偏好 3-shot」全是**混池**數字，79% 來自單語的 tydiqa（≈ 測試任務的 4%）；數字對得起來：16.92 = (615×15.61 + 165×21.82)/780。連帶地，「adapter 在 tydiqa 崩潰是 chrF 假象」這個辯護也不必了 —— tydiqa 崩不崩潰**根本不影響路由**，它不是這個任務的 proxy。**這個決定既不需要官方指標、也不需要 sqrt(EM×chrF) 對沖**（那條規則在 tydiqa 上是 17.79 vs 17.46，本來就分不出勝負），只需要用對 proxy。詳見 EXPERIMENTS.md 的拆分表與待辦 #6。 缺口：`run_test.py` 還沒有 `--shots`（variant1 安全牌也需要）—— **qa-context 那半邊現在跑不了**。⚠️ **belebele 與 aya 的分數都不能拿來做路由決策**：測試集無選擇題（belebele 不轉移），且 aya 的 gold 中位僅 24 詞、測試 qa-oeg 要 120-180 詞（OEG gold 中位 175 詞）—— aya 不是 qa-oeg 的 proxy，2026-07-15 對官方檔案實測。dev 2,978 列裡只有 ~877 列（29%）有預測力 |
| F | **推理期品質守門** | fastText LID（<1MB）檢查輸出語言、錯了換 seed 重生成；可試 best-of-N + 9B 自評 | 🟡 起步了：`bho_lid.py`（D 的副產品）可當 bho 守門員，對 sib200 實測 3 句以上 recall 94% / precision 99%（單句 73%，別用）。⚠️ **但這組數字只代表 FLORES 新聞文體，不能外推**——第一版就是在 sib200 拿 91%/100%，卻把真正的博傑普爾**網頁**文自信地判成 Nepali（marker 表缺了日常的 -ela/-ala 動詞；且 margin 規則在對手密度為 0 時恆真，已加絕對下限 `MIN_DENSITY` 修掉）。教訓：**換語料就要重新抽樣查證，別信舊的評測數字**。它也只認 bho/hin/mai/npi 四語——全 24 語守門要用 **GlotLID**（`cis-lmu/glotlid`，有 bho_Deva）；fastText `lid.176` 會把 bho/mai/mag 混成一個 `bh`，別用。best-of-N 未動工 |
| G | **三份提交對沖** | primary = 蒸餾+路由完全體；variant1 = 9B 3-shot（27.64 安全牌）；variant2 = 激進版（best-of-N） | ⬜ 策略已定，最後 3 天執行：用 100% 樣本資料重練最終版、跑測試集、提交 |

## 時程（截止 2026-08-01 AoE）

- **第 1 週（~07-20）**：A ✅ → B teacher 生成 ✅/🟡 → 過濾閾值定案 → C ✅ + D ✅ 資料準備
- **第 2 週（~07-27）**：C+D 混進同一次 SFT 重練 → E（含 `run_test.py --shots`）+ F 推理管線 →
  dev 上用測試格式驗證整條路由
- **最後 3 天**：G — 100% 資料重練最終 adapter、跑官方測試集三種配置、Google Form 提交

## 待決策 / 待辦（2026-07-15 晚，主線 session 交出）

1. **`--unescape` 要不要進 primary**（見 A 列）。無 dev proxy，只能靠 qualitative smoke 判斷。
2. ~~**寄信給主辦方**（schmidtova@ufal.mff.cuni.cz）~~ → **關閉（使用者決定，2026-07-16）：不寄信。**
   官方指標仍然未知 —— 這是「在未知下做決定」，**不是**把未知解掉了。
   （原本要問的另兩件事 —— 雙重跳脫、8 筆 `{country}`/`{language}` placeholder —— 也隨之不問；
   100 筆空 prompt 主辦方已於 07-16 自行修掉。）
   **選型規則：`COMBINED` = mean(chrF, BERTScore, ROUGE-L)，所有 sub-task 一致**（jobs 3865022-25）。
   **當天即被取代的前一版是 `sqrt(EM × chrF)`**，理由值得記住：幾何平均實際上把決定權交給 **EM**
   （EM 的相對跨度 3.9× vs chrF 的 2.3×），而 EM 偏偏**只在 tydiqa 上有解析度** —— 也就是那個
   *不像*測試集的 proxy。在真正能決定事情的 proxy 上（MCIF、以及整個 qa-oeg），gold 太長、EM 被壓在
   地板，規則不是半盲就是在乘一個 ≈0 的因子。**一條「只在量錯的地方才有效」的規則不是規則。**
   ⚠️ 新規則也**不是中立的**：原始值的算術平均會依變異量隱性加權 —— BERTScore 跨度僅 1.24×
   （chrF 2.35×），它決定水位、幾乎不決定名次；而 chrF 與 ROUGE-L 都在量表面重疊，等於「2 票表面、
   1 票語意」。這是個把拇指壓在表面重疊上的可辯護折衷，細節寫在 `evaluate.py:combined()`。
   **換規則沒有改變任何已做的路由決定。**

6. 🔴 **`qa-context` 的 dev proxy 有 79% 是錯的任務（2026-07-16 實測，EXPERIMENTS.md 有完整表格）**
   —— 使用者留在 EXPERIMENTS.md 的那則「test set 跟 dev 很不一樣，去讀 qa-context」註記已查證，**是對的**：
   - **qa-context 只有 100 個 unique item**（跟 qa-oeg 一樣是平行語料，不是 8,640 個相異問題）。
     id 格式 `qa-context_{n}_{問題語}_{文章語}` —— **問題語在前**，讀反會把 `fra` 誤認成答題語言。
   - 每個 item 用全部 24 種問題語問；變動的是**文章**被翻成幾種語言 → 展開極不平均：
     **5 個 item 各 24×25 = 600 列，合計佔 35%**；另外 75 個 item 各只有 24 列。
   - **96% 的列是跨語言的**。**文章語有 25 種、問題語只有 24 種** → `fra` 是**只當文章**的語言。
     第 5 條的「fra/swh/tel/tha 從測試集消失」對**答題語言**成立（`question_lang` 欄位各 0 列），
     對**文章語言不成立**。
   - **tydiqa（615 列、79%）是單語**（阿拉伯文文章+問題+答案）≈ 只代表測試任務的 4%；
     **MCIF（165 列、21%）是唯一跨語言、唯一忠實的 proxy** —— 而 MCIF 上 adapter 大勝（chrF 49.26
     vs 3-shot 34.61）。整場 chrF vs EM 之爭是在錯的 source 上打的。
   - **待辦**：`evaluate.py` 目前把 EM/F1 在 `TASK_PROXY`（tydiqa+MCIF）層級混算 → 決策表裡每個 EM
     都是 79% 的 tydiqa。**修法是重新計分（4 份 prediction CSV 都還在），不用重跑推理。**
   - **這就是使用者說「We need an whole new train/dev set」的理由**：唯一忠實的跨語言 QA 來源只有
     MCIF，n=165，而且是 TED 逐字稿、答案是句子長度（不是 `evaluate.py` header 假設的 2 詞抽取
     —— 那個假設來自 tydiqa，我們**沒有**測試集 gold）。
3. **C 尚未套用**：等 35B shards（3859277-79）合併過濾出 `data/sft-distilled.jsonl` 後，
   跑 `augment_constraints.py` 產 `-c.jsonl`，再跟 `data/sft-bho.jsonl` 串起來練。
4. **D 的 bho 資料還沒被模型看過** — 8,009 列已就緒但尚未進任何一次 SFT；
   `bho_lid.py` 可在 eval 後直接量「輸出到底是不是 bho」。
5. **拒答字串**（`constraint_bank.context_tail(lang).refusal_phrase`）目前只是被抽出來，
   還沒被任何訓練/推理路徑使用 — 對治 smoke 的假拒答（4/10 arb）是現成的一步。
