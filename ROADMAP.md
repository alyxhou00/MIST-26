# Roadmap to the 2026-08-01 (AoE) submission

The strategy brainstorm from 2026-07-14 (the day the test set dropped), kept verbatim as
the plan of record, plus a live status column. Detailed findings live in
[TEST_SET_ANALYSIS.md](TEST_SET_ANALYSIS.md) (test-set facts),
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) ("System Architecture" section: measured
numbers, 10B accounting, distillation pipeline, infra rules),
[EXPERIMENTS_NEW.md](EXPERIMENTS_NEW.md) (per-job results on the v2 item-split dev — the
live log) and [EXPERIMENTS.md](EXPERIMENTS.md) (the closed pre-v2 log; its dev leaks,
DATA_AUDIT.md §2). 繁中原文保留 — 這是計畫的原始語言。

## 改變策略的關鍵事實（2026-07-14）

1. **評測有人工評審，且優先評 primary 系統** — 短而乾的 gold-style 答案能討好 chrF，但人類偏好
   流暢、有用、完整的回答。這讓蒸餾的價值又上升一級（teacher 的答案風格天然更討喜）。
2. **測試 prompt 是 self-contained，還內嵌字數限制、風格要求等額外指令** — 我們的訓練/推理是用
   自家模板（外加 lang-hint system turn），存在 train/test 格式落差。
   （07-15 實證更新：*每一筆*測試 prompt 都有內嵌指令，含「無答案」逃生門和「用 X 語回答」結尾；
   字數預算**每個語言都有、但只在 20% 的 qa-oeg prompt 上**（各語言精確 **20/100**，因為 qa-oeg 是
   100 個 prompt 翻 24 語的平行語料；parser 找到 465/2,359 列 ±5%。本段先前寫 21/100、471 列 ——
   07-16 已在 TEST_SET_ANALYSIS §4 更正（第 21 條是俳句的 5-7-5 格式約束、不是字數預算），
   這裡當時漏改）— 見 TEST_SET_ANALYSIS.md §4。）
3. **`task` 欄位在測試時直接給你**（qa-context / qa-oeg / sum-sum）— 按任務路由合法，一個 9B
   底座掛多個 LoRA adapter 依任務切換完全在 10B 限制內（實測帳：9.44B + 0.029B/adapter，
   IMPLEMENTATION_NOTES §2）。
4. **別再看 dev 的整體分數做決定 —— 但也別過度修剪 dev**（2026-07-16 修訂）。
   dev 是我們的「模擬考」，但題型分佈跟真正的考試不同，所以**總分是假的**。要**分 sub-task 看**，
   而且每個 sub-task 用對的指標。

   > ⚠️ **本條的表格與列數是 v1（洩漏的舊 dev）時代的，07-17 起被 v2 取代**，保留當歷史。
   > v2 之後：dev = `data/dev_v2.jsonl`（2,949 qa 列），**belebele-v2（1,270）與 tydiqa-v2（485）
   > 已重製成測試版型（跨語言、含拒答列），是合法的 qa-context proxy**，不再「零預測力」；
   > MCIF（160）仍是長 context 端的 proxy。qa-oeg 聚合 = **0.87·OEG(90) + 0.13·aya(944)**，
   > 絕不混池。規則與現行數字見 EXPERIMENTS_NEW.md；「總分是假的」與「分 sub-task 看」不變。

   | source | dev 列數 | 代表什麼 |
   |---|---|---|
   | `facebook/belebele` | 1,123 | ❌ **選擇題。測試集一題都沒有。38% 的 dev，零預測力。** |
   | `copenlu/answerable_tydiqa` | 615 | ❌ **（2026-07-16 撤回）單語 —— 不是 qa-context 的 proxy。見第 6 條。** |
   | `FBK-MT/MCIF` | 165 | ✅ **qa-context 唯一忠實的 proxy**（跨語言，對上測試集的 96%） |
   | `wmt25-mist-oeg-gpt-4.1` | 97 | ✅ **qa-oeg 的長篇端**（gold 中位 175 詞）≈ 87% 的 qa-oeg prompt |
   | `CohereLabs/aya_dataset` | 978 | ✅ **qa-oeg 的短答端**（gold 中位 24 詞）≈ 13% 的 qa-oeg prompt |

   實務規則（2026-07-16 修訂）：
   - **qa-context 只看 MCIF**；**qa-oeg 把 oeg 和 aya 當兩個獨立欄位看**（它們量的是同一任務的
     兩端，**絕不可平均**）；**belebele 不計分**。
   - **指標一律用 `COMBINED` = mean(chrF, BERTScore, ROUGE-L)**（見待辦 #2）。~~qa-context 用
     EM/F1~~ 已撤回：EM 只在 tydiqa 上有解析度，而 tydiqa 不是這個任務的 proxy。
   - ⚠️ **dev 的權重是反的**：aya 有 978 列卻只對應 ~13% 的 qa-oeg；oeg 只有 97 列卻對應 ~87%。
     **qa-context 更糟**：615 列的 tydiqa 零預測力，能用的只有 165 列的 MCIF。
   - **能用的 dev 只剩 ~1,240 / 2,978 列（42%）**，其中 qa-context 那一半只有 165 列。
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
5. **驚喜語言 bho（Bhojpuri）** — sample 訓練資料零筆；fra/swh/tel/tha 從測試集消失（別再為它們優化）。
   每隊可交 3 份輸出（primary + 2 variants），可以對沖。
   （07-15 更新：最終測試集確認只有這一個驚喜語言。07-18 更新：D 的 8,009 列 bho pack
   已隨 3869129 進 SFT——「零筆」自此不成立。）

## 方法清單（按投報比排序）＋ 現況

| # | 方法 | 原始計畫 | 現況（2026-07-15 晚） |
|---|---|---|---|
| A | **對齊測試格式**（必做，投報比最高） | 寫 `run_test.py` 讀官方 JSONL、直接餵 self-contained prompt、輸出 `{id, output}`；dev 改「測試格式」重跑 sanity check — lang-hint 拿掉後大跌就表示依賴自家模板 | ✅ **完成並收尾**。`run_test.py`＋sbatch 就緒、TEST_SET_ANALYSIS.md 文件化。**lang-hint 依賴度 A/B 已出（3859645）：拿掉幾乎不痛 → 可直接用無 hint 的測試格式，不必重練。** ✅ **2026-07-17 在 adapter 上也證實了（3866054）**：gold adapter 用 `--no-lang-hint` 重測，三個 routing 欄都在開 hint 版（3857589）的 0.6 COMBINED 內（MCIF 63.18 vs 62.55、OEG 46.60 vs 46.44、aya 34.73 vs 34.62）——先前只在 base 3-shot 測過的疑慮解除，整張路由表的 gold-LoRA 數字都可直接對照 `run_test.py` 的無 hint 推理。25.97 vs 27.64（−1.67），但這個平均是誤導的：損失幾乎全在 belebele（52.70→32.42），而測試集無選擇題、不轉移；測試集真正有的 source 只掉 1 分內（tydiqa −4.55、MCIF −0.81、aya −0.34、OEG **+0.09**）。附帶觀察：hint 撐的是 MC 的「格式」，與 few-shot 先前被歸功的效果冗餘 —— 拿掉一個就崩。⚠️ 別試圖算「排除 belebele 後的整體 chrF」：整體是 corpus-level 聚合、非 per-source 加權平均（n 加權得 38.61，與實際 27.64 對不上）。⚠️ **新發現（commit df12b0a）**：官方檔案是**雙重跳脫**的——全部 8,640 筆 qa-context prompt 帶的是字面上的 `\` `n` 兩個字元（不是換行），位置正好在「文章／問題／指令」的段落邊界。目前 verbatim 餵法讓模型在 **79% 的 qa 列**讀到字面 `\n\n`。TEST_SET_ANALYSIS §2 原本寫反了（用 `'\n' in prompt` 測，那測的是真換行）。`run_test.py --unescape` 可還原，**預設關**（會動到官方輸入，且 dev 沒有對應樣本可 A/B）→ 建議先 qualitative smoke，並列為 variant 提交的候選軸 |
| B | **蒸餾**（人工評審讓它更值錢） | teacher 在 train split 生成 → chrF/BERTScore 對 gold 過濾 → teacher+gold 混合練全新 adapter（不接續 3822375，保持單變數可比）。OEG 是主要得分空間 | ❌ **敗給純 gold SFT，收案（2026-07-17）**。全流程跑完：teacher 生成 → 過濾 → SFT（3864945）→ 評測（3865036）。在唯一能決定路由的兩欄都輸：`qa-context` MCIF **−11.92** COMBINED、`qa-oeg` 長篇 OEG **−12.37**，只在不轉移到測試集的 aya 短答 +1.28。confound（蒸餾同時拿掉了 lang-hint）已由 **3866054** 排除——gold adapter 用 `--no-lang-hint` 重測只動 ≤0.6 COMBINED，所以 ~12 分的損失是 **teacher data 本身**，不是 hint。原因見 §5.4 的「價值十字」：蒸餾最有 headroom 的地方（aya）不轉移，最需要的地方（OEG）幾乎沒 headroom（GPT-4.1 gold 已 94% 通過過濾）。**路由不變：gold-LoRA 3822375 保留兩個 qa 任務。** 人工評審是唯一自動指標關不掉的活口，但不足以單獨支撐部署一個在忠實 proxy 上輸 12 分的 adapter。以下為過程記錄：⚠️ **資料形狀已查證（TEST_SET_ANALYSIS §5c）**：122B 那 4,126 列 = aya 3,763 + **oeg 僅 363**，也就是 **91% 是 aya** → **不是**現成的 qa-oeg 訓練集，不能靠它單獨練 qa-oeg adapter；35B 獨有的 7,789 列裡 4,577 列是 belebele（格式不轉移），真正有用的獨有部分是 tydiqa 2,497 + MCIF 715。**qa-oeg 是全盤最薄的一環：2,359 測試列，只有 97 dev proxy + 363 訓練列。** 122B(vLLM)/aya+oeg ✅（3859682，17 分鐘）；35B 全量 3 shards ✅ 已完成 07-16（3859277-79，11,915 列；先前這裡寫「跑步中」）；`filter_teacher.py` ✅ 已在真資料出 report（3860144：30/70 留 44.3%，OEG 對 GPT-4.1 gold 分數特別高）；`train_lora.py --data` ✅ |
| C | **指令遵循增強** | 訓練例隨機加「N 字內」「條列式」等約束並改寫目標答案。非英語長度控制是通用模型弱點，可拉開差距 | ✅ 程式就緒（`constraint_bank.py` + `augment_constraints.py`，commit db1addc）。**改良**：約束措辭不用手寫翻譯，直接從測試集**提取**——每個語言的 qa-context 尾巴全 360 列一字不差，可原樣取用；`--selftest` 對 tests.jsonl 驗證每條主張。**兩個易錯點**：jpn/zho 的預算單位是「字」（字元）不是詞，且有換算（150 words → zho 250字 → jpn 300字）；數字字形是**逐語言**而非逐文字系統（ben ১০০、mar १००、ckb ١٠٠、pes ۱۰۰，但 arb/hin/bho 都用 ASCII 100）。副產品：拿到各語言**確切的拒答字串**（"not answerable"/"无法回答"/…），可直接對治 smoke 的假拒答。→ ✅ **已套用在 v2 substrate 上（2026-07-18）**：v2 的 qa-context 本身就內建完整 test tail＋拒答列（拒答已證實訓得進去：3869088 量到 hit 97.5%/88.5%、假拒答 0.1%/4.9%），所以 C 縮減為「qa-oeg 加字數預算」——`augment_constraints.py` 改成 task-aware，`data/train_v2-cd.jsonl` 內 840 列帶預算（≈20% 的 qa-oeg，貼近 test 的 20/100）。**C+D 合體 SFT = 3869129 ✅ / eval = 3869130 ✅（07-19 完成）**。→ **C 的效果驗證中，初步是大幅正向**：dev **量不到** C（`dev_v2` 的 qa-oeg prompt 帶預算的有 **0 列**，預算只存在訓練檔），所以拿官方測試集的 465 列預算 prompt 重跑（job **3875151** C+D vs **3875152** plain 對照，`--task qa-oeg`，用 `scripts/verify_outputs.py` 計分）。**完整 2,359 列 / 24 語（07-21）：compliance 65.8% vs 44.9%（+20.9pp）**，而且 C 消掉的是**答太短**（under 51.2%→27.1%），不是灌水。**且 C 是有條件的**：非預算的 1,894 列兩邊長度幾乎一樣（中位 73 vs 81），只有預算列被拉動（119→149）→ dev 的 −1.42 不該記在 C 頭上。**結論：C 留**，缺的是 C-only adapter（見「接下來」#1） |
| D | **Bhojpuri 應急包** | FLORES-200、Aya collection 撈 bho_Deva 混進 SFT；驗證輸出不滑回 Hindi | ✅ 資料已產出：**8,009 列** `data/sft-bho.jsonl`（在叢集，commit ab5aad3 + 修正）。⚠️ **原計畫的兩個資料源都是死路**（已對 HF API 查證）：`openlanguagedata/flores_plus` **也是 gated**（正是為了避開 gated 才選它）；`CohereLabs/aya_collection_language_split` 132 個語言 config **完全沒有 bho**。改用：`HuggingFaceFW/fineweb-2` config `bho_Deva`（18,666 篇原生網頁文，唯一有量的來源）→ 續寫任務 6,000 列；`CohereLabs/xP3x` config `bho_Deva`（**未 gated** 地拿到 FLORES 的 bho）→ hin→bho 翻譯 2,009 列。注意 xP3x 的 1.22M 列其實只有 **2,009 句** unique（200+ 來源語 × 3 template 展開），列數不等於資料量。品質閘：`bho_lid.py`（功能詞判別 bho/hin/mai/npi）。⚠️ **更正**：本文件先前寫「fineweb 的 bho 子集混了 167 篇 Hindi/80 篇 Nepali/16 篇 Maithili」——**那是錯的**，經抽樣查證，那些多半是分類器自己把真正的博傑普爾語誤判（原因見下）。實測 fineweb bho_Deva **約 96% 是真 bho，只有約 1% 判為 hin/npi/mai，子集基本乾淨**；這個閘的價值是在邊緣棄權，不是攔截大量污染。→ ✅ **8,009 列首次進 SFT（2026-07-18）**：`augment_constraints.py --append-bho` 正規化成 v2 schema（task=`bho-pack`、question_lang=`bho`）併入 `train_v2-cd.jsonl`，隨 job 3869129 訓練 ✅。→ **bho 輸出驗證：初步成功但別過度樂觀（2026-07-20）**。`dev_v2` 的 bho 是 **0 列**（pack 只進訓練），所以只能在測試集的 100 列 bho qa-oeg 上驗（jobs 3875151/3875152，這 100 列已全部生成完）：**bho 40% vs 對照組 12%，Hindi 漂移 67%→36%**（`bho_lid`，棄權 23%/21% 另計；2,359 列跑完後數字未變，這 100 列本來就早已完成）。⚠️ **人工讀 12 例後的修正**：(a) 被判 `hin` 的很多其實是**混語**（有 होला/एगो/काहें 但用 है/करता 的印地語繫詞，marker 規則吃繫詞很重）——該說「36% 偏印地語混合」而非「36% 純印地語」；(b) **兩個系統都出現退化重複**（同一句重複 8–10 次），這不是 runaway（沒有假對話輪，stop-fix 有效）而是 OOD 語言的一般退化，而且**會灌水 bho_lid**（重複的 bho 句 density 0.38，對照組有一列 `bho` 標籤就是重複迴圈）→ +28pp 方向可信、幅度要打折；(c) 部分 bho 答案只有 5–10 詞（過短殘留，比 aya 更嚴重） |
| E | **任務路由**（幾乎零成本，合法） | qa-context 用 few-shot 示範（+35 chrF 來源）、qa-oeg 用蒸餾 adapter、sum-sum 接隊友。與隊友合流成聯合系統（共用同一個 9B base，否則爆 10B）才有總榜資格 | 🟡 **adapter 角色已拍板：原始設計不變，但兩者絕不疊加。** 3858987（adapter+3shot）= **21.64**，比純 3-shot（27.64）和 adapter 0-shot（26.56）**都差**，且除 OEG 外每個 source 都是三者最差（belebele 52.70/85.82→26.66、MCIF 34.61/49.26→20.98、tydiqa 38.94/19.53→**14.46**、aya 24.19/21.95→19.94；只有 OEG 25.55/29.06→29.62 撐住，n=97）。原假設「demo 救回 tydiqa 同時保住 adapter 增益」完全失敗 —— demo 反把 tydiqa 壓到比 adapter-only 更低。解釋：adapter 在 **0-shot 格式**微調（`train_lora.py` 無 demo），few-shot prompt 對它是 OOD。**推論（與 A 列合起來）：蒸餾 adapter 應直接用測試格式（無 hint、無自家模板）訓練並 0-shot 評測** —— 訓練/推理格式必須一致。**路由表（依據 = 唯一忠實的 proxy，見 TEST_SET_ANALYSIS §5b）：`qa-oeg`（2,359 列）→ adapter（OEG chrF 29.06 vs 3-shot 25.55、BERTScore 72.89 vs 69.38，一致）；`sum-sum`（1,776 列）→ 隊友；`qa-context`（8,640 列）→ ✅ **定案：adapter，0-shot**（2026-07-16，jobs 3865022-25 per-source 重算）。**那場「chrF vs EM 誰對」的爭議根本不存在 —— 是我們把兩個不同的任務混在同一個欄位裡算。** 拆開後，**唯一忠實的 proxy（MCIF，跨語言，n=165）上 adapter 四個指標全勝**：EM 21.82 vs 0.61（36 倍）、F1 57.92 vs 28.15、chrF 49.26 vs 34.61、BERTScore 86.41 vs 74.38 —— **沒有任何指標有異議**。先前寫的「EM 16.92 vs 6.54、F1 打平、只有 chrF 偏好 3-shot」全是**混池**數字，79% 來自單語的 tydiqa（≈ 測試任務的 4%）；數字對得起來：16.92 = (615×15.61 + 165×21.82)/780。連帶地，「adapter 在 tydiqa 崩潰是 chrF 假象」這個辯護也不必了 —— tydiqa 崩不崩潰**根本不影響路由**，它不是這個任務的 proxy。**這個決定既不需要官方指標、也不需要 sqrt(EM×chrF) 對沖**（那條規則在 tydiqa 上是 17.79 vs 17.46，本來就分不出勝負），只需要用對 proxy。詳見 EXPERIMENTS.md 的拆分表與待辦 #6。~~缺口：`run_test.py` 還沒有 `--shots` —— qa-context 那半邊現在跑不了~~ → **兩個缺口都沒了**：`--shots` 已於 2026-07-16 實作（commit bae02b9），而且 primary 根本不再需要它 —— **qa-context 和 qa-oeg 現在都走 `--lora` 0-shot，同一條路徑**。`--shots` 現在的用途是 variant1（純 demo 的安全牌）。⚠️ **belebele 與 aya 的分數都不能拿來做路由決策**：測試集無選擇題（belebele 不轉移），且 aya 的 gold 中位僅 24 詞、測試 qa-oeg 要 120-180 詞（OEG gold 中位 175 詞）—— aya 不是 qa-oeg 的 proxy，2026-07-15 對官方檔案實測。dev 2,978 列裡只有 **1,240 列（42%）**有預測力：MCIF 165 + OEG 97 + aya 978（先前寫的「877 列 / 29%」是舊帳 —— 它把 aya 當雜訊排除、又把單語的 tydiqa 615 列算進去，兩處都已撤回）。→ ✅ **07-18 在 v2 乾淨 dev 上重驗：路由結論存活**——plain-v2 adapter（3869088，truncated）**每一欄都贏** base 0-shot 和 3-shot（belebele-v2 44.33 / tydiqa-v2 72.04 / MCIF 50.95 / qa-oeg agg 39.99 vs 3-shot 的 37.25/57.37/45.67/34.47），且 v2 之後 qa-context 有三個合法 proxy（本列前段「belebele/aya 不能用」的量測顧慮是 v1 時代的）。**條件：adapter 推理必須帶 runaway 防護**（見「接下來」#0）。primary 候選 = C+D 完全體（3869129/3869130 出分後定案） |
| F | **推理期品質守門** | fastText LID（<1MB）檢查輸出語言、錯了換 seed 重生成；可試 best-of-N + 9B 自評 | 🟡 起步了：`bho_lid.py`（D 的副產品）可當 bho 守門員，對 sib200 實測 3 句以上 recall 94% / precision 99%（單句 73%，別用）。⚠️ **但這組數字只代表 FLORES 新聞文體，不能外推**——第一版就是在 sib200 拿 91%/100%，卻把真正的博傑普爾**網頁**文自信地判成 Nepali（marker 表缺了日常的 -ela/-ala 動詞；且 margin 規則在對手密度為 0 時恆真，已加絕對下限 `MIN_DENSITY` 修掉）。教訓：**換語料就要重新抽樣查證，別信舊的評測數字**。它也只認 bho/hin/mai/npi 四語——全 24 語守門要用 **GlotLID**（`cis-lmu/glotlid`，有 bho_Deva）；fastText `lid.176` 會把 bho/mai/mag 混成一個 `bh`，別用。best-of-N 未動工。→ **07-18 補充：F 的第一個守門已上線**——runaway 防護（`stop_strings` + `truncate_runaway`，commit d5a65b3）進了 benchmark.py 和 run_test.py；GlotLID 語言守門仍待做 |
| G | **三份提交對沖** | primary = 蒸餾+路由完全體；variant1 = 9B 3-shot（27.64 安全牌）；variant2 = 激進版（best-of-N） | ⬜ 策略微調後照走（蒸餾出局 → primary = **gold-v2 adapter 路由**；3875151/3875152 已跑完 → 現在等 **C-only** adapter 的 dev 來三選一，見「接下來」#1）；具體排程見「接下來做什麼」#5。variant1 = 3-shot 安全牌在 v2 dev 的新基準是 37.25/57.37/45.67/34.47 |

## 時程（截止 2026-08-01 AoE）

- **第 1 週（~07-20）**：A ✅ → B teacher 生成 ✅（結論：蒸餾出局）→ C ✅ + D ✅ 資料準備
  →（計畫外但必要）**dev/train 洩漏審計 + v2 重建 ✅ + 乾淨基準線 ✅ + runaway 修正 ✅**
- **第 2 週（~07-27）**：C+D 混進同一次 SFT 重練 ✅（3869129/3869130，07-19 完成）→
  🟡 **primary 還沒定案，但收斂了**（C+D 在 dev 輸 1.42 qa-oeg，但 dev 對 C 和 D 都是盲的；
  驗證 jobs 3875151/3875152 ✅ 完整跑完 → C ✅ 有效且無副作用、D ✅ 有效，代價來自 bho pack
  稀釋 → 缺 **C-only** 這一顆 adapter 才能三選一，見「接下來」#1）→ E ✅（路由在 v2 重驗存活）+ F 🟡（runaway 守門 ✅、
  GlotLID 待做）
- **最後 3 天**：G — 100%（train_v2+dev_v2）重練最終 adapter、跑官方測試集三種配置、
  Google Form 提交（細節見「接下來做什麼」#5）

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
   - **tydiqa（615 列、79%）是單語**（文章/問題/答案同語言，共 11 種語言 — 先前這裡寫「阿拉伯文」
     是以偏概全，見 DATA_AUDIT.md §1）≈ 只代表測試任務的 4%；
     **MCIF（165 列、21%）是唯一跨語言、唯一忠實的 proxy** —— 而 MCIF 上 adapter 大勝（chrF 49.26
     vs 3-shot 34.61）。整場 chrF vs EM 之爭是在錯的 source 上打的。
   - **待辦**：`evaluate.py` 目前把 EM/F1 在 `TASK_PROXY`（tydiqa+MCIF）層級混算 → 決策表裡每個 EM
     都是 79% 的 tydiqa。**修法是重新計分（4 份 prediction CSV 都還在），不用重跑推理。**
   - **這就是使用者說「We need an whole new train/dev set」的理由**：唯一忠實的跨語言 QA 來源只有
     MCIF，n=165，而且是 TED 逐字稿、答案是句子長度（不是 `evaluate.py` header 假設的 2 詞抽取
     —— 那個假設來自 tydiqa，我們**沒有**測試集 gold）。
   - ✅ **重建完成（2026-07-17）**：`data/{train,dev}_v2.jsonl`（item-split、測試版型、含拒答訊號），
     `scripts/build_dataset.py` 可重現；決策記錄在 DATA_AUDIT.md §7，新實驗一律記
     EXPERIMENTS_NEW.md（舊 dev 數字全部不可比）。v2 之後 qa-context 的 dev proxy 不再只剩 MCIF ——
     belebele-v2 / tydiqa-v2 都是測試版型（跨語言、含 unanswerable），但仍分欄看。
3. ~~C 的 substrate 待決~~ → ✅ **全部落地（2026-07-18）**：substrate = v2（使用者定案 07-17），
   `data/train_v2-cd.jsonl` 已建（C 的 qa-oeg 字數預算 840 列 + D 的 bho 8,009 列，增強列繼承
   `item_group`），**C+D 合體 SFT = 3869129 / eval = 3869130**。見 EXPERIMENTS_NEW.md。
4. ~~D 的 bho 資料還沒被模型看過~~ → ✅ 已進 3869129；`bho_lid.py` 驗證見 D 列（初步 40% vs 12%）。
5. ~~拒答字串沒被使用~~ → ✅ v2 資料內建拒答列且**證實訓得進去**（3869088：hit 97.5%/88.5%、
   假拒答 0.1%/4.9%，三個系統中最佳）。⚠️ 教訓：**拒答指標只能在 truncated 預測上量**——
   raw CSV 會被 runaway 垃圾拖到低估 25 倍（3.9% vs 92.7%）。

## 接下來做什麼（2026-07-21 更新，按優先序）

0. 🔴 **一切 adapter 推理都必須帶 runaway 防護**（已內建於 benchmark.py / run_test.py，
   commit d5a65b3——generate `stop_strings` + `truncate_runaway()`；smoke 3869113 = 0/60 污染）。
   自己另寫推理路徑的話記得掛上 `prompt_template.RUNAWAY_STOP_STRINGS`。
1. 🟡 **primary 配方：驗證跑完了，答案是「C 留、D 留、但 C+D 綁在同一顆 adapter 不該當
   primary」——缺的那一顆是 C-only，去把它練出來（2026-07-21）**。
   - **完整結果（3875151 C+D vs 3875152 plain，各 2,359 列 / 24 語全到齊，13h25）**：
     compliance **65.8% vs 44.9%（+20.9pp）**、bho **40% vs 12%**、Hindi 漂移 **67%→36%**。
     前 517 列的初步讀數站得住：語言排序偏誤把兩邊的絕對值各拉高約 9pp，**差距沒變**。
   - **C 是「有條件」的，不是全域變短**（這次才量到，是關鍵）：擔心的是 840 列預算樣本會讓
     adapter 到處都變簡短，那會在**沒有**預算的 1,894 列上賠掉 recall。實測：非預算列兩邊幾乎
     一樣（中位長度 73 vs 81），預算列 C+D 從 119 → **149**——長度變化是**跟著指令走的**。
     → **dev 的 −1.42 不是 C 造成的，是 bho pack 佔 41% 稀釋 qa 頭造成的。**
   - **D 也該留**：錯語言的輸出 chrF 近乎零分，bho 從 12%→40% 對 bho 那一欄的拉抬，跟 OEG 整體
     −1.67 是**同一個數量級**（bho 是 24 語之一）——這正是 dev 的 −1.42 定不了案的原因。
   - **→ 下一步（擋著 07-27 定案的唯一一件事）**：練 **C-only**——
     `augment_constraints.py`**不加** `--append-bho`，出 `data/train_v2-c.jsonl`（11,674 列），
     照 3867139 的配方 SFT + dev eval（估 ~6.5h + ~5h，07-27 前綽綽有餘）。
     → ✅ **已建檔並開跑（07-21）**：`data/train_v2-c.jsonl` = 18,901 列（訓練用 **11,674** 列，
     與 3867139 **逐列相同**，只有 840 列多了預算句、0 列 bho）→ **單變數可比**。
     SFT job **3876434** RUNNING（`--data data/train_v2-c.jsonl --no-lang-hint`）；
     跑完接 `sbatch slurm/lora_eval.sbatch <adapter>`，再用 `run_test.py --task qa-oeg`
     + `verify_outputs.py` 補 compliance。
     **判準**：C-only 若把 dev 追回 plain-v2 的水準、compliance 又守住 ~65%，
     它就是 primary，C+D 降為對沖 bho 的 variant；若 C-only 的 dev 也掉，
     那 −1.42 就不是 bho pack 的錯，回頭選 plain-v2 當 primary、C+D 當 variant。
2. ~~讀 aya 的風格偏移~~ → ✅ **讀完，主體是 chrF 假象，不處理（2026-07-18，906 列對讀 + 12 例
   人工讀，EXPERIMENTS_NEW.md）**：gold 中位 26 詞；base 答 105 詞＋**89% 帶 markdown**（灌水
   餵飽 chrF 的 recall）；adapter 答 11 詞、gold 式簡答（markdown 2.3%）——BERTScore/ROUGE 升
   才是對的訊號。殘餘的真問題（不動，記錄在案）：(a) 少數補全/開放題**過短**（arb 補全 7 詞 vs
   gold 60 詞）；(b) 知識性列舉會**編造**（新浪潮片單張冠李戴）——這與人評「偏好完整流暢」的
   張力一起留給 G 階段的 variant 決策，不影響路由。
3. **`--unescape` qualitative smoke**（待辦 #1，一直沒動）：literal `\n\n` 還原與否，
   variant 提交的現成軸；一個 smoke job 對比輸出品質即可。
4. **F：GlotLID 全語言輸出守門**（`cis-lmu/glotlid`，有 bho_Deva）＋ 錯語言重生成；
   best-of-N 仍未動工，時間不夠就只上語言守門。
5. **G：最後衝刺排程**（截止 08-01 AoE，倒數兩週）：
   - ~07-27 前定案 primary 配方（C-only vs C+D vs plain-v2 三選一，看 #1）；
   - 用 **100% 資料（train_v2 + dev_v2 合併）** 重練最終 adapter（split 只是量測工具，
     交卷前要把 dev 的訊號拿回來）；
   - 跑官方測試集三種配置：primary = 最終 adapter 路由（qa-context + qa-oeg 都走 adapter、
     0-shot、無 hint）＋隊友的 sum；variant1 = 9B 3-shot 安全牌；variant2 = 激進軸
     （--unescape 或 best-of-N，看 #3/#4 誰成熟）；
   - Google Form 提交。
   - 🔴 **排程警告（07-20 實測）**：`run_test.py` 在測試集上是 **~22 s/列**（jobs 3875151/3875152），
     不是 sbatch header 寫的 4.5–8.3 s/列。**全部 10,999 列 qa ≈ 67 小時，是 24h 上限的近 3 倍**
     → 最終三份提交**一定要 `--shard i/n` 切**（n=4 每片約 17h），而且要提早開跑。
     細節記在 IMPLEMENTATION_NOTES §6。
6. **兩個輸出瑕疵，已在完整 2,359 列上量化**（兩個 adapter 都有，所以都不是 C 或 D 造成的；
   EXPERIMENTS_NEW.md 有細節）：
   (a) 🟢 **字面 `<br>` markup**：**731 列（31.0%）vs plain 657 列（27.9%）**——**每三筆
   qa-oeg 預測就有一筆帶 HTML 標記**。來自 qa 語料（網頁來源的 aya/oeg），不是 bho pack。
   **這是整張 roadmap 上最便宜的一分**：交卷前一行後處理清掉即可，該在 G 階段的提交腳本裡做掉。
   (b) 🟡 **退化重複**（同一句重複 ≥4 次）：**52 列（2.2%）vs plain 57 列（2.4%）**——量比
   想像中小，而且不是 C+D 特有。不是 runaway（沒有假對話輪），現有的 stop-fix 攔不到；
   要治得用 `repetition_penalty` 或 n-gram 阻擋，但 2% 的量級不值得為它冒改變解碼參數的風險，
   除非 F 的語言守門順手一起做。
7. **小工具債**：`evaluate.py` 自動輸出 qa-oeg 加權聚合（0.87/0.13）＋ truncated 拒答指標，
   免得每次手算；EM/F1 的 TASK_PROXY 混池問題（待辦 #6）在 v2 時代已無關緊要，不修。
8. **傳話給隊友**：test sum 輸入 p50 2,773 詞超過 `train_lora.py` 的 2048-token 截斷；
   另外 v2 的 sum-sum 列已按 item-group 切好（CrossSum 的 context_lang 是 null、
   跨語言平行文章抓不到——要用的話得回 upstream 拿配對 metadata）。
