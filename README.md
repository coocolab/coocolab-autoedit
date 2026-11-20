# Coocolab AutoEdit｜錄完丟進來，AI 幫你剪

口播影片的 AI 自動剪輯工具。你照常錄，卡住、重講、停頓都沒關係，把影片丟進資料夾，跟 Claude Code 或 Codex 說一句「幫我剪這支」就好。

> 酷可實驗室，酷的實驗你也可以。

## 它會做什麼

| 步驟 | 內容 |
|---|---|
| 聽打 | 把人聲切成一段一段分開聽打，講到一半重講也抓得到 |
| 剪掉清單 | AI 標出重講、卡住、語助詞、喊卡，**先列清單給你確認**，你說 OK 才剪 |
| 剪片 | 剪點逐格對齊，嘴型不會跑掉。太長的停頓會縮短，跳剪時交替放大一點點，看起來像換鏡頭 |
| 字幕 | 字跟著聲音一個一個跳，重點詞變橘色，中英文之間自動空格，另外輸出 .srt |
| 字卡 | AI 挑重點，講到那句話的時候字卡就彈出來 |
| 動態特效 | 跟人疊在一起的特效：字從頭後滑出、光沿著輪廓跑一圈、便條紙、數字卡、清單卡、會自己打字的指令列、照片牆、影片卡、3D 軌道 |
| 語速 | 量你每秒講幾個字，建議要不要整支加速。音調不變 |
| 音量 | 調到 -14 LUFS（YouTube／IG 的標準音量），人聲放在左右聲道正中間 |
| 封面 | 挑一格畫面，加上大標題 |

全部在你自己的電腦上跑，不用月費，影片不會上傳到任何地方。

## 安裝（最簡單的方法）

到 [酷可實驗室官網的免費資源區](https://coocolab.com) 複製「一鍵剪片」的 prompt，整段貼進 Claude Code 或 Codex。它會：
1. 檢查你的電腦：Mac 還是 Windows、夠不夠力、缺哪些工具。
2. 只下載你缺的東西，每裝一樣都會先問你。
3. 建好 `AI剪片` 資料夾，裡面有「原片」和「成片」兩個資料夾。

## 手動安裝

需要：Python 3.10～3.13、ffmpeg、Git。

```bash
git clone https://github.com/coocolab/coocolab-autoedit.git
cd coocolab-autoedit
python3 install/setup.py check      # 看缺什麼
python3 install/setup.py install    # 建環境、裝套件、建 AI剪片 資料夾（預設在桌面）
```

Windows 把 `python3` 換成 `py`。

## 怎麼用

1. 把影片放進 `AI剪片/原片/`。
2. 在 `AI剪片` 資料夾打開 Claude Code（或 Codex）。
3. 說「幫我剪這支」。

這個資料夾裡的 `CLAUDE.md`／`AGENTS.md` 會告訴 AI 整個流程和規矩，所以不用再貼任何 prompt。

也可以自己下指令：

```bash
AE="<.venv 裡的 python> <這個 repo>/ae.py"
$AE init "原片/我的影片.mp4"
$AE transcribe 工作區/我的影片
$AE suggest 工作區/我的影片        # → 剪掉清單.md，看過、改好 cuts.json
$AE cut 工作區/我的影片
$AE subs 工作區/我的影片           # → subtitles.txt，錯字寫進 fixes.json
$AE fx 工作區/我的影片             # 照 fx.json 做動態特效（可省略）
$AE finish 工作區/我的影片         # → 成片/我的影片.mp4 + .srt
$AE cover 工作區/我的影片 12.5 "主標" "副標"
```

## 聽打引擎

| 電腦 | 用什麼 | 速度（15 分鐘的影片） |
|---|---|---|
| Mac（M 系列晶片） | mlx-whisper，large-v3-turbo | 幾分鐘 |
| 有 NVIDIA 顯示卡 | faster-whisper，large-v3-turbo | 幾分鐘 |
| 其他（只有 CPU） | faster-whisper，medium | 比較久，可以先去泡咖啡 |

第一次會自動下載模型，約 1.5 GB。

## 錄影小技巧

- 講錯就停一秒，從那句的**開頭**重講。AI 會留最後一次講完整的。
- 不要在一句話中間接著重講，停一下再講，比較好剪。
- 收音盡量安靜。只有一邊聲道有聲音也沒關係，工具會自動處理。

## 想改成你自己的

這套是一個架構，不是一個黑盒子。有剪輯經驗、或有自己想要的功能，都可以拿去改：

- `autoedit/suggest.py`：判斷要剪什麼的規則。
- `autoedit/cut.py`：停頓長度（`LONG_PAUSE`、`PAUSE_KEEP`）、放大倍率（`PUSH_IN`）。
- `autoedit/finish.py`：字幕和字卡的樣式、顏色、位置。
- `autoedit/fx.py`：動態特效，每一種都是一個 class，可以照著加新的。
- `agent/AGENTS.template.md`：AI 的工作流程和規矩。

改完歡迎發 PR。

## 用到的開源專案

- [Whisper](https://github.com/openai/whisper) 語音模型（MIT）
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（MIT）
- [mlx-whisper](https://github.com/ml-explore/mlx-examples)（MIT）
- [OpenCC](https://github.com/BYVoid/OpenCC)：簡轉繁（Apache 2.0）
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)：人像去背，用在動態特效（Apache 2.0），模型第一次使用時下載
- [OpenCV](https://opencv.org)（Apache 2.0）
- [Noto Sans TC](https://github.com/notofonts/noto-cjk)：字型（SIL OFL 1.1），安裝時下載
- [FFmpeg](https://ffmpeg.org)：影音處理，需要自己另外安裝

## 授權

MIT。可以自由使用、修改、商用，保留原作者聲明就好。
