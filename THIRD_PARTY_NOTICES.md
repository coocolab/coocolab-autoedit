# 第三方授權說明

Coocolab AutoEdit 本身是 MIT 授權（見 `LICENSE`）。這個 repo **沒有**附帶下面任何一個套件、模型或字型的檔案；它們是在安裝或第一次使用時，從各自的官方來源下載到你的電腦，並各自適用自己的授權。

## 安裝時下載的 Python 套件（`requirements.txt`、`requirements-fx.txt`）

| 套件 | 用途 | 授權 |
|---|---|---|
| [numpy](https://numpy.org) | 聲音運算 | BSD-3-Clause |
| [Pillow](https://python-pillow.org) | 畫字幕、字卡、封面 | MIT-CMU（HPND） |
| [opencc-python-reimplemented](https://github.com/yichen0831/opencc-python) | 簡轉繁 | Apache-2.0 |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（Windows／Linux／Intel Mac） | 聽打 | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2)（faster-whisper 的依賴） | 聽打推論 | MIT |
| [mlx-whisper](https://github.com/ml-explore/mlx-examples)、[MLX](https://github.com/ml-explore/mlx)（M 系列 Mac） | 聽打 | MIT |
| [PyTorch](https://pytorch.org)、[Numba](https://numba.pydata.org)、tiktoken 等（mlx-whisper 的間接依賴） | — | BSD-3-Clause／BSD-2-Clause／MIT |
| [certifi](https://github.com/certifi/python-certifi) | 下載用的網路憑證 | MPL-2.0 |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe)（選用） | 人像去背（動態特效） | Apache-2.0 |
| [OpenCV](https://opencv.org)（opencv-contrib-python，MediaPipe 帶進來） | 影像處理（動態特效） | Apache-2.0 |

完整清單：裝好後在工具資料夾執行 `.venv/bin/pip list`（Windows：`.venv\Scripts\pip list`）。

## 模型權重

| 模型 | 來源 | 授權 |
|---|---|---|
| Whisper large-v3-turbo（MLX 轉換版） | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo)，原始權重來自 [OpenAI Whisper](https://github.com/openai/whisper) | MIT |
| Whisper large-v3-turbo／medium（CTranslate2 轉換版） | faster-whisper 從 Hugging Face 下載（[mobiuslabsgmbh/faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo)、[Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium)） | MIT |
| MediaPipe Selfie Segmenter（float16，版本 1） | [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter#selfie-model)，[model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf) | Apache-2.0 |

去背模型的網址和 SHA-256 寫在 `autoedit/resources.py`。

## 字型

| 字型 | 來源 | 授權 |
|---|---|---|
| Noto Sans TC Bold（思源黑體繁中子集） | [notofonts/noto-cjk](https://github.com/notofonts/noto-cjk)，固定在 commit `165c01b` | SIL Open Font License 1.1 |

安裝時會把授權全文下載成 `assets/fonts/OFL.txt`，放在字型旁邊。字型的網址和 SHA-256 寫在 `autoedit/resources.py`。下載失敗時，工具會改用電腦內建的中文字型（例如 macOS 的黑體、Windows 的微軟正黑體），那些字型屬於作業系統，不由本工具散布。

## 不包含、需要自己安裝的軟體

- [FFmpeg](https://ffmpeg.org)：影音處理。由你自己用 Homebrew／winget／apt 安裝，授權依你裝的版本而定（LGPL 或 GPL）。本 repo 不散布 FFmpeg。
- [Python](https://www.python.org)：PSF License。
