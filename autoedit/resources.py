"""Files downloaded at install / first use: pinned versions + SHA-256, so every user gets the same bytes and a
broken or tampered download is caught. Standard library only (the installer imports this before any package)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Noto Sans TC Bold (SIL OFL 1.1), pinned to a noto-cjk commit
_NOTO = "https://raw.githubusercontent.com/notofonts/noto-cjk"
FONT_URL = f"{_NOTO}/165c01b46ea533872e002e0785ff17e44f6d97d8/Sans/SubsetOTF/TC/NotoSansTC-Bold.otf"
FONT_SHA256 = "55420b259eb119bf5f2a0aadba10cf9d736c12d64ab93e78546d69ef5f43558b"
FONT = ROOT / "assets" / "fonts" / "NotoSansTC-Bold.otf"
FONT_LICENSE_URL = f"{_NOTO}/f8d157532fbfaeda587e826d4cd5b21a49186f7c/Sans/LICENSE"
FONT_LICENSE_SHA256 = "6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2"
FONT_LICENSE = ROOT / "assets" / "fonts" / "OFL.txt"

# MediaPipe selfie segmenter (Apache 2.0), version 1 (the same file "latest" pointed to in 2026)
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/"
             "selfie_segmenter.tflite")
MODEL_SHA256 = "191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b"
MODEL = ROOT / "assets" / "models" / "selfie_segmenter.tflite"
