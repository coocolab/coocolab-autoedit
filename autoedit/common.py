"""Shared helpers: ffmpeg/ffprobe calls, audio I/O, fonts, project files."""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # the repo folder
SR = 16000                                             # analysis / ASR sample rate
SR_OUT = 48000                                         # output audio sample rate


def die(msg):
    print(f"\n[錯誤] {msg}", file=sys.stderr)
    sys.exit(1)


def need(tool):
    p = shutil.which(tool)
    if not p:
        die(f"找不到 {tool}。請先執行安裝步驟（python install/check_env.py 會告訴你缺什麼）。")
    return p


def run(cmd, quiet=True):
    """Run a command; raise with stderr on failure."""
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        tail = "\n".join(r.stderr.strip().splitlines()[-15:])
        die(f"指令失敗：{' '.join(map(str, cmd[:6]))} ...\n{tail}")
    if not quiet:
        print(r.stdout)
    return r


def ffmpeg(*args):
    return run([need("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-y", *map(str, args)])


def probe(path):
    """Return fps (as fraction num/den), duration, width, height, audio channels."""
    r = run([need("ffprobe"), "-v", "error", "-show_entries",
             "format=duration,start_time:stream=codec_type,width,height,r_frame_rate,avg_frame_rate,channels,start_time",
             "-of", "json", str(path)])
    j = json.loads(r.stdout)
    v = next((s for s in j["streams"] if s.get("codec_type") == "video"), None)
    a = next((s for s in j["streams"] if s.get("codec_type") == "audio"), None)
    if v is None:
        die(f"{path} 裡面沒有影像。")
    if a is None:
        die(f"{path} 裡面沒有聲音，沒辦法聽打。")
    num, den = map(int, (v.get("r_frame_rate") or v.get("avg_frame_rate")).split("/"))
    if num == 0:
        num, den = map(int, v["avg_frame_rate"].split("/"))
    return {"fps_num": num, "fps_den": den, "fps": num / den,
            "duration": float(j["format"]["duration"]),
            "width": int(v["width"]), "height": int(v["height"]),
            "channels": int(a.get("channels", 1)),
            # where video / audio start inside the file (usually 0); used to keep picture and sound in sync
            "fmt_start": float(j["format"].get("start_time", 0) or 0),
            "v_start": float(v.get("start_time", 0) or 0),
            "a_start": float(a.get("start_time", 0) or 0)}


def read_wav(path):
    """16-bit PCM wav -> float32 array shape (n, ch), sample rate."""
    with wave.open(str(path)) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, ch), sr


def write_wav(path, x, sr):
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def rms_db(x, sr, hop_s=0.01, win_s=0.03):
    """Loudness envelope in dBFS, one value per hop (10 ms)."""
    hop, win = int(sr * hop_s), int(sr * win_s)
    p = np.convolve(x * x, np.ones(win) / win, "same")[::hop]
    return 10 * np.log10(p + 1e-10)


def load(d, name, default=None):
    p = Path(d) / name
    if not p.exists():
        if default is not None:
            return default
        die(f"找不到 {p}。是不是前一步還沒跑？")
    return json.loads(p.read_text(encoding="utf-8"))


def save(d, name, obj):
    (Path(d) / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------- Chinese text helpers

_s2t = None


def to_trad(s):
    """Simplified -> Traditional (Taiwan) characters. Whisper often answers in simplified."""
    global _s2t
    if _s2t is None:
        try:
            from opencc import OpenCC
            _s2t = OpenCC("s2tw")
        except Exception:
            _s2t = False
    return _s2t.convert(s) if _s2t else s


def norm(s):
    """Compare-friendly text: traditional, no punctuation/space, lower case, 他/她/牠 -> 它."""
    s = re.sub(r"[^\w]", "", to_trad(s)).lower().replace("_", "")
    return s.translate(str.maketrans("他她牠", "它它它"))


# ---------- fonts

FONT_CANDIDATES = [
    ROOT / "assets" / "fonts" / "NotoSansTC-Bold.otf",            # downloaded by the installer (SIL OFL)
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),              # macOS
    Path("C:/Windows/Fonts/msjhbd.ttc"),                          # Windows: Microsoft JhengHei Bold
    Path("C:/Windows/Fonts/msjh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),  # Linux
    Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"),
]


def font_path(custom=None):
    if custom and Path(custom).exists():
        return str(custom)
    for p in FONT_CANDIDATES:
        if p.exists():
            return str(p)
    die("找不到中文粗體字型。請重新執行安裝，或在 project.json 的 font 填一個 .ttf/.ttc 路徑。")


def os_name():
    s = platform.system()
    return {"Darwin": "mac", "Windows": "windows"}.get(s, "linux")


def ffmpeg_major():
    r = subprocess.run([need("ffmpeg"), "-version"], capture_output=True, text=True)
    m = re.search(r"ffmpeg version n?(\d+)", r.stdout)
    return int(m.group(1)) if m else 6


def cpu_threads():
    return max(1, (os.cpu_count() or 4) - 1)
