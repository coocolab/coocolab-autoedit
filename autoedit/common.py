"""Shared helpers: ffmpeg/ffprobe calls, audio I/O, fonts, project files."""
import json
import os
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
        die(f"找不到 {tool}。請先執行安裝步驟（python install/setup.py check 會告訴你缺什麼）。")
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


HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}          # HLG (iPhone default) and PQ (HDR10 / Dolby Vision)
STD_RATES = [(24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1), (50, 1), (60000, 1001), (60, 1)]


def _frac(s):
    try:
        a, b = map(int, str(s).split("/"))
        return (a, b) if a > 0 and b > 0 else None
    except ValueError:
        return None


def _seconds(s):
    """'12.5' or '00:00:12.500000000' -> 12.5; None if missing / unreadable."""
    try:
        if s is None:
            return None
        if ":" in str(s):
            h, m, sec = str(s).split(":")
            return int(h) * 3600 + int(m) * 60 + float(sec)
        return float(s)
    except ValueError:
        return None


def _rotation(v):
    """Display rotation in degrees (phones store portrait video as landscape + a 'turn 90 degrees' flag)."""
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    try:
        return int(float((v.get("tags") or {}).get("rotate", 0))) % 360
    except (TypeError, ValueError):
        return 0


def cfr_rate(avg):
    """A steady frame rate for a variable-frame-rate clip: the standard rate near its average, else a round number."""
    num, den = avg
    f = num / den
    for n, d in STD_RATES:
        if abs(f - n / d) / (n / d) < 0.03:
            return n, d
    return min(60, max(24, round(f))), 1


def probe(path):
    """fps (fraction), duration, size as shown (rotation applied), audio channels, start times, and flags for
    clips that need a steady work copy first: variable frame rate (vfr) and HDR."""
    r = run([need("ffprobe"), "-v", "error", "-show_entries",
             "format=duration,start_time:stream=codec_type,width,height,r_frame_rate,avg_frame_rate,channels,"
             "start_time,duration,color_transfer,pix_fmt:stream_side_data=rotation:stream_tags=rotate,DURATION",
             "-of", "json", str(path)])
    return parse_probe(json.loads(r.stdout), path)


def parse_probe(j, path=""):
    v = next((s for s in j["streams"] if s.get("codec_type") == "video"), None)
    a = next((s for s in j["streams"] if s.get("codec_type") == "audio"), None)
    if v is None:
        die(f"{path} 裡面沒有影像。")
    if a is None:
        die(f"{path} 裡面沒有聲音，沒辦法聽打。")
    rf, af = _frac(v.get("r_frame_rate")), _frac(v.get("avg_frame_rate"))
    num, den = rf or af or (30, 1)
    # r_frame_rate is the container's "nominal" rate; on screen recordings and some phones the real average is far
    # lower. Then frame k is NOT at k/fps seconds and a cut drifts out of sync -> make a steady (CFR) copy first.
    vfr = bool(rf and af and abs(rf[0] / rf[1] - af[0] / af[1]) / (rf[0] / rf[1]) > 0.01)
    w, h = int(v["width"]), int(v["height"])
    rot = _rotation(v)
    if rot in (90, 270):           # ffmpeg turns the picture upright when decoding, so use the upright size
        w, h = h, w
    dur = float(j["format"]["duration"])
    v_dur = _seconds(v.get("duration")) or _seconds((v.get("tags") or {}).get("DURATION")) or dur   # mkv: tag
    return {"fps_num": num, "fps_den": den, "fps": num / den,
            "duration": dur, "v_duration": v_dur,
            "width": w, "height": h, "rotation": rot,
            "channels": int(a.get("channels", 1)),
            "vfr": vfr, "avg_fps": list(af) if af else None,
            "hdr": v.get("color_transfer") in HDR_TRANSFERS,
            # where video / audio start inside the file (usually 0); used to keep picture and sound in sync
            "fmt_start": float(j["format"].get("start_time", 0) or 0),
            "v_start": float(v.get("start_time", 0) or 0),
            "a_start": float(a.get("start_time", 0) or 0)}


def has_filter(name):
    r = subprocess.run([need("ffmpeg"), "-hide_banner", "-filters"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", r.stdout, re.M) is not None


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
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        die(f"{p.name} 的格式壞了（第 {e.lineno} 行第 {e.colno} 個字附近：{e.msg}）。請修正這個檔，再重跑這一步。")


def save(d, name, obj):
    """Write via a temp file + rename, so a step that gets interrupted never leaves half a json behind."""
    p = Path(d) / name
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


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


_warned = {}


def font_path(custom=None):
    if custom and Path(custom).exists():
        return str(custom)
    from .download import is_ok
    from .resources import FONT_SHA256
    for p in FONT_CANDIDATES:
        if p.exists():
            if p == FONT_CANDIDATES[0] and not is_ok(p, FONT_SHA256):
                if not _warned.get("font"):
                    print("[注意] 下載的字型檔不完整，先改用電腦內建字型。重跑一次 install 會重新下載。", file=sys.stderr)
                    _warned["font"] = True
                continue
            return str(p)
    die("找不到中文粗體字型。請重新執行安裝，或在 project.json 的 font 填一個 .ttf/.ttc 路徑。")


def ffmpeg_version():
    """(major, minor) of the installed ffmpeg, or None when the build does not say (e.g. git snapshot builds)."""
    r = subprocess.run([need("ffmpeg"), "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", r.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None
