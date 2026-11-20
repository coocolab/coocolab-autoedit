"""Steps 4-5: subtitles, word cards, speed, loudness, final video.

subs   -> subtitles.txt (one line per subtitle, for proofreading) + speaking-rate report
finish -> 成片/<name>.mp4 + .srt, using fixes.json / keywords / cards.json written by the assistant

Subtitles and cards are drawn with Pillow into transparent PNGs and laid over the video with ffmpeg's
overlay filter. This works with any ffmpeg build (no libass / drawtext needed) on Mac, Windows and Linux.
"""
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import die, ffmpeg, font_path, load, need, norm, run

PUNCT = "，。！？、；：,.!?;:…—「」『』（）()\"'"
ORANGE = (255, 159, 28, 255)
WHITE = (255, 255, 255, 255)


# ---------------------------------------------------------------- subtitle lines

def _chars(words):
    """Words -> [[char, t0, t1, island, word_index]], each word's time split evenly over its characters."""
    out = []
    for wi, w in enumerate(words):
        txt = w["w"]
        n = max(1, len(txt))
        for k, ch in enumerate(txt):
            out.append([ch, w["t0"] + (w["t1"] - w["t0"]) * k / n, w["t0"] + (w["t1"] - w["t0"]) * (k + 1) / n,
                        w.get("island"), w.get("_i", wi)])
    return out


def _width(s):
    """Display width in 'Chinese characters': Latin letters and digits count about half."""
    return sum(0.55 if ord(ch) < 0x2E80 else 1.0 for ch in s if ch not in PUNCT)


def spaced(s):
    """A space between Chinese and Latin/digits: 'Claude說' -> 'Claude 說'."""
    s = re.sub(r"([\u2E80-\u9FFF])([A-Za-z0-9$])", r"\1 \2", s)
    return re.sub(r"([A-Za-z0-9%])([\u2E80-\u9FFF])", r"\1 \2", s)


def glue(words):
    """The speech model splits English into pieces ('many', 'chat'). Join pieces that had no space before them;
    keep one space between real English words ('Claude Code')."""
    out = []
    for w in words:
        if out and re.search(r"[A-Za-z0-9]$", out[-1]["w"]) and re.match(r"[A-Za-z0-9]", w["w"]):
            out[-1] = {**out[-1], "w": out[-1]["w"] + (" " if w.get("sp") else "") + w["w"], "t1": w["t1"]}
        else:
            out.append(dict(w))
    for i, w in enumerate(out):
        w["_i"] = i
    return out


CLINGY = "多個的了嗎呢吧啊喔耶們後裡上下中"     # a line should not start with these
SENT_END = "，。！？,.!?；;"


def _fix_key(k):
    """fixes.json keys ignore the spaces that subtitles.txt adds between Chinese and English."""
    return re.sub(r"(?<![A-Za-z0-9]) | (?![A-Za-z0-9])", "", k)


def _apply_fixes(ln, fixes):
    for wrong, right in (fixes or {}).items():
        wrong = _fix_key(wrong)
        if not wrong:
            continue
        s = "".join(c[0] for c in ln)
        p = s.find(wrong)
        while p >= 0:
            t0, t1 = ln[p][1], ln[p + len(wrong) - 1][2]
            n = max(1, len(right))
            new = [[ch, t0 + (t1 - t0) * i / n, t0 + (t1 - t0) * (i + 1) / n, ln[p][3], -1]
                   for i, ch in enumerate(right)]
            ln[p:p + len(wrong)] = new
            s = "".join(c[0] for c in ln)
            p = s.find(wrong, p + len(right))
    return ln


def _split(ln, maxc):
    """Split a too-long phrase at the best place: a pause, a word boundary, near the middle."""
    w = _width("".join(c[0] for c in ln))
    if w <= maxc or len(ln) < 4:
        return [ln]
    best, bi = None, None
    for i in range(2, len(ln) - 1):
        a, b = ln[i - 1], ln[i]
        if re.match(r"[A-Za-z0-9.%$]", a[0]) and re.match(r"[A-Za-z0-9.%$]", b[0]):
            continue                                   # never inside '900' or 'ManyChat'
        left = _width("".join(c[0] for c in ln[:i]))
        right = w - left
        sc = min(0.5, b[1] - a[2]) * 8                 # a pause is the best place
        sc += 0.8 if a[4] != b[4] else -1.5            # word boundary vs inside a word
        sc -= 1.2 if b[0] in CLINGY else 0
        sc -= abs(left - right) / max(1, w) * 2.5      # balanced lines
        sc -= 4 if max(left, right) > maxc + 1 else 0
        if best is None or sc > best:
            best, bi = sc, i
    if bi is None:
        return [ln]
    return _split(ln[:bi], maxc) + _split(ln[bi:], maxc)


def build_lines(words, vertical, fixes):
    """Words -> subtitle lines. A new phrase starts at a pause, a new speech island, or the end of a sentence;
    long phrases are split at the best point. In fixes.json, "｜" in a replacement forces a line break."""
    maxc = 12 if vertical else 16
    words = glue(words)
    chars = _chars(words)
    phrases, cur = [], []
    for c in chars:
        if cur:
            gap = c[1] - cur[-1][2]
            if gap > 0.35 or c[3] != cur[-1][3] or (cur[-1][0] in SENT_END and len(cur) >= 5):
                phrases.append(cur)
                cur = []
        cur.append(c)
    if cur:
        phrases.append(cur)
    lines = []
    for ph in phrases:
        ph = [c for c in ph if c[0] not in PUNCT]
        ph = _apply_fixes(ph, fixes)
        parts, part = [], []
        for c in ph:                                   # manual break
            if c[0] == "｜":
                parts.append(part)
                part = []
            else:
                part.append(c)
        parts.append(part)
        for part in parts:
            while part and part[0][0] == " ":
                part.pop(0)
            while part and part[-1][0] == " ":
                part.pop()
            if part:
                lines += _split(part, maxc)
    # a 1-2 character line joins its neighbour in the same island when it fits
    merged = []
    for ln in lines:
        if (merged and len(ln) <= 2 and merged[-1][-1][3] == ln[0][3]
                and _width("".join(c[0] for c in merged[-1] + ln)) <= maxc + 1 and ln[0][1] - merged[-1][-1][2] < 0.5):
            merged[-1] += ln
        else:
            merged.append(ln)
    return merged


def subs(d):
    d = Path(d)
    proj = load(d, "project.json")
    tl = load(d, "timeline.json")
    fixes = load(d, "fixes.json", {})
    vertical = proj["out_h"] > proj["out_w"]
    lines = build_lines(tl["words"], vertical, fixes)
    txt = [f"{i:03d}  {ln[0][1]:7.2f}  {spaced(''.join(c[0] for c in ln))}" for i, ln in enumerate(lines)]
    (d / "subtitles.txt").write_text("\n".join(txt), encoding="utf-8")
    n = sum(len(norm(''.join(c[0] for c in ln))) for ln in lines)
    talk = sum(ln[-1][2] - ln[0][1] for ln in lines)
    rate = n / max(talk, 1e-6)
    print(f"字幕 {len(lines)} 行 → subtitles.txt")
    tip = ("不用加速。" if rate >= 5 else
           f"建議加速，可在 project.json 設 speed 為 {1.1 if rate >= 4 else 1.2}（先問使用者）。")
    print(f"語速：每秒約 {rate:.1f} 字（講話的時間裡）。一般口播約 4.5～5.5 字，{tip}")


# ---------------------------------------------------------------- drawing

class Painter:
    def __init__(self, proj):
        self.W, self.H = proj["out_w"], proj["out_h"]
        self.vertical = self.H > self.W
        base = self.W if self.vertical else self.H
        fp = font_path(proj.get("font"))
        self.sub_size = int(base * (0.068 if self.vertical else 0.058))
        self.f_sub = ImageFont.truetype(fp, self.sub_size)
        self.f_card = ImageFont.truetype(fp, int(base * (0.085 if self.vertical else 0.072)))
        self.f_small = ImageFont.truetype(fp, int(base * (0.045 if self.vertical else 0.036)))
        self.keywords = [k for k in proj.get("keywords", []) if k]

    def sub(self, img, chars, shown):
        """Draw one subtitle line; only the first `shown` characters are visible."""
        d = ImageDraw.Draw(img)
        text = "".join(c[0] for c in chars)
        hot = [False] * len(text)
        for k in self.keywords:
            for m in re.finditer(re.escape(k), text):
                for i in range(m.start(), m.end()):
                    hot[i] = True
        widths = [d.textlength(ch, font=self.f_sub) for ch in text]
        sp = self.sub_size * 0.04
        cjk = [ord(ch) >= 0x2E80 for ch in text]
        # gap before each char: normal spacing, wider between Chinese and Latin/digits
        gaps = [0.0] + [sp + (self.sub_size * 0.22 if cjk[i] != cjk[i - 1] else 0) for i in range(1, len(text))]
        total = sum(widths) + sum(gaps)
        font = self.f_sub
        if total > self.W * 0.92:                       # never run off the screen
            k = self.W * 0.92 / total
            font = ImageFont.truetype(self.f_sub.path, max(10, int(self.sub_size * k)))
            widths = [d.textlength(ch, font=font) for ch in text]
            gaps = [g * k for g in gaps]
            total = sum(widths) + sum(gaps)
        x = (self.W - total) / 2
        y = self.H * (0.70 if self.vertical else 0.86) - self.sub_size / 2
        stroke = max(2, int(self.sub_size * 0.10))
        for i, ch in enumerate(text):
            x += gaps[i]
            if i < shown:
                d.text((x, y), ch, font=font, fill=ORANGE if hot[i] else WHITE,
                       stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            x += widths[i]

    def card(self, img, c, prog_in, prog_out):
        """A word card. prog_in / prog_out: 0..1 animation progress (1 = fully shown)."""
        main, small = c.get("text", ""), c.get("sub", "")
        tmp = ImageDraw.Draw(img)
        mw = tmp.textlength(main, font=self.f_card)
        sw = tmp.textlength(small, font=self.f_small) if small else 0
        pad = int(self.f_card.size * 0.45)
        cw = int(max(mw, sw) + pad * 2 + pad * 0.5)
        ch = int(self.f_card.size * 1.25 + (self.f_small.size * 1.35 if small else 0) + pad * 1.2)
        layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        r = int(pad * 0.6)
        ld.rounded_rectangle([0, 0, cw - 1, ch - 1], r, fill=(18, 18, 24, 215))
        ld.rounded_rectangle([0, 0, int(pad * 0.32), ch - 1], r // 2, fill=ORANGE)
        ld.text((pad, pad * 0.5), main, font=self.f_card, fill=WHITE)
        if small:
            ld.text((pad, pad * 0.5 + self.f_card.size * 1.22), small, font=self.f_small, fill=(210, 210, 215, 255))
        # pop in (overshoot), fade out
        e = 1 - (1 - prog_in) ** 3
        s = 0.82 + 0.18 * e + 0.06 * (1 - abs(2 * prog_in - 1)) * (prog_in < 1)
        alpha = min(e, prog_out)
        if s != 1:
            layer = layer.resize((max(1, int(cw * s)), max(1, int(ch * s))), Image.LANCZOS)
        if alpha < 1:
            a = layer.getchannel("A").point(lambda v: int(v * alpha))
            layer.putalpha(a)
        pos = c.get("pos", "top" if self.vertical else "left")
        m = int(min(self.W, self.H) * 0.05)
        if pos == "right":
            x = self.W - m - layer.width
        elif pos == "top":
            x = (self.W - layer.width) // 2
        else:
            x = m
        y = int(self.H * (0.14 if self.vertical else 0.08)) + (ch - layer.height) // 2
        img.alpha_composite(layer, (max(0, x), max(0, y)))


# ---------------------------------------------------------------- cards on the timeline

def place_cards(cards, lines, speed):
    """cards.json items: {"at": "words where it should pop up", "text": "...", "sub": "...", "dur": 2.5, "pos": ...}"""
    full, times = "", []
    for ln in lines:
        for c in ln:
            n = norm(c[0])
            for ch in n:
                full += ch
                times.append(c[1])
    placed, missing = [], []
    for c in cards:
        key = norm(c.get("at", ""))
        p = full.find(key) if key else -1
        if p < 0:
            missing.append(c.get("at", ""))
            continue
        t0 = times[p] / speed
        placed.append({**c, "t0": t0, "t1": t0 + float(c.get("dur", 2.6))})
    return placed, missing


# ---------------------------------------------------------------- final render

def finish(d):
    d = Path(d)
    proj = load(d, "project.json")
    tl = load(d, "timeline.json")
    speed = float(proj.get("speed", 1.0))
    fps = proj["fps"]
    vertical = proj["out_h"] > proj["out_w"]
    lines = build_lines(tl["words"], vertical, load(d, "fixes.json", {}))
    cards, missing = place_cards(load(d, "cards.json", []), lines, speed)
    if missing:
        print("[注意] 這些字卡找不到對應的字幕，沒有放：", "、".join(missing))
    dur = tl["duration"] / speed
    P = Painter(proj)

    # subtitle state changes: every character start; a line stays up until the next line (max +0.6 s)
    L = []
    for i, ln in enumerate(lines):
        s0 = ln[0][1] / speed
        nxt = lines[i + 1][0][1] / speed if i + 1 < len(lines) else dur
        e = min(nxt, ln[-1][2] / speed + 0.6)
        L.append((s0, e, ln))
    events = {0.0, dur}
    for s0, e, ln in L:
        events.add(round(s0, 3))
        events.add(round(e, 3))
        for c in ln:
            events.add(round(max(s0, c[1] / speed), 3))
    ANIM_IN, ANIM_OUT = 0.28, 0.22
    for c in cards:
        for k in range(int(ANIM_IN * fps) + 1):
            events.add(round(c["t0"] + k / fps, 3))
        for k in range(int(ANIM_OUT * fps) + 1):
            events.add(round(c["t1"] - ANIM_OUT + k / fps, 3))
        events.add(round(c["t1"], 3))
    ev = sorted(t for t in events if 0 <= t <= dur)

    ov = d / "overlay"
    if ov.exists():
        shutil.rmtree(ov)
    ov.mkdir()
    cache, entries = {}, []
    li = 0
    for k, t in enumerate(ev[:-1]):
        nxt = ev[k + 1]
        if nxt - t < 1e-4:
            continue
        while li + 1 < len(L) and L[li + 1][0] <= t + 1e-6:
            li += 1
        sub_state = None
        if L and L[li][0] <= t + 1e-6 < L[li][1]:
            ln = L[li][2]
            shown = sum(1 for c in ln if c[1] / speed <= t + 1e-6)
            sub_state = (li, max(1, shown))
        card_state = []
        for ci, c in enumerate(cards):
            if c["t0"] <= t < c["t1"]:
                pin = min(1.0, (t - c["t0"]) / ANIM_IN)
                pout = min(1.0, (c["t1"] - t) / ANIM_OUT)
                card_state.append((ci, round(pin, 2), round(pout, 2)))
        key = (sub_state, tuple(card_state))
        if key not in cache:
            img = Image.new("RGBA", (proj["out_w"], proj["out_h"]), (0, 0, 0, 0))
            for ci, pin, pout in card_state:
                P.card(img, cards[ci], pin, pout)
            if sub_state:
                P.sub(img, L[sub_state[0]][2], sub_state[1])
            name = hashlib.md5(repr(key).encode()).hexdigest()[:16] + ".png"
            img.save(ov / name, compress_level=1)
            cache[key] = name
        entries.append((cache[key], nxt - t))
    with open(ov / "list.txt", "w", encoding="utf-8") as f:
        for name, du in entries:
            f.write(f"file '{name}'\nduration {du:.4f}\n")
        f.write(f"file '{entries[-1][0]}'\n")
    print(f"字幕 {len(lines)} 行、字卡 {len(cards)} 張，畫了 {len(cache)} 張圖層")

    # sound: speed, loudness -14 LUFS (two passes), voice in the middle of left and right
    a_in = d / "cut48.wav"
    tempo = f"atempo={speed}," if speed != 1 else ""
    tempo += "aresample=48000,pan=stereo|c0=c0|c1=c0,"     # voice in the middle; measure loudness after this
    r = run([need("ffmpeg"), "-nostdin", "-hide_banner", "-i", str(a_in), "-af",
             f"{tempo}loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"])
    m = json.loads(re.findall(r"\{[^{}]+\}", r.stderr)[-1])
    try:
        ok = all(math.isfinite(float(m[k])) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"))
    except (KeyError, ValueError):
        ok = False
    if ok:
        ln_args = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                   f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
                   "linear=true,")
    else:
        print("[注意] 聲音幾乎都是靜音，量不到音量，這次不調整音量。請確認原片有收到聲音。")
        ln_args = ""
    ffmpeg("-i", a_in, "-af", f"{tempo}{ln_args}aresample=48000", "-c:a", "pcm_s16le",
           d / "final_audio.wav")

    out_dir = Path(proj["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{proj['name']}.mp4"
    if out.resolve() == Path(proj["source"]).resolve():
        die(f"成片的檔名會跟原片一樣（{out}），會蓋掉原片，所以停下來。請把原片放在「原片」資料夾，重新 init。")
    # frame k of the cut is at k / fps (the cut is built that way); index-based times avoid a dropped or doubled
    # frame where two pieces join
    fr_s = f"{proj['fps_num']}/{proj['fps_den']}"
    vf = (f"[0:v]setpts=N/({fr_s})/TB/{speed}[b];[1:v]format=rgba[o];"
          f"[b][o]overlay=0:0:eof_action=pass:format=auto,fps={fr_s},format=yuv420p[v]")
    print("輸出成片中…（長片要幾分鐘）")
    video = d / "cut_video.mp4"
    fxv = proj.get("video_for_finish")
    if fxv:
        if (d / fxv).exists() and os.path.getmtime(d / fxv) >= os.path.getmtime(video):
            video = d / fxv
        else:
            print("[注意] 特效版影片比剪好的影片舊（重剪過之後沒有重跑 fx），這次用沒有特效的版本。")
    tmp = out.with_name(out.stem + ".輸出中.mp4")
    ffmpeg("-i", video, "-f", "concat", "-safe", 0, "-i", ov / "list.txt", "-i", d / "final_audio.wav",
           "-filter_complex", vf, "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-preset", "medium", "-crf", 19,
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", tmp)
    os.replace(tmp, out)

    # .srt next to the video
    srt = []
    for i, (s0, e, ln) in enumerate(L, 1):
        def ts(t):
            ms = int(round(t * 1000))
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        srt += [str(i), f"{ts(s0)} --> {ts(e)}", spaced("".join(c[0] for c in ln)), ""]
    out.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")
    print(f"完成：{out}")
    qa(out)


def qa(path):
    """Loudness, left/right balance, picture vs sound length."""
    r = run([need("ffmpeg"), "-nostdin", "-hide_banner", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"])
    I = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    r2 = run([need("ffmpeg"), "-nostdin", "-hide_banner", "-i", str(path), "-af", "astats=measure_overall=none",
              "-f", "null", "-"])
    rms = [float(x) for x in re.findall(r"RMS level dB: (-?[\d.]+)", r2.stderr)]
    r3 = run([need("ffprobe"), "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)])
    st = {s["codec_type"]: float(s.get("duration", 0)) for s in json.loads(r3.stdout)["streams"]}
    loud = float(I[-1]) if I else None
    print("── 交片檢查 ──")
    print(f"響度：{loud} LUFS（目標 -14）", "PASS" if loud is not None and abs(loud + 14) <= 1 else "注意")
    if len(rms) >= 2:
        print(f"左右聲道：{rms[0]:.1f} / {rms[1]:.1f} dB", "PASS" if abs(rms[0] - rms[1]) <= 1 else "注意")
    if "video" in st and "audio" in st:
        diff = abs(st["video"] - st["audio"])
        print(f"影像 {st['video']:.2f} 秒／聲音 {st['audio']:.2f} 秒", "PASS" if diff < 0.1 else "注意")
