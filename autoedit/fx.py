"""Motion effects that live AROUND the person: words sliding out from behind the head, a gold light tracing the
outline, sticky notes, glass HUD cards, a typing prompt bar, a photo wall, a video card, a 3D orbit.

How it works: every frame of cut_video.mp4 goes through Python. Where an effect is active, a person matte
(MediaPipe selfie segmentation) cuts the person out, the "back" layer is drawn on the picture, the person is put
back on top, then the "front" layer (glow, the front half of the orbit) is drawn. Output: fx_video.mp4, same
frames, same length, so subtitles and sound from the next step still line up.

fx.json (written by the assistant; "at" = a few words from the subtitles, the effect starts when they are said):
  {"type": "word",  "at": "...", "main": "免費", "small": "FREE", "gold": true, "big": false, "side": "L|R"}
  {"type": "glow",  "at": "..."}
  {"type": "note",  "at": "...", "step": "STEP 1", "title": "先列清單", "lines": [{"at": "...", "text": "..."}]}
  {"type": "stat",  "at": "...", "title": "每月費用", "value": "0", "unit": "元", "sub": "全部在自己電腦"}
  {"type": "progress", "at": "...", "title": "已省下", "from": 0, "to": 5, "unit": "小時"}
  {"type": "list",  "at": "...", "title": "AI 會剪掉", "rows": [{"at": "...", "text": "重講", "tag": "剪", "warn": false}]}
  {"type": "prompt", "at": "...", "text": "幫我剪這支", "checks": [{"at": "...", "label": "聽打"}]}
  {"type": "wall",  "at": "...", "images": "素材/縮圖"}          (no images -> frames taken from this video)
  {"type": "vcard", "at": "...", "src": "素材/示範.mp4", "label": "成片"}
  {"type": "orbit", "at": "...", "items": ["聽打", "剪片", "字幕"], "act": [{"at": "...", "i": 1}]}
  Every effect also takes "dur" (seconds) or "until" (words where it ends), and "side" ("L" / "R") where it fits.
"""
import math
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .common import ROOT, die, ffmpeg, font_path, load, need, norm, save

GOLD = (232, 196, 120)
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/"
             "selfie_segmenter.tflite")
MODEL = ROOT / "assets" / "models" / "selfie_segmenter.tflite"
DEFAULT_DUR = {"word": 2.4, "glow": 1.7, "note": 4.0, "stat": 3.0, "progress": 3.5, "list": 5.0, "prompt": 4.5,
               "wall": 4.0, "vcard": 4.5, "orbit": 6.0}
BACK_TYPES = {"word", "note", "stat", "progress", "list", "prompt", "wall", "vcard"}


# ------------------------------------------------------------------ easing / helpers

def ease_out(p, k=3):
    p = min(max(p, 0.0), 1.0)
    return 1 - (1 - p) ** k


def ease_in(p, k=2):
    p = min(max(p, 0.0), 1.0)
    return p ** k


def back_out(p, s=1.6):
    p = min(max(p, 0.0), 1.0) - 1
    return p * p * ((s + 1) * p + s) + 1


def blit(dst, spr, x, y, alpha=1.0):
    """Alpha-blend an RGBA uint8 sprite onto an RGB float32 frame at (x, y)."""
    if spr is None or alpha <= 0.003:
        return
    H, W = dst.shape[:2]
    h, w = spr.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = s[..., 3:4] / 255.0 * alpha
    reg = dst[y0:y1, x0:x1]
    reg *= (1 - a)
    reg += s[..., :3] * a


def scaled(spr, s):
    if abs(s - 1) < 0.01:
        return spr
    import cv2
    h, w = spr.shape[:2]
    return cv2.resize(spr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def to_np(img):
    return np.asarray(img.convert("RGBA"), dtype=np.uint8).copy()


class Kit:
    """Fonts and the glass / paper drawing style, sized to the video."""

    def __init__(self, W, H, fp):
        self.W, self.H = W, H
        self.u = min(W, H) / 1080.0             # 1.0 at 1080p
        self.fp = fp
        self._f = {}

    def font(self, px):
        px = max(8, int(px * self.u))
        if px not in self._f:
            self._f[px] = ImageFont.truetype(self.fp, px)
        return self._f[px]

    def px(self, v):
        return int(v * self.u)

    def glass(self, w, h, r=26):
        """Charcoal glass card with a thin light border and a soft shadow; returns (img, pad) where pad = shadow margin."""
        pad = self.px(40)
        img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        sh = Image.new("L", img.size, 0)
        ImageDraw.Draw(sh).rounded_rectangle([pad, pad + self.px(14), pad + w, pad + h + self.px(14)], self.px(r), fill=120)
        sh = sh.filter(ImageFilter.GaussianBlur(self.px(22)))
        img.putalpha(sh)
        body = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        grad = Image.linear_gradient("L").resize((w, h))
        top, bot = np.array([34, 40, 56, 205], np.float32), np.array([16, 20, 30, 215], np.float32)
        g = np.asarray(grad, np.float32)[..., None] / 255.0
        col = Image.fromarray((top * (1 - g) + bot * g).astype(np.uint8), "RGBA")
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], self.px(r), fill=255)
        body.paste(col, (0, 0), mask)
        ImageDraw.Draw(body).rounded_rectangle([0, 0, w - 1, h - 1], self.px(r), outline=(255, 255, 255, 46), width=max(1, self.px(1.5)))
        img.alpha_composite(body, (pad, pad))
        return img, pad

    def cap(self, d, x, y, text):
        """Gold dot + small spaced label."""
        f = self.font(24)
        r = self.px(5)
        cy = y + f.size * 0.6
        d.ellipse([x, cy - r, x + 2 * r, cy + r], fill=GOLD + (255,))
        d.text((x + self.px(20), y), " ".join(text) if len(text) <= 8 else text, font=f, fill=GOLD + (240,))


# ------------------------------------------------------------------ person matte

class Matte:
    def __init__(self, W, H):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        if not MODEL.exists():
            MODEL.parent.mkdir(parents=True, exist_ok=True)
            print("下載人像去背模型（約 250 KB）…")
            urllib.request.urlretrieve(MODEL_URL, MODEL)
        opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=str(MODEL)),
                                            running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True)
        self.seg = vision.ImageSegmenter.create_from_options(opts)
        self.mp = mp
        self.W, self.H = W, H
        self.sw = 640 if W >= H else 360
        self.sh = int(self.sw * H / W) // 2 * 2
        self.prev = None
        self.last_t = -1

    def __call__(self, rgb, t):
        import cv2
        small = cv2.resize(rgb, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
        ts = max(self.last_t + 1, int(t * 1000))
        self.last_t = ts
        r = self.seg.segment_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)), ts)
        m = r.confidence_masks[0].numpy_view().astype(np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        self.prev = m if self.prev is None or self.prev.shape != m.shape else self.prev * 0.45 + m * 0.55
        a = np.clip((self.prev - 0.3) / 0.4, 0, 1)
        a = cv2.GaussianBlur(a, (5, 5), 0)
        return cv2.resize(a, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

    def reset(self):
        self.prev = None


def head_box(m):
    """Head of the person from a matte: (left, right, top, bottom) in px, or None."""
    rows = np.where(m.max(axis=1) > 0.2)[0]          # hair is often only half "person"
    if len(rows) < 10:
        return None
    top = rows[0]
    H, W = m.shape
    band = m[top:min(H, top + int(H * 0.22))] > 0.5
    cols = np.where(band.any(axis=0))[0]
    if not len(cols):
        return None
    return int(cols[0]), int(cols[-1]), int(top), int(min(H, top + H * 0.22))


# ------------------------------------------------------------------ effects

class Effect:
    layer = "back"

    def __init__(self, e, kit, t0, t1, ctx):
        self.e, self.k, self.t0, self.t1, self.ctx = e, kit, t0, t1, ctx

    def active(self, t):
        return self.t0 - 0.4 <= t <= self.t1 + 0.1

    def side(self, head, want=None):
        W = self.k.W
        s = want or self.e.get("side")
        if s in ("L", "R"):
            return s
        if head is None:
            return "R"
        return "L" if head[0] > W - head[1] else "R"

    def enter_exit(self, t, enter=0.8, leave=0.45):
        pin = ease_out((t - self.t0) / enter, 4)
        pout = 1 - ease_in((t - (self.t1 - leave)) / leave)
        return pin, pout


class Word(Effect):
    """Characters slide out from behind the head, one after another, with a top-to-bottom gradient."""

    def setup(self, head):
        k = self.k
        self.s = self.side(head)
        main = self.e.get("main", "")
        vertical = k.H > k.W
        big = self.e.get("big")
        size = (260 if big else 128) if not vertical else (200 if big else 150)
        if head:
            room = (head[0] if self.s == "L" else k.W - head[1]) - k.px(60)
        else:
            room = k.W * 0.4
        if vertical:
            room = k.W * 0.86
        f = k.font(size)
        tw = sum(f.getlength(c) for c in main) + k.px(6) * len(main)
        while tw > room and size > 48:
            size -= 8
            f = k.font(size)
            tw = sum(f.getlength(c) for c in main) + k.px(6) * len(main)
        gold = self.e.get("gold")
        stops = ([(255, 250, 240), (246, 220, 156), (217, 167, 74)] if gold else [(255, 255, 255), (227, 237, 251), (141, 183, 236)])
        self.chars = []
        hgt = int(f.size * 1.25)
        for c in main:
            cw = int(f.getlength(c)) + 2
            mask = Image.new("L", (cw + k.px(40), hgt + k.px(40)), 0)
            ImageDraw.Draw(mask).text((k.px(20), k.px(12)), c, font=f, fill=255)
            g = np.linspace(0, 1, mask.height)[:, None]
            col = np.zeros((mask.height, mask.width, 4), np.float32)
            for ch in range(3):
                col[..., ch] = np.interp(g, [0, .45, .8], [stops[0][ch], stops[1][ch], stops[2][ch]])
            col[..., 3] = np.asarray(mask, np.float32) * np.interp(g, [0, .8, 1], [1, 1, .05])
            shadow = mask.filter(ImageFilter.GaussianBlur(k.px(14)))
            sh = np.zeros_like(col)
            sh[..., :3] = (10, 20, 40)
            sh[..., 3] = np.asarray(shadow, np.float32) * 0.35
            out = sh.copy()
            a = col[..., 3:4] / 255.0
            out[..., :3] = col[..., :3] * a + sh[..., :3] * (1 - a)
            out[..., 3] = np.maximum(col[..., 3], sh[..., 3])
            self.chars.append((np.clip(out, 0, 255).astype(np.uint8), cw + k.px(6)))
        self.tw = sum(w for _, w in self.chars)
        if vertical or not head:
            self.x = (k.W - self.tw) / 2
            # about two thirds of the word shows above the head; the rest hides behind it
            self.y = max(k.px(30), head[2] - hgt * 0.92) if head else k.H * 0.12
        else:
            self.x = head[0] - k.px(30) - self.tw if self.s == "L" else head[1] + k.px(30)
            self.y = head[2] + (head[3] - head[2]) * 0.25
        self.small = None
        if self.e.get("small"):
            sf = k.font(38)
            txt = " ".join(self.e["small"]) if len(self.e["small"]) <= 6 else self.e["small"]
            im = Image.new("RGBA", (int(sf.getlength(txt)) + 10, int(sf.size * 1.4)), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((0, 0), txt, font=sf, fill=(214, 226, 245, 220))
            self.small = to_np(im)
        self.hx = (head[0] + head[1]) / 2 if head else k.W / 2

    def draw(self, fr, t, layer):
        k = self.k
        pout = 1 - ease_in((t - (self.t1 - 0.45)) / 0.45)
        n = len(self.chars)
        x = self.x
        order = range(n) if self.s == "R" else range(n - 1, -1, -1)
        rank = {i: r for r, i in enumerate(order)}
        for i, (spr, w) in enumerate(self.chars):
            p = ease_out((t - self.t0 + 0.06 - rank[i] * 0.045) / 0.8, 5)
            if k.H > k.W:                                       # rises from behind the top of the head
                blit(fr, spr, x - k.px(20), self.y - k.px(12) + (1 - p) * spr.shape[0] * 0.8, min(p * 1.5, 1) * pout)
            else:                                               # slides out from the head's centre
                sx = self.hx + (x - self.hx) * p
                blit(fr, spr, sx - k.px(20), self.y - k.px(12), min(p * 1.5, 1) * pout)
            x += w
        if self.small is not None:
            p = ease_out((t - self.t0 - 0.2) / 0.5)
            sx = self.x if self.s == "R" or k.H > k.W else self.x + self.tw - self.small.shape[1]
            if k.H > k.W:
                sx = (k.W - self.small.shape[1]) / 2
            blit(fr, self.small, sx, self.y - self.small.shape[0] - k.px(4) + (1 - p) * k.px(10), p * pout)


class Glow(Effect):
    """A gold light runs from the top of the head down both sides of the outline, holds, then fades."""
    layer = "front"

    def draw(self, fr, t, layer, m=None):
        import cv2
        if m is None:
            return
        TRACE, HOLD, FADE = 0.85, 0.35, 0.45
        kk = t - self.t0
        if kk < 0:
            return
        H, W = m.shape
        cs, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return
        c = max(cs, key=cv2.contourArea)[:, 0, :]
        c = c[c[:, 1] < H - 6]
        if len(c) < 10:
            return
        c = np.roll(c, -int(np.argmin(c[:, 1])), axis=0)
        n = len(c)
        prog = min(1, kk / TRACE)
        half = int(n / 2 * (1 - (1 - prog) ** 3))
        alpha = 1.0 if kk < TRACE + HOLD else max(0, 1 - (kk - TRACE - HOLD) / FADE)
        if alpha <= 0 or half < 2:
            return
        can = np.zeros((H, W), np.float32)
        th = max(2, self.k.px(4))
        cv2.polylines(can, [c[:half].reshape(-1, 1, 2)], False, 1.0, th, cv2.LINE_AA)
        cv2.polylines(can, [c[n - half:].reshape(-1, 1, 2)], False, 1.0, th, cv2.LINE_AA)
        if prog < 1:
            for hx, hy in (c[half - 1], c[n - half]):
                cv2.circle(can, (int(hx), int(hy)), self.k.px(9), 1.0, -1, cv2.LINE_AA)
        g = cv2.GaussianBlur(can, (0, 0), self.k.px(16)) * 2.2 + cv2.GaussianBlur(can, (0, 0), self.k.px(4)) * 1.2 + can
        a = np.clip(g, 0, 1) * alpha * (m < 0.98)
        col = np.array(GOLD, np.float32)[::1]
        core = np.clip(can, 0, 1)[..., None]
        rgb = col * (1 - core) + 255 * core
        fr *= (1 - a[..., None])
        fr += rgb * a[..., None]


class Card(Effect):
    """Base for glass / paper cards that slide in from the side of the head (or the top in vertical video)."""

    def pre(self, head):
        """Vertical video has no room beside the head: cards go above the head, in front of the person."""
        k = self.k
        self.vert = k.H > k.W
        self.maxh = None
        if self.vert:
            self.layer = "front"
            room = (head[2] - k.H * 0.025) if head else k.H * 0.2
            self.maxh = max(room, k.H * 0.15)

    def place(self, head, w, h):
        k = self.k
        if k.H > k.W:
            return (k.W - w) / 2, k.H * 0.02, 0
        s = self.side(head)
        if head:
            x = head[0] - k.px(70) - w if s == "L" else head[1] + k.px(70)
            x = min(max(x, k.px(40)), k.W - w - k.px(40))
        else:
            x = k.px(80) if s == "L" else k.W - w - k.px(80)
        y = self.e.get("y")
        y = k.H * y if isinstance(y, (int, float)) and y <= 1 else k.H * 0.16
        return x, y, (-1 if s == "L" else 1)

    def fit(self, img):
        """Shrink a card that is wider than the free space."""
        k = self.k
        maxw = k.W * (0.88 if k.H > k.W else 0.42)
        s = min(1.0, maxw / img.width)
        if getattr(self, "maxh", None):
            s = min(s, self.maxh / img.height)
        return img if s >= 1 else img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)

    def draw_card(self, fr, t, spr):
        pin, pout = self.enter_exit(t)
        dx = self.k.px(160) * self.dir
        x = self.x + dx * (1 - pin) + dx * 0.6 * (1 - pout)
        y = self.y - (self.k.px(60) * (1 - pin) if self.dir == 0 else 0)
        blit(fr, spr, x, y, min(pin * 1.3, 1) * pout)


class Note(Card):
    """Sticky note with tape, pinned on the wall behind the person; lines appear when they are said."""

    def setup(self, head):
        self.pre(head)
        self.lines = [(self.ctx.find(l.get("at"), self.t0), l.get("text", "")) for l in self.e.get("lines", [])]
        self.cache = {}
        im = self.render(len(self.lines))
        self.x, self.y, self.dir = self.place(head, im.width, im.height)
        self.rot = self.e.get("rot", -2 if self.dir < 0 else 2)

    def render(self, n):
        if n in self.cache:
            return self.cache[n]
        k = self.k
        ft, fl, fs = k.font(44), k.font(30), k.font(22)
        title = self.e.get("title", "")
        w = int(max([ft.getlength(title)] + [fl.getlength(x) for _, x in self.lines] + [k.px(320)]) + k.px(70))
        h = int(k.px(46) + (fs.size * 1.6 if self.e.get("step") else 0) + ft.size * 1.3 + len(self.lines) * fl.size * 1.55 + k.px(34))
        pad = k.px(40)
        img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        sh = Image.new("L", img.size, 0)
        ImageDraw.Draw(sh).rectangle([pad, pad + k.px(18), pad + w, pad + h + k.px(18)], fill=140)
        img.putalpha(sh.filter(ImageFilter.GaussianBlur(k.px(20))))
        paper = Image.new("RGBA", (w, h), (247, 243, 234, 255))
        g = np.linspace(0, 1, h)[:, None, None]
        arr = np.asarray(paper, np.float32)
        arr[..., :3] = arr[..., :3] * (1 - g * 0.06)
        paper = Image.fromarray(arr.astype(np.uint8), "RGBA")
        d = ImageDraw.Draw(paper)
        y = k.px(46)
        if self.e.get("step"):
            d.text((k.px(34), y), " ".join(self.e["step"]), font=fs, fill=(138, 115, 80, 255))
            y += fs.size * 1.6
        d.text((k.px(34), y), title, font=ft, fill=(28, 33, 48, 255))
        y += ft.size * 1.3
        for i, (_, txt) in enumerate(self.lines):
            if i < n:
                d.text((k.px(34), y), "・" + txt, font=fl, fill=(44, 52, 72, 255))
            y += fl.size * 1.55
        img.alpha_composite(paper, (pad, pad))
        tape = Image.new("RGBA", (k.px(150), k.px(40)), (232, 214, 170, 190)).rotate(-3, expand=True)
        img.alpha_composite(tape, (pad + w // 2 - tape.width // 2, pad - k.px(16)))
        img = self.fit(img)
        self.cache[n] = img
        return img

    def draw(self, fr, t, layer):
        n = sum(1 for tl, _ in self.lines if t >= tl)
        key = ("rot", n)
        if key not in self.cache:
            self.cache[key] = to_np(self.render(n).rotate(self.rot, resample=Image.BICUBIC, expand=True))
        pin = back_out((t - self.t0 + 0.1) / 0.7)
        pout = 1 - ease_in((t - (self.t1 - 0.45)) / 0.45)
        spr = scaled(self.cache[key], 1.08 - 0.08 * min(pin, 1))
        y = self.y - self.k.px(60) * (1 - min(pin, 1)) + self.k.px(30) * (1 - pout)
        blit(fr, spr, self.x, y, min(1, max(0, pin)) * pout)


class Stat(Card):
    """Big-number card. A numeric value counts up."""

    def setup(self, head):
        self.pre(head)
        self.cache = {}
        im = self.render(str(self.e.get("value", "")))
        self.x, self.y, self.dir = self.place(head, im.shape[1], im.shape[0])

    def render(self, val):
        if val in self.cache:
            return self.cache[val]
        k = self.k
        fv, fu, fs = k.font(150), k.font(48), k.font(30)
        unit, sub, title = self.e.get("unit", ""), self.e.get("sub", ""), self.e.get("title", "")
        full = str(self.e.get("value", ""))
        vw = fv.getlength(full) + (fu.getlength(" " + unit) if unit else 0)
        w = int(max(vw, fs.getlength(sub), k.px(300)) + k.px(64))
        h = int(k.px(28) + k.px(40) + fv.size * 1.25 + (fs.size * 1.6 if sub else 0) + k.px(26))
        img, pad = k.glass(w, h)
        d = ImageDraw.Draw(img)
        k.cap(d, pad + k.px(32), pad + k.px(26), title)
        y = pad + k.px(66)
        d.text((pad + k.px(32), y), val, font=fv, fill=(255, 255, 255, 255))
        if unit:
            d.text((pad + k.px(32) + fv.getlength(val) + k.px(12), y + fv.size - fu.size * 1.05), unit, font=fu, fill=GOLD + (255,))
        if sub:
            d.text((pad + k.px(32), y + fv.size * 1.28), sub, font=fs, fill=(200, 205, 215, 230))
        img = to_np(self.fit(img))
        self.cache[val] = img
        return img

    def draw(self, fr, t, layer):
        v = str(self.e.get("value", ""))
        m = re.fullmatch(r"(\D*)(\d+(?:\.\d+)?)(\D*)", v)
        if m:                                                   # count up: 0 -> value
            p = ease_out((t - self.t0 - 0.15) / 0.9)
            num = float(m.group(2)) * p
            dec = len(m.group(2).split(".")[1]) if "." in m.group(2) else 0
            v = f"{m.group(1)}{num:.{dec}f}{m.group(3)}"
        self.draw_card(fr, t, self.render(v))


class Progress(Card):
    def setup(self, head):
        self.pre(head)
        self.cache = {}
        im = self.render(0.0)
        self.x, self.y, self.dir = self.place(head, im.shape[1], im.shape[0])

    def render(self, p):
        key = round(p, 2)
        if key in self.cache:
            return self.cache[key]
        k = self.k
        a, b = float(self.e.get("from", 0)), float(self.e.get("to", 100))
        val = a + (b - a) * p
        dec = 1 if (b - a) < 10 and not float(b).is_integer() else 0
        txt = f"{val:.{dec}f}"
        fv, fu, fs = k.font(110), k.font(40), k.font(28)
        w = k.px(560)
        h = k.px(270)
        img, pad = k.glass(w, h)
        d = ImageDraw.Draw(img)
        k.cap(d, pad + k.px(32), pad + k.px(26), self.e.get("title", ""))
        d.text((pad + k.px(32), pad + k.px(70)), txt, font=fv, fill=(255, 255, 255, 255))
        d.text((pad + k.px(32) + fv.getlength(txt) + k.px(12), pad + k.px(70) + fv.size - fu.size * 1.1),
               self.e.get("unit", ""), font=fu, fill=GOLD + (255,))
        by = pad + h - k.px(48)
        d.rounded_rectangle([pad + k.px(32), by, pad + w - k.px(32), by + k.px(12)], k.px(6), fill=(255, 255, 255, 40))
        x1 = pad + k.px(32) + (w - k.px(64)) * p
        if p > 0.01:
            d.rounded_rectangle([pad + k.px(32), by, x1, by + k.px(12)], k.px(6), fill=GOLD + (255,))
        if self.e.get("sub"):
            d.text((pad + k.px(32), by - fs.size * 1.5), self.e["sub"], font=fs, fill=(200, 205, 215, 230))
        img = to_np(self.fit(img))
        self.cache[key] = img
        return img

    def draw(self, fr, t, layer):
        p = ease_out((t - self.t0 - 0.2) / max(0.5, (self.t1 - self.t0) * 0.7), 2)
        self.draw_card(fr, t, self.render(p))


class ListCard(Card):
    """Rows show dimmed, light up when said; the tag on the right pops."""

    def setup(self, head):
        self.pre(head)
        self.rows = [(self.ctx.find(r.get("at"), self.t0 + 0.3 * i), r.get("text", ""), r.get("tag", ""), r.get("warn"))
                     for i, r in enumerate(self.e.get("rows", []))]
        self.cache = {}
        im = self.render(0)
        self.x, self.y, self.dir = self.place(head, im.shape[1], im.shape[0])

    def render(self, n):
        if n in self.cache:
            return self.cache[n]
        k = self.k
        fr_, ft = k.font(36), k.font(26)
        w = int(max([fr_.getlength(r[1]) + ft.getlength(r[2]) for r in self.rows] + [k.px(360)]) + k.px(150))
        rh = int(fr_.size * 1.75)
        h = k.px(80) + rh * len(self.rows) + k.px(16)
        img, pad = k.glass(w, h)
        d = ImageDraw.Draw(img)
        k.cap(d, pad + k.px(32), pad + k.px(26), self.e.get("title", ""))
        for i, (_, txt, tag, warn) in enumerate(self.rows):
            y = pad + k.px(80) + i * rh
            on = i < n
            d.text((pad + k.px(32), y), txt, font=fr_, fill=(255, 255, 255, 255 if on else 72))
            if tag and on:
                tw = ft.getlength(tag) + k.px(28)
                c = (255, 120, 100) if warn else GOLD
                x0 = pad + w - k.px(32) - tw
                d.rounded_rectangle([x0, y + k.px(2), x0 + tw, y + ft.size + k.px(16)], k.px(18), fill=c + (255,))
                d.text((x0 + k.px(14), y + k.px(6)), tag, font=ft, fill=(26, 20, 8, 255))
        img = to_np(self.fit(img))
        self.cache[n] = img
        return img

    def draw(self, fr, t, layer):
        n = sum(1 for r in self.rows if t >= r[0])
        self.draw_card(fr, t, self.render(n))


class Prompt(Card):
    """Glass prompt bar: the text types itself, cursor blinks, the checklist ticks."""

    def setup(self, head):
        self.pre(head)
        self.checks = [(self.ctx.find(c.get("at"), self.t0 + 1.2 + 0.5 * i), c.get("label", ""))
                       for i, c in enumerate(self.e.get("checks", []))]
        self.cache = {}
        im = self.render(len(self.e.get("text", "")), True, len(self.checks))
        self.x, self.y, self.dir = self.place(head, im.shape[1], im.shape[0])

    def render(self, nchar, cursor, nck):
        key = (nchar, cursor, nck)
        if key in self.cache:
            return self.cache[key]
        k = self.k
        ft, fc = k.font(34), k.font(30)
        text = self.e.get("text", "")
        w = int(max(ft.getlength(text) + k.px(40), k.px(600)) + k.px(60))
        rows = math.ceil(len(self.checks) / 2)
        h = int(k.px(70) + ft.size * 1.5 + (rows * fc.size * 1.6 + k.px(22) if self.checks else 0) + k.px(26))
        img, pad = k.glass(w, h)
        d = ImageDraw.Draw(img)
        k.cap(d, pad + k.px(30), pad + k.px(26), self.e.get("title", "PROMPT"))
        y = pad + k.px(70)
        shown = text[:nchar]
        d.text((pad + k.px(30), y), shown, font=ft, fill=(243, 245, 250, 255))
        if cursor:
            cx = pad + k.px(34) + ft.getlength(shown)
            d.rectangle([cx, y + k.px(4), cx + k.px(3), y + ft.size + k.px(6)], fill=GOLD + (255,))
        y += ft.size * 1.5 + k.px(22)
        colw = (w - k.px(60)) / 2
        for i, (_, lab) in enumerate(self.checks):
            x = pad + k.px(30) + (i % 2) * colw
            yy = y + (i // 2) * fc.size * 1.6
            on = i < nck
            r = k.px(17)
            if on:
                d.ellipse([x, yy, x + 2 * r, yy + 2 * r], fill=GOLD + (255,))
                d.line([(x + r * 0.55, yy + r * 1.0), (x + r * 0.9, yy + r * 1.35), (x + r * 1.5, yy + r * 0.65)],
                       fill=(26, 20, 8, 255), width=max(2, k.px(3)))
            else:
                d.ellipse([x, yy, x + 2 * r, yy + 2 * r], outline=(255, 255, 255, 76), width=max(1, k.px(1.5)))
            d.text((x + 2 * r + k.px(12), yy - k.px(2)), lab, font=fc, fill=(255, 255, 255, 255 if on else 90))
        img = to_np(self.fit(img))
        self.cache[key] = img
        return img

    def draw(self, fr, t, layer):
        text = self.e.get("text", "")
        n = int(max(0, t - self.t0 - 0.3) * 16)
        n = min(n, len(text))
        cursor = n < len(text) or int(t * 2.2) % 2 == 0
        nck = sum(1 for c in self.checks if t >= c[0])
        self.draw_card(fr, t, self.render(n, cursor, nck))


class Wall(Effect):
    """A wall of thumbnails behind the person, in 3D perspective, slowly panning."""

    def setup(self, head):
        import cv2
        k = self.k
        src = self.e.get("images")
        imgs = []
        if src:
            p = Path(src)
            p = p if p.is_absolute() else self.ctx.root / p
            files = sorted([f for f in (p.iterdir() if p.is_dir() else []) if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
            imgs = [Image.open(f).convert("RGB") for f in files[:36]]
        if not imgs:
            imgs = self.ctx.thumbs(18)
        cols = 9 if k.W >= k.H else 5
        tw = int(k.W * 1.35 / cols)
        th = int(tw * 16 / 9) if imgs and imgs[0].height > imgs[0].width else int(tw * 9 / 16)
        gap = k.px(22)
        rows = max(3, math.ceil(k.H * 1.5 / (th + gap)))
        G = Image.new("RGBA", (cols * (tw + gap), rows * (th + gap)), (0, 0, 0, 0))
        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tw - 1, th - 1], k.px(16), fill=255)
        for i in range(cols * rows):
            im = imgs[i % len(imgs)]
            s = max(tw / im.width, th / im.height)
            im2 = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.LANCZOS)
            x0, y0 = (im2.width - tw) // 2, (im2.height - th) // 2
            G.paste(im2.crop((x0, y0, x0 + tw, y0 + th)), ((i % cols) * (tw + gap), (i // cols) * (th + gap)), mask)
        self.grid = np.asarray(G, np.uint8)
        gh, gw = self.grid.shape[:2]
        self.src_pts = np.float32([[0, 0], [gw, 0], [gw, gh], [0, gh]])
        yy, xx = np.mgrid[0:k.H, 0:k.W]
        d = np.sqrt(((xx - k.W * 0.54) / (k.W * 0.6)) ** 2 + ((yy - k.H * 0.52) / (k.H * 0.7)) ** 2)
        self.shade = np.clip(0.15 + (d - 0.3) * 0.9, 0.15, 0.82).astype(np.float32)[..., None]
        self.cv2 = cv2

    def draw(self, fr, t, layer):
        cv2, k = self.cv2, self.k
        p = (t - self.t0 + 0.2) / max(0.5, self.t1 - self.t0 + 0.2)
        fade = min(ease_out((t - self.t0 + 0.2) / 0.5), 1 - ease_in((t - (self.t1 - 0.5)) / 0.6))
        if fade <= 0:
            return
        W, H = k.W, k.H
        gh, gw = self.grid.shape[:2]
        sx = W * 1.35 / gw
        cw, ch = gw * sx, gh * sx
        pan = -0.04 * W + 0.08 * W * p
        skew = 0.10 - 0.04 * p                       # perspective: the right edge is a bit further away
        cx, cy = W / 2 + pan, H / 2
        dst = np.float32([[cx - cw / 2, cy - ch / 2 + ch * skew * 0.3], [cx + cw / 2, cy - ch / 2],
                          [cx + cw / 2, cy + ch / 2], [cx - cw / 2, cy + ch / 2 - ch * skew * 0.3]])
        M = cv2.getPerspectiveTransform(self.src_pts, dst)
        warped = cv2.warpPerspective(self.grid, M, (W, H), flags=cv2.INTER_LINEAR)
        a = warped[..., 3:4].astype(np.float32) / 255.0 * fade
        bg = fr * (1 - fade) + np.full_like(fr, 10) * fade
        fr[:] = bg * (1 - a) + warped[..., :3] * a
        fr *= (1 - self.shade * fade)


class VCard(Effect):
    """A 9:16 video card turns out from behind a shoulder and plays a clip."""

    def setup(self, head):
        k = self.k
        src = Path(self.e.get("src", ""))
        src = src if src.is_absolute() else self.ctx.root / src
        if not src.exists():
            die(f"影片卡找不到素材：{src}")
        vertical = k.H > k.W
        if vertical:
            self.layer = "front"
        self.ch = int(k.H * (0.64 if not vertical else 0.26))
        self.cw = int(self.ch * 9 / 16) // 2 * 2
        self.ch = self.ch // 2 * 2
        fps = self.ctx.fps
        n = int((self.t1 - self.t0 + 0.6) * fps)
        raw = subprocess.run([need("ffmpeg"), "-v", "error", "-i", str(src), "-frames:v", str(n), "-vf",
                              f"fps={fps},scale={self.cw}:{self.ch}:force_original_aspect_ratio=increase,crop={self.cw}:{self.ch}",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        self.frames = np.frombuffer(raw, np.uint8).reshape(-1, self.ch, self.cw, 3) if raw else np.zeros((1, self.ch, self.cw, 3), np.uint8)
        mask = Image.new("L", (self.cw, self.ch), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, self.cw - 1, self.ch - 1], k.px(30), fill=255)
        self.mask = np.asarray(mask, np.uint8)
        bd = Image.new("RGBA", (self.cw, self.ch), (0, 0, 0, 0))
        ImageDraw.Draw(bd).rounded_rectangle([0, 0, self.cw - 1, self.ch - 1], k.px(30), outline=(255, 255, 255, 60), width=max(1, k.px(1.5)))
        self.border = np.asarray(bd, np.uint8)
        self.s = self.side(head)
        if vertical:
            self.x = k.px(28) if self.s == "L" else k.W - self.cw - k.px(28)
            self.y = k.H * 0.36
        else:
            self.x = (head[0] - k.px(80) - self.cw) if (head and self.s == "L") else ((head[1] + k.px(80)) if head else k.W * 0.62)
            self.x = min(max(self.x, k.px(40)), k.W - self.cw - k.px(40))
            self.y = (k.H - self.ch) / 2 - k.px(40)
        self.label = None
        if self.e.get("label"):
            f = k.font(26)
            im = Image.new("RGBA", (int(f.getlength(self.e["label"]) * 1.4) + 20, int(f.size * 1.5)), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((0, 0), " ".join(self.e["label"]), font=f, fill=GOLD + (255,))
            self.label = to_np(im)
        self.dir = -1 if self.s == "L" else 1

    def draw(self, fr, t, layer):
        k = self.k
        i = int(max(0, t - self.t0 + 0.3) * self.ctx.fps)
        img = self.frames[min(i, len(self.frames) - 1)]
        spr = np.dstack([img, self.mask])
        spr = np.where(self.border[..., 3:4] > 0, self.border, spr)
        pin = ease_out((t - self.t0 + 0.3) / 0.9, 4)
        pout = 1 - ease_in((t - (self.t1 - 0.5)) / 0.5)
        sc = 0.9 + 0.1 * pin
        dx = k.px(220) * self.dir * (1 - pin) + k.px(130) * self.dir * (1 - pout)
        s2 = scaled(spr, sc)
        # shadow
        sh = np.zeros((s2.shape[0], s2.shape[1], 4), np.uint8)
        sh[..., 3] = (scaled(spr, sc)[..., 3] * 0.45).astype(np.uint8)
        import cv2
        sh[..., 3] = cv2.GaussianBlur(sh[..., 3], (0, 0), k.px(24))
        x = self.x + dx + (self.cw - s2.shape[1]) / 2
        y = self.y + (self.ch - s2.shape[0]) / 2
        a = min(1, pin * 1.4) * pout
        blit(fr, sh, x, y + k.px(24), a)
        blit(fr, s2, x, y, a)
        if self.label is not None:
            p = ease_out((t - self.t0 - 0.3) / 0.5)
            blit(fr, self.label, self.x + dx, self.y + self.ch + k.px(16), p * pout)


class Orbit(Effect):
    """3D ring at chest height; the back half passes behind the person. 'act' turns an item to front-left 45 deg."""
    layer = "both"

    def setup(self, head):
        k = self.k
        self.items = self.e.get("items", [])
        self.acts = [(self.ctx.find(a.get("at"), self.t0 + 1 + i), int(a.get("i", 0))) for i, a in enumerate(self.e.get("act", []))]
        vertical = k.H > k.W
        self.cx = k.W / 2 if not head else (head[0] + head[1]) / 2
        self.cy = k.H * (0.60 if not vertical else 0.55)
        self.rx = k.W * (0.42 if not vertical else 0.44)
        self.ry = k.H * (0.10 if not vertical else 0.06)
        self.sp = self.e.get("speed", 14)
        self.pills = {}
        f = k.font(46 if vertical else 34)
        for i, it in enumerate(self.items):
            for on in (False, True):
                w = int(f.getlength(it) + k.px(56))
                h = int(f.size * 1.9)
                img, pad = k.glass(w, h, r=h // 2)
                d = ImageDraw.Draw(img)
                if on:
                    d.rounded_rectangle([pad, pad, pad + w - 1, pad + h - 1], h // 2, outline=GOLD + (255,), width=max(2, k.px(2.5)))
                d.text((pad + k.px(28), pad + (h - f.size * 1.2) / 2), it, font=f, fill=(GOLD + (255,)) if on else (255, 255, 255, 255))
                self.pills[(i, on)] = to_np(img)

    def angle(self, t):
        N = max(1, len(self.items))
        base = (t - self.t0) * self.sp
        if not self.acts or t < self.acts[0][0] - 1.1:
            return base, None
        tg = lambda k_: 135 - k_ * 360 / N              # angle at which item k sits front-left
        free = (self.acts[0][0] - 1.1 - self.t0) * self.sp
        cur, ts, prev = None, None, None
        for ta, kk in self.acts:
            if t >= ta - 0.15:
                prev, cur, ts = cur, kk, ta
        if cur is None:
            p = (t - (self.acts[0][0] - 1.1)) / 1.1
            q = 4 * p ** 3 if p < .5 else 1 - (-2 * p + 2) ** 3 / 2
            a1 = tg(self.acts[0][1])
            while a1 < free:
                a1 += 360
            return free + (a1 - free) * q, None
        a1 = tg(cur)
        a0 = tg(prev) if prev is not None else a1
        while a1 < a0:
            a1 += 360
        p = min(1, (t - ts + 0.15) / 0.8)
        q = 1 - (1 - p) ** 3
        return a0 + (a1 - a0) * q, cur

    def draw(self, fr, t, layer):
        import cv2
        k = self.k
        fade = min(ease_out((t - self.t0 + 0.2) / 0.6), 1 - ease_in((t - (self.t1 - 0.5)) / 0.5))
        if fade <= 0:
            return
        rot, cur = self.angle(t)
        N = max(1, len(self.items))
        # ring line
        if layer == "back":
            can = np.zeros(fr.shape[:2], np.float32)
            cv2.ellipse(can, (int(self.cx), int(self.cy)), (int(self.rx), int(self.ry)), 0, 180, 360, 1.0, max(1, k.px(2)), cv2.LINE_AA)
            a = (can * 0.35 * fade)[..., None]
            fr *= (1 - a)
            fr += np.array(GOLD, np.float32) * a
        else:
            can = np.zeros(fr.shape[:2], np.float32)
            cv2.ellipse(can, (int(self.cx), int(self.cy)), (int(self.rx), int(self.ry)), 0, 0, 180, 1.0, max(1, k.px(2)), cv2.LINE_AA)
            a = (can * 0.55 * fade)[..., None]
            fr *= (1 - a)
            fr += np.array(GOLD, np.float32) * a
        order = []
        for i in range(N):
            ang = math.radians(rot + i * 360 / N)
            x = self.cx + self.rx * math.cos(ang)
            y = self.cy + self.ry * math.sin(ang)
            front = math.sin(ang) > 0
            order.append((math.sin(ang), i, x, y, front))
        for depth, i, x, y, front in sorted(order):
            if (layer == "back") == front:
                continue
            on = cur == i
            s = 0.78 + 0.22 * (depth + 1) / 2
            spr = scaled(self.pills[(i, on)], s)
            a = fade * (0.45 + 0.55 * (depth + 1) / 2)
            blit(fr, spr, x - spr.shape[1] / 2, y - spr.shape[0] / 2, a)


KINDS = {"word": Word, "glow": Glow, "note": Note, "stat": Stat, "progress": Progress, "list": ListCard,
         "prompt": Prompt, "wall": Wall, "vcard": VCard, "orbit": Orbit}


# ------------------------------------------------------------------ timeline lookup

class Ctx:
    def __init__(self, d, proj, tl):
        self.d, self.proj = Path(d), proj
        self.root = Path(proj["output_dir"]).parent
        self.fps = proj["fps"]
        # search the subtitles (with fixes.json applied), so "at" matches what subtitles.txt shows
        from .finish import build_lines
        lines = build_lines(tl["words"], proj["out_h"] > proj["out_w"], load(d, "fixes.json", {}))
        self.full, self.times = "", []
        for ln in lines:
            for c in ln:
                for ch in norm(c[0]):
                    self.full += ch
                    self.times.append(c[1])
        self.missing = []

    def find(self, phrase, default=None, after=0.0):
        key = norm(phrase or "")
        if not key:
            return default
        p = self.full.find(key)
        while p >= 0 and self.times[p] < after - 0.01:
            p = self.full.find(key, p + 1)
        if p < 0:
            self.missing.append(phrase)
            return default
        return self.times[p]

    def find_end(self, phrase, after):
        key = norm(phrase or "")
        p = self.full.find(key)
        while p >= 0 and self.times[p] < after:
            p = self.full.find(key, p + 1)
        if p < 0:
            return None
        return self.times[min(len(self.times) - 1, p + len(key) - 1)] + 0.3

    def thumbs(self, n):
        """Frames from this video, for a photo wall when no pictures were given."""
        dur = load(self.d, "timeline.json")["duration"]
        out = []
        for i in range(n):
            t = dur * (i + 0.5) / n
            raw = subprocess.run([need("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", str(self.d / "cut_video.mp4"),
                                  "-frames:v", "1", "-vf", "scale=480:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                                 capture_output=True).stdout
            if raw:
                import io
                out.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        return out


# ------------------------------------------------------------------ main

def main(d):
    d = Path(d)
    proj = load(d, "project.json")
    tl = load(d, "timeline.json")
    specs = load(d, "fx.json", [])
    if not specs:
        print("沒有 fx.json，跳過動態特效。")
        return
    ctx = Ctx(d, proj, tl)
    W, H = proj["out_w"], proj["out_h"]
    kit = Kit(W, H, font_path(proj.get("font")))
    effects = []
    for e in specs:
        kind = e.get("type")
        if kind not in KINDS:
            print(f"[注意] 不認得的特效 {kind}，跳過")
            continue
        t0 = ctx.find(e.get("at"))
        if t0 is None:
            continue
        if e.get("until"):
            t1 = ctx.find_end(e["until"], t0) or t0 + DEFAULT_DUR[kind]
        else:
            t1 = t0 + float(e.get("dur", DEFAULT_DUR[kind]))
        t1 = min(t1, tl["duration"])
        effects.append(KINDS[kind](e, kit, t0, t1, ctx))
    if ctx.missing:
        print("[注意] 這些 at／until 找不到對應的字幕：", "、".join(dict.fromkeys(ctx.missing)))
    if not effects:
        return
    effects.sort(key=lambda f: f.t0)
    print(f"特效 {len(effects)} 個：" + "、".join(f"{type(f).__name__}@{f.t0:.1f}s" for f in effects))

    fps_s = f"{proj['fps_num']}/{proj['fps_den']}"
    dec = subprocess.Popen([need("ffmpeg"), "-nostdin", "-v", "error", "-i", str(d / "cut_video.mp4"),
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([need("ffmpeg"), "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", fps_s, "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "16", "-pix_fmt", "yuv420p", str(d / "fx_video.mp4")], stdin=subprocess.PIPE)
    matte = Matte(W, H)
    fsz = W * H * 3
    fps = proj["fps"]
    i = 0
    was_active = False
    while True:
        buf = dec.stdout.read(fsz)
        if len(buf) < fsz:
            break
        t = i / fps
        act = [f for f in effects if f.active(t)]
        if not act:
            enc.stdin.write(buf)
            if was_active:
                matte.reset()
            was_active = False
            i += 1
            continue
        was_active = True
        base = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        m = matte(base, t)
        for f in act:
            if not hasattr(f, "_ready"):
                f.setup(head_box(m)) if hasattr(f, "setup") else None
                f._ready = True
        fr = base.astype(np.float32)
        for f in act:
            if f.layer in ("back", "both"):
                f.draw(fr, t, "back")
        mm = m[..., None]
        fr = fr * (1 - mm) + base.astype(np.float32) * mm         # the person stays in front
        for f in act:
            if isinstance(f, Glow):
                f.draw(fr, t, "front", m)
            elif f.layer in ("front", "both"):
                f.draw(fr, t, "front")
        enc.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
        i += 1
        if i % 300 == 0:
            print(f"  {i / fps:.0f} 秒")
    enc.stdin.close()
    enc.wait()
    dec.wait()
    if i != tl["frames"]:
        print(f"[注意] 特效影片 {i} 格，預期 {tl['frames']} 格")
    proj["video_for_finish"] = "fx_video.mp4"
    save(d, "project.json", proj)
    print(f"動態特效完成：fx_video.mp4（{i} 格）")


def gaps(d, limit=3.5):
    """Stretches where nothing new happens on screen (no cut, no card, no effect) for more than `limit` seconds."""
    d = Path(d)
    tl = load(d, "timeline.json")
    proj = load(d, "project.json")
    ctx = Ctx(d, proj, tl)
    marks = [s["out_a"] for s in tl["spans"]] if proj.get("zoom_cuts", True) else []
    covered = []                                   # while a card or an effect is on screen, something is happening
    for c in load(d, "cards.json", []):
        t = ctx.find(c.get("at"))
        if t is not None:
            covered.append((t, t + float(c.get("dur", 2.6))))
    for e in load(d, "fx.json", []):
        t = ctx.find(e.get("at"))
        if t is not None:
            covered.append((t, t + float(e.get("dur", DEFAULT_DUR.get(e.get("type"), 3)))))
    marks = sorted(set([0.0, tl["duration"]] + marks + [x for c in covered for x in c]))
    out = []
    for a_, b_ in zip(marks, marks[1:]):
        mid = (a_ + b_) / 2
        if b_ - a_ > limit and not any(c0 <= mid <= c1 for c0, c1 in covered):
            out.append((a_, b_))
    speed = float(proj.get("speed", 1.0))
    if out:
        print(f"畫面超過 {limit} 秒沒有變化的地方（成片時間）：")
        for a, b in out:
            seg = "".join(w["w"] for w in tl["words"] if a <= w["t0"] < b)[:30]
            print(f"  {a / speed:6.1f}–{b / speed:6.1f}  {seg}")
    else:
        print(f"沒有超過 {limit} 秒沒變化的地方。")
