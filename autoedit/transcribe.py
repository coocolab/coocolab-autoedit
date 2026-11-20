"""Step 1: pull out the voice, split it into speech "islands" at pauses, transcribe each island on its own.

Why island by island: when the whole clip is transcribed in one go, the speech model tends to merge
"I said half a sentence, stopped, and said it again" into one clean sentence, so the retake becomes
invisible. Transcribing each island separately keeps both takes in the transcript.
"""
import re
import time
from pathlib import Path

import numpy as np

from .common import (SR, SR_OUT, die, ffmpeg, load, norm, probe, read_wav, rms_db, save,
                     to_trad, write_wav)

# Text the speech model invents on silence / noise
HALLU = re.compile(r"(謝謝(大家)?(收看|觀看)|字幕(製作|由)|請不吝|點贊|訂閱|明鏡|mingpao|下(個|一個)?(節目|影片|集|期)再見|"
                   r"amara|優優獨播|中文字幕|by索蘭婭)", re.I)


def looks_made_up(ws, dur):
    """The speech model 'hallucinates' on laughs and noise: the same 1-4 characters looping, or far more
    characters than anyone can say in that time (people say at most ~9 Chinese characters a second)."""
    if not ws:
        return False
    txt = norm("".join(w for w, _, _ in ws))
    if re.search(r"(.{1,4})\1{4,}", txt):
        return True
    # letters from scripts nobody is speaking here (Korean, Japanese kana, Cyrillic, Arabic, Thai…)
    if re.search(r"[\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0e00-\u0e7f]", txt):
        return True
    return len(txt) > max(4, 11 * dur)


def extract_voice(d, src, info):
    """Write voice48.wav (mono, 48 kHz) and a16.wav (mono, 16 kHz). Pick the channel that has the voice."""
    d = Path(d)
    tmp = d / "src48.wav"
    ffmpeg("-i", src, "-vn", "-ac", str(min(info["channels"], 2)), "-ar", SR_OUT, "-c:a", "pcm_s16le", tmp)
    x, sr = read_wav(tmp)
    if x.shape[1] == 2:
        lv = np.sqrt(np.mean(x[:, 0] ** 2)) + 1e-9
        rv = np.sqrt(np.mean(x[:, 1] ** 2)) + 1e-9
        if rv / lv > 8:
            voice, how = x[:, 1], "只有右聲道有聲音，改用右聲道（輸出會放到左右兩邊）"
        elif lv / rv > 8:
            voice, how = x[:, 0], "只有左聲道有聲音，改用左聲道（輸出會放到左右兩邊）"
        else:
            voice, how = x.mean(axis=1), "左右聲道都有聲音，合成一聲道（人聲置中）"
    else:
        voice, how = x[:, 0], "單聲道錄音（輸出會放到左右兩邊）"
    write_wav(d / "voice48.wav", voice, SR_OUT)
    ffmpeg("-i", d / "voice48.wav", "-ar", SR, "-ac", 1, d / "a16.wav")
    tmp.unlink()
    print("聲道：", how)
    return how


def find_islands(a16, min_sil=0.32, pad=0.08):
    """Speech segments separated by pauses >= min_sil seconds."""
    x, sr = read_wav(a16)
    db = rms_db(x[:, 0], sr)
    floor = np.percentile(db, 10)
    loud = np.percentile(db, 95)
    thr = min(max(floor + 10, loud - 38, -55), -28)
    voiced = db > thr
    n = len(voiced)
    isl, i = [], 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n:
            if voiced[j]:
                j += 1
                continue
            k = j                         # silence run start
            while k < n and not voiced[k]:
                k += 1
            if (k - j) / 100 >= min_sil or k >= n:
                break
            j = k
        a, b = i / 100, j / 100
        if b - a >= 0.12:
            isl.append([round(max(0, a - pad), 3), round(min(n / 100, b + pad), 3)])
        i = j + 1
    return isl, float(thr)


class Engine:
    """mlx-whisper on Apple Silicon, faster-whisper everywhere else (NVIDIA GPU if present, else CPU)."""

    def __init__(self, model=None, engine=None):
        import platform
        self.kind = engine
        if self.kind is None:
            try:
                import mlx_whisper  # noqa: F401
                self.kind = "mlx" if platform.machine() == "arm64" and platform.system() == "Darwin" else "fw"
            except Exception:
                self.kind = "fw"
        if self.kind == "mlx":
            import mlx_whisper
            self.mlx = mlx_whisper
            self.model = model or "mlx-community/whisper-large-v3-turbo"
        else:
            from faster_whisper import WhisperModel
            dev, ct = "cpu", "int8"
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    dev, ct = "cuda", "float16"
            except Exception:
                pass
            self.model = model or ("large-v3-turbo" if dev == "cuda" else "medium")
            print(f"載入聽打模型 {self.model}（{dev}）…第一次會先下載，請稍等")
            self.fw = WhisperModel(self.model, device=dev, compute_type=ct)
        print(f"聽打引擎：{self.kind} / {self.model}")

    def words(self, audio, prompt, lang):
        """audio: float32 mono 16 kHz. Returns [(word, t0, t1)] in seconds from the start of `audio`."""
        out = []
        if self.kind == "mlx":
            r = self.mlx.transcribe(audio, path_or_hf_repo=self.model, language=lang, word_timestamps=True,
                                    initial_prompt=prompt or None, condition_on_previous_text=False,
                                    verbose=None)
            for s in r.get("segments", []):
                for w in s.get("words", []):
                    out.append((w["word"], float(w["start"]), float(w["end"])))
        else:
            segs, _ = self.fw.transcribe(audio, language=lang, word_timestamps=True, initial_prompt=prompt or None,
                                         condition_on_previous_text=False, vad_filter=False, beam_size=5)
            for s in segs:
                for w in (s.words or []):
                    out.append((w.word, float(w.start), float(w.end)))
        return out


def main(d, model=None, engine=None, lang="zh"):
    d = Path(d)
    proj = load(d, "project.json")
    src = proj["source"]
    if not Path(src).exists():
        die(f"找不到原片：{src}")
    info = probe(src)
    proj.update(info)
    print(f"原片：{Path(src).name}｜{info['width']}x{info['height']}｜{info['fps']:.3f} fps｜{info['duration'] / 60:.1f} 分鐘")
    proj["channel_note"] = extract_voice(d, src, info)
    isl, thr = find_islands(d / "a16.wav")
    proj["speech_threshold_db"] = thr
    save(d, "project.json", proj)
    print(f"找到 {len(isl)} 段有聲音的片段，開始逐段聽打…")

    x, sr = read_wav(d / "a16.wav")
    x = x[:, 0]
    eng = Engine(model, engine)
    # "以下是繁體中文" nudges the model toward traditional characters; the user's vocabulary helps names
    prompt = "以下是繁體中文的句子。" + (proj.get("vocab") or "")
    out, t_start = [], time.time()
    for i, (a, b) in enumerate(isl):
        seg = x[int(a * sr):int(b * sr)]
        ws = [] if b - a < 0.2 else eng.words(np.ascontiguousarray(seg), prompt, lang)
        if looks_made_up(ws, b - a):
            # short noisy bits (a laugh, a breath) make the model loop or invent text: retry once without the prompt
            ws = eng.words(np.ascontiguousarray(seg), None, lang)
        unclear = looks_made_up(ws, b - a)
        if unclear:
            ws = []
        words = []
        for w, t0, t1 in ws:
            sp = w.startswith(" ")                     # a real space before an English word
            w = to_trad(w).strip()
            if not w:
                continue
            words.append({"w": w, "t0": round(a + t0, 3), "t1": round(a + min(t1, b - a), 3), "sp": sp})
        text = "".join(w["w"] for w in words) if lang.startswith("zh") else " ".join(w["w"] for w in words)
        if HALLU.search(text) or (len(norm(text)) <= 1 and b - a < 0.6 and not words):
            words, text = [], ""
        item = {"id": i, "a": a, "b": b, "text": text, "words": words}
        if unclear:
            item["unclear"] = True
        out.append(item)
        if i % 20 == 0 or i == len(isl) - 1:
            el = time.time() - t_start
            print(f"  {i + 1}/{len(isl)}（{el:.0f} 秒）")
    save(d, "transcript.json", {"islands": out})
    lines = ["# 逐字稿（每一行是一段有聲音的片段）", "",
             "格式：[編號] 開始秒數–結束秒數  內容。剪片時用編號決定留哪些。", ""]
    for s in out:
        empty = "（聽不清楚：可能是笑聲、雜音，或講太快）" if s.get("unclear") else "（沒有字：換氣、雜音或嗯啊）"
        lines.append(f"[{s['id']:03d}] {s['a']:7.2f}–{s['b']:7.2f}  {s['text'] or empty}")
    (d / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"聽打完成：transcript.md（{len(out)} 段）")
