"""Step 1: pull out the voice, split it into speech "islands" at pauses, transcribe each island on its own.

Why island by island: when the whole clip is transcribed in one go, the speech model tends to merge
"I said half a sentence, stopped, and said it again" into one clean sentence, so the retake becomes
invisible. Transcribing each island separately keeps both takes in the transcript.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from .common import (SR, SR_OUT, cfr_rate, die, ffmpeg, has_filter, load, norm, probe, read_wav, rms_db, save,
                     to_trad, write_wav)

PUSH_IN_MARGIN = 1.08    # keep this much extra resolution in a work copy, so the cut's push-in stays sharp

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


def _hf_cached(repo):
    """Is this Hugging Face model already downloaded (so we don't announce a 1.5 GB download that won't happen)?"""
    if os.path.isdir(repo):
        return True
    home = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")
    return os.path.isdir(os.path.join(home, "models--" + repo.replace("/", "--"), "snapshots"))


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
            if not _hf_cached(self.model):
                print("第一次聽打會先下載語音模型（約 1.5 GB），網路慢的話要等幾分鐘，下載完之後就不用再下載。")
        else:
            self._load_fw(model)
        print(f"聽打引擎：{self.kind} / {self.model}")

    def _load_fw(self, model):
        from faster_whisper import WhisperModel
        dev = "cpu"
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                dev = "cuda"
        except Exception:
            pass
        if dev == "cuda":
            # The NVIDIA driver alone makes the GPU visible, but faster-whisper also needs the cuBLAS / cuDNN
            # libraries, which most PCs don't have. Try one second of silence; on any error use the CPU instead.
            try:
                self.model = model or "large-v3-turbo"
                print(f"載入聽打模型 {self.model}（NVIDIA 顯示卡）…第一次會先下載約 1.5 GB，請稍等")
                self.fw = WhisperModel(self.model, device="cuda", compute_type="float16")
                list(self.fw.transcribe(np.zeros(SR, np.float32), language="zh")[0])
                return
            except Exception as e:  # noqa: BLE001
                print(f"顯示卡加速沒開成（{type(e).__name__}：{str(e)[:120]}），改用 CPU 聽打，會比較慢但結果一樣能用。")
                self.fw = None
        self.model = model or "medium"
        print(f"載入聽打模型 {self.model}（CPU）…第一次會先下載約 1.5 GB，請稍等")
        self.fw = WhisperModel(self.model, device="cpu", compute_type="int8")

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


def _source_stamp(src):
    st = Path(src).stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def work_copy(d, proj, info):
    """Clips that the cut can't handle as they are get one steady work copy (the original is never touched):
    - variable frame rate (screen recordings, QuickTime, some phones): picture and sound would drift apart;
    - HDR (iPhone records HLG by default): without tone mapping the result looks grey and washed out.
    Returns the info of the file the next steps should read."""
    src = proj["source"]
    if not (info["vfr"] or info["hdr"]):
        proj.pop("work_source", None)
        return info
    d = Path(d)
    out = (d / "source_work.mkv").absolute()
    stamp = _source_stamp(src)
    if out.exists() and proj.get("work_source_stamp") == stamp:
        print("沿用已經轉好的工作檔 source_work.mkv")
        proj["work_source"] = str(out)
        return {**probe(out), "vfr": False, "hdr": False}
    vf, notes = [], []
    if info["vfr"]:
        num, den = cfr_rate(info["avg_fps"] or (info["fps_num"], info["fps_den"]))
        notes.append(f"這支是「可變幀率」影片（螢幕錄影、QuickTime、部分手機常見），先轉成固定 {num / den:.3g} fps，剪出來嘴型才不會跑掉")
    else:
        num, den = info["fps_num"], info["fps_den"]
    # keep only the resolution the cut needs (short side = max_size x push-in), it makes the copy much faster
    cap = proj.get("max_size", 1080)
    short = min(info["width"], info["height"])
    target = int(cap * PUSH_IN_MARGIN) // 2 * 2 if cap else 0
    if target and short > target:
        if info["width"] <= info["height"]:
            vf.append(f"scale={target}:-2")
        else:
            vf.append(f"scale=-2:{target}")
    if info["hdr"]:
        if has_filter("zscale") and has_filter("tonemap"):
            vf.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=0,"
                      "zscale=t=bt709:m=bt709:r=tv")
            notes.append("這支是 HDR 影片（iPhone 預設會錄 HDR），先轉成一般亮度（SDR），成片才不會偏灰、偏淡")
        else:
            print("[注意] 這支是 HDR 影片，但你的 ffmpeg 沒有 zscale，沒辦法轉成一般亮度，成片顏色可能會偏灰。"
                  "以後錄影可以在相機設定把 HDR 影片關掉。")
    vf.append("format=yuv420p")
    for n in notes:
        print(n)
    print(f"轉工作檔中（原片不會動），大約要影片長度的 0.3～1 倍時間：{out.name}")
    tmp = d / "source_work.tmp.mkv"
    ffmpeg("-i", src, "-map", "0:v:0", "-map", "0:a:0", "-vf", ",".join(vf), "-fps_mode", "cfr", "-r", f"{num}/{den}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", 14, "-color_primaries", "bt709", "-color_trc", "bt709",
           "-colorspace", "bt709", "-c:a", "copy", tmp)
    os.replace(tmp, out)
    proj["work_source"] = str(out)
    proj["work_source_stamp"] = stamp
    return {**probe(out), "vfr": False, "hdr": False}


def transcript_fp(islands):
    """Fingerprint of a transcript: cuts.json made for another transcript must not be used (the ids changed)."""
    key = json.dumps([[s["id"], s["a"], s["b"], s["text"]] for s in islands], ensure_ascii=False)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def prepare(d):
    """Everything before the speech model: probe, work copy if needed, voice tracks, speech islands."""
    d = Path(d)
    proj = load(d, "project.json")
    src = proj["source"]
    if not Path(src).exists():
        die(f"找不到原片：{src}")
    info = probe(src)
    print(f"原片：{Path(src).name}｜{info['width']}x{info['height']}｜{info['fps']:.3f} fps｜{info['duration'] / 60:.1f} 分鐘"
          + (f"｜旋轉 {info['rotation']} 度" if info["rotation"] else ""))
    info = work_copy(d, proj, info)
    proj.update(info)
    proj["channel_note"] = extract_voice(d, proj.get("work_source", src), info)
    isl, thr = find_islands(d / "a16.wav")
    proj["speech_threshold_db"] = thr
    save(d, "project.json", proj)
    return proj, isl


def main(d, model=None, engine=None, lang="zh"):
    d = Path(d)
    proj, isl = prepare(d)
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
    write_transcript(d, out)


def write_transcript(d, out):
    d = Path(d)
    fp = transcript_fp(out)
    save(d, "transcript.json", {"islands": out, "fp": fp})
    lines = ["# 逐字稿（每一行是一段有聲音的片段）", "",
             "格式：[編號] 開始秒數–結束秒數  內容。剪片時用編號決定留哪些。", ""]
    for s in out:
        empty = "（聽不清楚：可能是笑聲、雜音，或講太快）" if s.get("unclear") else "（沒有字：換氣、雜音或嗯啊）"
        lines.append(f"[{s['id']:03d}] {s['a']:7.2f}–{s['b']:7.2f}  {s['text'] or empty}")
    (d / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"聽打完成：transcript.md（{len(out)} 段）")
    old = load(d, "cuts.json", {}) if (d / "cuts.json").exists() else {}
    if old and old.get("transcript_fp") != fp:
        print("[注意] 逐字稿重做過了，舊的 cuts.json 是照舊逐字稿寫的，段落編號已經不一樣。請重跑 suggest，重新確認剪掉清單。")
