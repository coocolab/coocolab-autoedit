"""Step 3: cut the video by cuts.json.

- Kept pieces are snapped to whole video frames. Each piece is encoded with an exact frame count
  (-frames:v), and the sound is cut sample-exact from the same frame times, so lips stay in sync no
  matter how many cuts there are.
- Long pauses inside a kept piece are shortened.
- Jump cuts alternate between normal framing and a slight push-in (1.08x), which hides the jump.
- Output: cut_video.mp4 (no sound), cut48.wav, timeline.json (every kept word on the new timeline).
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .common import SR_OUT, die, ffmpeg, load, need, read_wav, rms_db, run, save, write_wav

LONG_PAUSE = 0.45      # a silence inside a kept piece longer than this gets shortened...
PAUSE_KEEP = 0.14      # ...to this much on each side
JOIN_GAP_MAX = 0.40    # between two pieces that were next to each other: keep the natural pause up to this
PUSH_IN = 1.08


def _snap_quiet(env, t, lo, hi):
    """Quietest 10 ms point in [lo, hi] (seconds)."""
    a, b = int(max(lo, 0) * 100), int(hi * 100)
    if b <= a or a >= len(env):
        return t
    b = min(b, len(env))
    return (a + int(np.argmin(env[a:b]))) / 100


def build_spans(d):
    proj = load(d, "project.json")
    isl = load(d, "transcript.json")["islands"]
    cuts = load(d, "cuts.json")
    x, sr = read_wav(Path(d) / "a16.wav")
    env = rms_db(x[:, 0], sr)
    thr = proj.get("speech_threshold_db", -40)

    pieces = []    # (src_a, src_b, island_id, word_from, word_to)
    for k in sorted(cuts["keep"], key=lambda k: (isl[k["id"]]["a"], k.get("from", 0))):
        s = isl[k["id"]]
        nw = len(s["words"])
        f, t = k.get("from", 0), k.get("to", nw)
        if nw == 0:
            pieces.append((s["a"], s["b"], s["id"], 0, 0))
            continue
        if f >= t:
            continue
        a = s["a"] if f == 0 else _snap_quiet(env, 0, s["words"][f - 1]["t1"], s["words"][f]["t0"])
        if f > 0:
            a = max(a, s["words"][f]["t0"] - 0.12)
        if t == nw:
            b = s["b"]
        else:
            # the speech model often ends a word early; reach for the dip before the next word
            b = _snap_quiet(env, 0, s["words"][t - 1]["t1"], s["words"][t]["t0"])
        pieces.append((a, b, s["id"], f, t))

    # shorten long pauses inside each piece
    spans = []
    for a, b, iid, f, t in pieces:
        i0, i1 = int(a * 100), int(b * 100)
        quiet = env[i0:i1] < thr
        cur = a
        k = 0
        while k < len(quiet):
            if not quiet[k]:
                k += 1
                continue
            j = k
            while j < len(quiet) and quiet[j]:
                j += 1
            s0, s1 = a + k / 100, a + j / 100
            if s1 - s0 > LONG_PAUSE and s0 > a + 0.05 and s1 < b - 0.05:
                spans.append([cur, s0 + PAUSE_KEEP, iid])
                cur = s1 - PAUSE_KEEP
            k = j
        spans.append([cur, b, iid])

    # natural pause between neighbouring islands that are both kept (nothing cut in between)
    for p, q in zip(spans, spans[1:]):
        if q[2] == p[2] + 1:
            gap = q[0] - p[1]
            if gap > 0:
                extra = min(gap, JOIN_GAP_MAX - 0.16)
                p[1] += extra * 0.6
                q[0] -= extra * 0.4
    return proj, isl, spans


def main(d, workers=3):
    d = Path(d)
    proj, isl, spans = build_spans(d)
    fps = proj["fps"]
    off = proj.get("v_start", 0) - proj.get("a_start", 0)     # video frame k plays at audio time off + k/fps
    # frames
    fr = []
    for a, b, iid in spans:
        fa, fb = round((a - off) * fps), round((b - off) * fps)
        fa = max(fa, 0)
        if fb - fa < 2:
            continue
        if fr and fa <= fr[-1][1]:                     # touching / overlapping: one continuous piece
            fr[-1][1] = max(fr[-1][1], fb)
            continue
        fr.append([fa, fb])
    if not fr:
        die("cuts.json 裡沒有要留的片段。")

    # output size
    W, H = proj["width"], proj["height"]
    cap = proj.get("max_size", 1080)                 # shorter side; 0 = keep original
    scale = 1.0 if not cap or min(W, H) <= cap else cap / min(W, H)
    ow, oh = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    proj.update({"out_w": ow, "out_h": oh})
    proj.pop("video_for_finish", None)          # effects must be redone after a new cut
    save(d, "project.json", proj)

    seg_dir = d / "segments"
    seg_dir.mkdir(exist_ok=True)
    for old in seg_dir.glob("*.mp4"):
        old.unlink()
    zoom_on = proj.get("zoom_cuts", True)
    jobs = []
    for k, (fa, fb) in enumerate(fr):
        z = PUSH_IN if (zoom_on and k % 2 == 1) else 1.0
        vf = []
        if z != 1.0:
            cw, ch = int(W / z) // 2 * 2, int(H / z) // 2 * 2
            vf.append(f"crop={cw}:{ch}:{(W - cw) // 2}:{(H - ch) // 3}")   # keep a bit more headroom
        vf += [f"scale={ow}:{oh}:flags=lanczos", "setsar=1", "format=yuv420p", "setpts=PTS-STARTPTS"]
        ss = proj.get("v_start", 0) - proj.get("fmt_start", 0) + (fa - 0.5) / fps
        out = seg_dir / f"{k:04d}.mp4"
        jobs.append((["-ss", f"{max(ss, 0):.6f}", "-i", proj["source"], "-frames:v", fb - fa, "-an",
                      "-vf", ",".join(vf), "-fps_mode", "passthrough",   # -r here would repeat the first frame
                      "-c:v", "libx264", "-preset", "veryfast", "-crf", 16, "-g", 60, out], fb - fa, out, z))

    print(f"剪成 {len(fr)} 段（原本就接在一起的段落會併成一段），開始輸出影像（{ow}x{oh}）…")
    done = [0]

    def work(job):
        args, n, out, z = job
        ffmpeg(*args)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  {done[0]}/{len(jobs)}")
        return out

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, jobs))
    lst = seg_dir / "list.txt"
    lst.write_text("".join(f"file '{j[2].name}'\n" for j in jobs), encoding="utf-8")
    ffmpeg("-f", "concat", "-safe", 0, "-i", lst, "-c", "copy", d / "cut_video.mp4")

    # sound: same frame times, sample exact, 6 ms fades at every join
    v48, sr = read_wav(d / "voice48.wav")
    v48 = v48[:, 0]
    parts, cum, fade = [], 0, int(0.006 * sr)
    words_out, spans_out = [], []
    for fa, fb in fr:
        n_out0 = round(cum / fps * sr)
        n_out1 = round((cum + fb - fa) / fps * sr)
        s0 = round((off + fa / fps) * sr)
        seg = np.zeros(n_out1 - n_out0, np.float32)
        src = v48[max(s0, 0):max(s0, 0) + len(seg)]
        seg[:len(src)] = src
        if len(seg) > 2 * fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
        parts.append(seg)
        a_src, b_src = off + fa / fps, off + fb / fps
        o0 = cum / fps
        spans_out.append({"src_a": round(a_src, 4), "src_b": round(b_src, 4), "out_a": round(o0, 4),
                          "out_b": round(o0 + (fb - fa) / fps, 4)})
        cum += fb - fa
    write_wav(d / "cut48.wav", np.concatenate(parts), sr)

    # every kept word on the new timeline (a word belongs to the piece that holds its middle)
    cuts = load(d, "cuts.json")
    keep_words = set()
    for k in cuts["keep"]:
        s = isl[k["id"]]
        for w in range(k.get("from", 0), k.get("to", len(s["words"]))):
            keep_words.add((k["id"], w))
    for s in isl:
        for wi, w in enumerate(s["words"]):
            if (s["id"], wi) not in keep_words:
                continue
            mid = (w["t0"] + w["t1"]) / 2
            sp = next((p for p in spans_out if p["src_a"] <= mid < p["src_b"]), None)
            if sp is None:
                continue
            t0 = max(sp["out_a"], sp["out_a"] + w["t0"] - sp["src_a"])
            t1 = min(sp["out_b"], sp["out_a"] + w["t1"] - sp["src_a"])
            words_out.append({"w": w["w"], "t0": round(t0, 3), "t1": round(max(t1, t0 + 0.04), 3),
                              "island": s["id"], "sp": w.get("sp", False)})
    dur = cum / fps
    save(d, "timeline.json", {"duration": round(dur, 4), "frames": cum, "spans": spans_out, "words": words_out})

    # check: picture and sound have the same length
    r = run([need("ffprobe"), "-v", "error", "-count_packets", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(d / "cut_video.mp4")])
    nv = int(r.stdout.strip() or 0)
    if nv != cum:
        print(f"[注意] 影像格數 {nv}，預期 {cum}")
    src_len = proj["duration"]
    print(f"剪好了：{src_len / 60:.1f} 分鐘 → {dur / 60:.1f} 分鐘（{len(fr)} 段，影像 {nv} 格，聲音 {len(np.concatenate(parts)) / sr:.2f} 秒）")
