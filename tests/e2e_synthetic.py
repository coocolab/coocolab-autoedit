"""End-to-end run on small videos made with ffmpeg, without the speech model (CI runs this on Windows, macOS, Linux).

  <.venv python> tests/e2e_synthetic.py [work folder]

Makes three clips (normal, phone portrait with a rotate-90 flag, variable frame rate), with tone bursts as "speech"
and names with spaces and Chinese. For each one: init -> prepare (probe, work copy, voice, speech islands) -> a
made-up transcript -> suggest -> cut -> subs -> fx (if MediaPipe works) -> finish -> cover -> qa, through ae.py
like the AI tools do. Checks: picture and sound end within 0.05 s, the portrait clip stays portrait, a cut.json
edit / fx.json removal is picked up on re-run, a silent track doesn't crash finish, and 原片/ is not changed.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from autoedit import common, transcribe  # noqa: E402

TEXTS = ["大家好今天要教你剪片", "這個工具這個工具很好用", "最重要的是它完全免費", "嗯", "字幕會一個一個跳出來", "我們下次見"]


def sh(*cmd, cwd=None, ok=True):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    out = (r.stdout + r.stderr).strip()
    print(f"$ {' '.join(map(str, cmd[2:] if cmd[0] == sys.executable else cmd))[:160]}\n{out[-1500:]}\n", flush=True)
    if ok and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode})")
    return r


def ae(root, *args, ok=True):
    return sh(sys.executable, REPO / "ae.py", *args, cwd=root, ok=ok)


def durations(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration,width,height",
                        "-of", "json", str(path)], capture_output=True, text=True)
    st = json.loads(r.stdout)["streams"]
    v = next(s for s in st if s["codec_type"] == "video")
    a = next(s for s in st if s["codec_type"] == "audio")
    return float(v["duration"]), float(a["duration"]), int(v["width"]), int(v["height"])


def make_clips(src):
    # 6 tone bursts of 1.2 s with 0.8 s pauses = "speech"
    expr = "0.4*sin(2*PI*220*t)*lt(mod(t\\,2)\\,1.2)"
    audio = ["-f", "lavfi", "-i", f"aevalsrc={expr}:s=48000:d=12"]
    clips = {
        "一般 橫式.mp4": ["-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=12", *audio, "-shortest"],
        "直拍 旋轉.mp4": None,
        "螢幕錄影 VFR.mp4": ["-f", "lavfi", "-i", "testsrc2=s=640x360:r=60:d=12", *audio, "-shortest",
                         "-vf", "select='lt(random(0)\\,0.45)+eq(n\\,0)'", "-fps_mode", "vfr"],
    }
    for name, args in clips.items():
        if args:
            sh("ffmpeg", "-v", "error", "-y", *args, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
               src / name)
    sh("ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", src / "一般 橫式.mp4", "-c", "copy",
       src / "直拍 旋轉.mp4")


def fake_transcript(wd):
    """What the speech model would give: the texts above, one per speech island, words spread evenly."""
    proj, isl = transcribe.prepare(wd)
    out = []
    for i, (a, b) in enumerate(isl):
        text = TEXTS[i % len(TEXTS)]
        step = (b - a) / len(text)
        ws = [{"w": ch, "t0": round(a + k * step, 3), "t1": round(a + (k + 1) * step, 3), "sp": False}
              for k, ch in enumerate(text)]
        out.append({"id": i, "a": a, "b": b, "text": text, "words": ws})
    transcribe.write_transcript(wd, out)
    return proj, out


def tree_hash(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def main(base):
    root = Path(base) / "AI剪片"
    if root.exists():
        shutil.rmtree(root)
    src = root / "原片"
    src.mkdir(parents=True)
    make_clips(src)
    before = tree_hash(src)
    from autoedit.fx import fx_available
    fx_ok = fx_available() is None
    print("動態特效可用：", fx_ok, fx_available() or "")

    for clip in sorted(src.iterdir()):
        name = clip.stem
        wd = root / "工作區" / name
        print(f"==================== {clip.name}")
        ae(root, "init", f"原片/{clip.name}")
        proj, isl = fake_transcript(wd)
        assert len(isl) >= 5, f"只找到 {len(isl)} 段聲音"
        ae(root, "suggest", f"工作區/{name}")
        ae(root, "cut", f"工作區/{name}")
        ae(root, "subs", f"工作區/{name}")
        common.save(wd, "cards.json", [{"at": "完全免費", "text": "0 元", "sub": "不用月費"}])
        if fx_ok:
            common.save(wd, "fx.json", [{"type": "word", "at": "大家好", "main": "你好", "small": "HELLO"},
                                        {"type": "stat", "at": "最重要的是", "title": "每月費用", "value": "0", "unit": "元"}])
            ae(root, "fx", f"工作區/{name}")
            assert common.load(wd, "project.json").get("video_for_finish") == "fx_video.mp4"
        ae(root, "finish", f"工作區/{name}")
        out = root / "成片" / f"{name}.mp4"
        v, a, w, h = durations(out)
        print(f"影像 {v:.3f} 秒／聲音 {a:.3f} 秒，{w}x{h}")
        assert abs(v - a) < 0.05, f"影音長度差 {abs(v - a):.3f} 秒"
        if "旋轉" in name:
            assert h > w, "直拍影片變成橫的了"
        if "VFR" in name:
            assert common.load(wd, "project.json").get("work_source"), "VFR 沒有轉工作檔"
        ae(root, "cover", f"工作區/{name}", "1.5", "測試封面", "副標")
        assert (root / "成片" / f"{name}_封面.jpg").exists()

    # re-run on the same clip: drop a kept island from cuts.json, remove fx.json
    name = "一般 橫式"
    wd = root / "工作區" / name
    cuts = common.load(wd, "cuts.json")
    first = cuts["keep"][0]["id"]
    cuts["keep"] = [k for k in cuts["keep"] if k["id"] != first]
    common.save(wd, "cuts.json", cuts)
    (wd / "fx.json").unlink(missing_ok=True)
    old_len = durations(root / "成片" / f"{name}.mp4")[0]
    ae(root, "list", f"工作區/{name}")
    ae(root, "cut", f"工作區/{name}")
    ae(root, "fx", f"工作區/{name}")
    assert "video_for_finish" not in common.load(wd, "project.json")
    ae(root, "finish", f"工作區/{name}")
    new_len, a, _, _ = durations(root / "成片" / f"{name}.mp4")
    print(f"重剪：{old_len:.2f} 秒 → {new_len:.2f} 秒")
    assert new_len < old_len - 0.5 and abs(new_len - a) < 0.05

    # a new transcript makes the old cuts.json invalid
    fake_transcript(wd)
    t = common.load(wd, "transcript.json")
    t["islands"][0]["text"] += "改"
    t["fp"] = transcribe.transcript_fp(t["islands"])
    common.save(wd, "transcript.json", t)
    r = ae(root, "cut", f"工作區/{name}", ok=False)
    assert r.returncode != 0 and "逐字稿重做過了" in r.stderr

    # silent sound: finish skips loudness instead of crashing
    ae(root, "suggest", f"工作區/{name}")
    ae(root, "cut", f"工作區/{name}")
    x, sr = common.read_wav(wd / "cut48.wav")
    common.write_wav(wd / "cut48.wav", x * 0, sr)
    r = ae(root, "finish", f"工作區/{name}")
    assert "量不到音量" in r.stdout

    # plain errors, no tracebacks
    r = ae(root, "words", f"工作區/{name}", "999", ok=False)
    assert r.returncode == 1 and "Traceback" not in r.stderr
    r = ae(root, "cut", ok=False)
    assert r.returncode == 1 and "少給了參數" in r.stderr

    assert tree_hash(src) == before, "原片資料夾被改到了"
    print("\nE2E OK：", ", ".join(p.name for p in sorted((root / "成片").iterdir())))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        with tempfile.TemporaryDirectory() as td:
            main(td)
