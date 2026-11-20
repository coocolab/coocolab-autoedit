"""Unit tests for the pure parts (no video, no speech model). Run:  python -m unittest discover -s tests -v
(pytest works too). Needs the packages from requirements.txt (numpy, Pillow)."""
import hashlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "install"))

import numpy as np  # noqa: E402

from autoedit import common, cut, download, finish, suggest, transcribe  # noqa: E402
from autoedit import __main__ as cli  # noqa: E402


def words(text, t0=0.0, step=0.25):
    return [{"w": ch, "t0": round(t0 + i * step, 3), "t1": round(t0 + (i + 1) * step, 3), "sp": False}
            for i, ch in enumerate(text)]


def island(i, a, text, step=0.25):
    ws = words(text, a, step)
    return {"id": i, "a": a, "b": round(a + max(len(text), 1) * step, 3), "text": text, "words": ws}


class TempDir(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.d = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------- probe parsing (M1, M2, R4)

class TestProbe(unittest.TestCase):
    def j(self, **v):
        video = {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "30/1",
                 "avg_frame_rate": "30/1", **v}
        return {"streams": [video, {"codec_type": "audio", "channels": 2}], "format": {"duration": "10.0"}}

    def test_rotation_side_data_swaps_size(self):
        info = common.parse_probe(self.j(side_data_list=[{"rotation": -90}]))
        self.assertEqual((info["width"], info["height"], info["rotation"]), (1080, 1920, 270))

    def test_rotation_old_tag(self):
        info = common.parse_probe(self.j(tags={"rotate": "90"}))
        self.assertEqual((info["width"], info["height"]), (1080, 1920))

    def test_rotation_180_keeps_size(self):
        info = common.parse_probe(self.j(side_data_list=[{"rotation": 180}]))
        self.assertEqual((info["width"], info["height"]), (1920, 1080))

    def test_vfr_detected(self):
        self.assertTrue(common.parse_probe(self.j(r_frame_rate="60/1", avg_frame_rate="6190/221"))["vfr"])
        self.assertFalse(common.parse_probe(self.j(r_frame_rate="30000/1001", avg_frame_rate="2997/100"))["vfr"])

    def test_hdr_detected(self):
        self.assertTrue(common.parse_probe(self.j(color_transfer="arib-std-b67"))["hdr"])
        self.assertTrue(common.parse_probe(self.j(color_transfer="smpte2084"))["hdr"])
        self.assertFalse(common.parse_probe(self.j(color_transfer="bt709"))["hdr"])

    def test_mkv_duration_tag(self):
        info = common.parse_probe(self.j(tags={"DURATION": "00:00:08.500000000"}))
        self.assertAlmostEqual(info["v_duration"], 8.5)

    def test_cfr_rate(self):
        self.assertEqual(common.cfr_rate((2997, 100)), (30000, 1001))
        self.assertEqual(common.cfr_rate((2950, 100)), (30000, 1001))
        self.assertEqual(common.cfr_rate((12, 1)), (24, 1))
        self.assertEqual(common.cfr_rate((43, 1)), (43, 1))


# ---------------------------------------------------------------- json files (M7)

class TestSaveLoad(TempDir):
    def test_atomic_save_no_tmp_left(self):
        common.save(self.d, "x.json", {"a": "中文"})
        self.assertEqual(common.load(self.d, "x.json"), {"a": "中文"})
        self.assertEqual([p.name for p in self.d.iterdir()], ["x.json"])

    def test_broken_json_is_plain_error(self):
        (self.d / "cuts.json").write_text('{"keep": [', encoding="utf-8")
        with self.assertRaises(SystemExit):
            common.load(self.d, "cuts.json")


# ---------------------------------------------------------------- downloads (M3, N7)

class TestDownload(TempDir):
    def setUp(self):
        super().setUp()
        self.src = self.d / "src.bin"
        self.src.write_bytes(os.urandom(300_000))
        self.sha = hashlib.sha256(self.src.read_bytes()).hexdigest()
        self.url = self.src.as_uri()
        self.dest = self.d / "out" / "model.bin"

    def test_fresh_download(self):
        download.fetch(self.url, self.dest, self.sha, "測試檔")
        self.assertTrue(download.is_ok(self.dest, self.sha))
        self.assertFalse(self.dest.with_name("model.bin.part").exists())

    def test_truncated_file_is_downloaded_again(self):
        self.dest.parent.mkdir()
        self.dest.write_bytes(self.src.read_bytes()[:1000])          # what an interrupted old version left
        download.fetch(self.url, self.dest, self.sha, "測試檔")
        self.assertEqual(self.dest.stat().st_size, 300_000)

    def test_wrong_hash_keeps_nothing(self):
        with self.assertRaises(download.DownloadError):
            download.fetch(self.url, self.dest, "0" * 64, "測試檔")
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.dest.with_name("model.bin.part").exists())

    def test_network_error_is_plain(self):
        with self.assertRaises(download.DownloadError) as cm:
            download.fetch((self.d / "missing.bin").as_uri(), self.dest, self.sha, "測試檔")
        self.assertIn("下載失敗", str(cm.exception))
        self.assertFalse(self.dest.exists())


# ---------------------------------------------------------------- suggest / cuts.json (M10a, R7)

class TestSuggest(TempDir):
    def test_inner_repeat(self):
        ws = [{"w": w} for w in ["最狠", "的", "是", "最狠", "的", "是", "還能", "設定"]]
        w0, w1, phrase = suggest.inner_repeat(ws)
        self.assertEqual((w0, w1), (0, 3))
        self.assertEqual(phrase, "最狠的是")

    def test_inner_repeat_none(self):
        self.assertIsNone(suggest.inner_repeat([{"w": w} for w in ["今天", "天氣", "很好"]]))

    def make(self, islands):
        transcribe.write_transcript(self.d, islands)

    def test_draft_marks_restart_and_filler(self):
        self.make([island(0, 0.0, "今天要教你"), island(1, 2.0, "今天要教你用AI剪片"), island(2, 5.0, "嗯"),
                   island(3, 6.0, "這個工具很好用")])
        suggest.draft(self.d)
        cuts = json.loads((self.d / "cuts.json").read_text(encoding="utf-8"))
        kept = [k["id"] for k in cuts["keep"]]
        self.assertNotIn(0, kept)
        self.assertNotIn(2, kept)
        self.assertIn(1, kept)
        self.assertEqual(cuts["transcript_fp"], json.loads((self.d / "transcript.json").read_text("utf-8"))["fp"])
        self.assertTrue((self.d / "剪掉清單.md").exists())

    def test_stale_cuts_after_new_transcript(self):
        self.make([island(0, 0.0, "第一句"), island(1, 2.0, "第二句")])
        suggest.draft(self.d)
        self.make([island(0, 0.0, "第一句"), island(1, 1.0, "插進來的"), island(2, 2.0, "第二句")])
        with self.assertRaises(SystemExit):
            suggest.load_cuts(self.d)

    def test_stale_cuts_without_fingerprint_uses_time(self):
        self.make([island(0, 0.0, "第一句")])
        (self.d / "cuts.json").write_text('{"keep": [{"id": 0}]}', encoding="utf-8")
        old = time.time() - 100
        os.utime(self.d / "cuts.json", (old, old))
        with self.assertRaises(SystemExit):
            suggest.load_cuts(self.d)

    def test_bad_cuts_are_plain_errors(self):
        self.make([island(0, 0.0, "第一句話"), island(1, 2.0, "第二句")])
        fp = json.loads((self.d / "transcript.json").read_text("utf-8"))["fp"]
        for bad in ([{"id": 7}], [{"id": "0"}], [{"id": 0, "from": 3, "to": 2}], [{"id": 0, "to": 99}], ["0"]):
            (self.d / "cuts.json").write_text(json.dumps({"keep": bad, "transcript_fp": fp}), encoding="utf-8")
            with self.assertRaises(SystemExit, msg=str(bad)):
                suggest.load_cuts(self.d)
        (self.d / "cuts.json").write_text(json.dumps({"keep": [{"id": 0, "from": 1, "to": 3}], "transcript_fp": fp}),
                                          encoding="utf-8")
        isl, cuts = suggest.load_cuts(self.d)
        self.assertEqual(len(isl), 2)


# ---------------------------------------------------------------- cut: time mapping

class TestBuildSpans(TempDir):
    def test_spans_follow_kept_words(self):
        isl = [island(0, 1.0, "大家好"), island(1, 3.0, "重講重講"), island(2, 5.0, "這是第三段")]
        transcribe.write_transcript(self.d, isl)
        fp = json.loads((self.d / "transcript.json").read_text("utf-8"))["fp"]
        common.save(self.d, "cuts.json", {"keep": [{"id": 0}, {"id": 2, "from": 2, "to": 5}], "transcript_fp": fp})
        common.save(self.d, "project.json", {"speech_threshold_db": -40})
        sr = common.SR
        x = np.zeros(int(sr * 8), np.float32)
        for s in isl:                                   # a tone wherever someone speaks
            n0, n1 = int(s["a"] * sr), int(s["b"] * sr)
            x[n0:n1] = 0.3 * np.sin(np.arange(n1 - n0) / sr * 2 * np.pi * 220)
        common.write_wav(self.d / "a16.wav", x, sr)
        _, _, spans = cut.build_spans(self.d)
        self.assertEqual([s[2] for s in spans], [0, 2])
        self.assertAlmostEqual(spans[0][0], 1.0, places=2)
        self.assertGreaterEqual(spans[1][0], 5.0 + 2 * 0.25 - 0.13)    # starts near word 2 of island 2
        self.assertAlmostEqual(spans[1][1], isl[2]["b"], places=2)


# ---------------------------------------------------------------- subtitles

class TestSubtitles(unittest.TestCase):
    def line_text(self, ln):
        return "".join(c[0] for c in ln)

    def test_spaced(self):
        self.assertEqual(finish.spaced("用Claude說100次"), "用 Claude 說 100 次")

    def test_glue_english_pieces(self):
        g = finish.glue([{"w": "Many", "t0": 0, "t1": 0.2, "sp": True}, {"w": "Chat", "t0": 0.2, "t1": 0.4, "sp": False},
                         {"w": "好", "t0": 0.4, "t1": 0.6}])
        self.assertEqual([w["w"] for w in g], ["ManyChat", "好"])

    def test_apply_fixes_and_manual_break(self):
        ws = [{**w, "island": 0} for w in words("他就會自動幫你剪片", 0, 0.2)]
        lines = finish.build_lines(ws, False, {"他就會自動": "它就會自動｜"})
        self.assertEqual([self.line_text(ln) for ln in lines], ["它就會自動", "幫你剪片"])

    def test_long_line_is_split_at_pause(self):
        ws = [{**w, "island": 0} for w in words("這是一個非常非常長的句子", 0, 0.2)]
        ws += [{**w, "island": 0} for w in words("後面接著講完", 2.6, 0.2)]   # 0.2 s pause, same island
        lines = finish.build_lines(ws, True, {})
        self.assertGreater(len(lines), 1)
        for ln in lines:
            self.assertLessEqual(finish._width(self.line_text(ln)), 13)

    def test_split_never_inside_number(self):
        chars = [[ch, i * 0.1, (i + 1) * 0.1, 0, i] for i, ch in enumerate("一個月要花900多美元的帳單喔")]
        for part in finish._split(chars, 6):
            t = self.line_text(part)
            self.assertFalse(t.endswith("9") or t.endswith("90"), t)

    def test_place_cards(self):
        ws = [{**w, "island": 0} for w in words("前兩個帳號免費", 1.0, 0.2)]
        lines = finish.build_lines(ws, False, {})
        placed, missing = finish.place_cards([{"at": "帳號免費", "text": "免費"}, {"at": "沒有這句"}], lines, 1.0)
        self.assertEqual(missing, ["沒有這句"])
        self.assertAlmostEqual(placed[0]["t0"], 1.6, places=2)


# ---------------------------------------------------------------- command line (R7, R9)

class TestCli(TempDir):
    def test_safe_name(self):
        self.assertEqual(cli.safe_name("../../etc"), "_.._etc")
        self.assertNotIn("/", cli.safe_name("a/b\\c"))
        self.assertEqual(cli.safe_name(".."), "影片")

    def test_same_stem_different_source_gets_own_folder(self):
        (self.d / "原片").mkdir()
        for n in ("a.mp4", "a.mov"):
            (self.d / "原片" / n).write_bytes(b"x")
        cli.init(self.d / "原片" / "a.mp4")
        cli.init(self.d / "原片" / "a.mov")
        cli.init(self.d / "原片" / "a.mp4")            # running init again reuses its own folder
        self.assertEqual(sorted(p.name for p in (self.d / "工作區").iterdir()), ["a", "a_2"])

    def test_missing_argument_is_plain(self):
        with mock.patch.object(sys, "argv", ["ae", "cut"]), self.assertRaises(SystemExit) as cm:
            cli.main()
        self.assertEqual(cm.exception.code, 1)

    def test_unexpected_error_has_no_traceback(self):
        err = []
        with mock.patch.object(cli, "run_step", side_effect=KeyError("from")), \
                mock.patch.object(sys, "argv", ["ae", "cut", str(self.d)]), \
                mock.patch.object(Path, "cwd", return_value=self.d), \
                mock.patch("sys.stderr", new=types.SimpleNamespace(write=err.append, flush=lambda: None,
                                                                  reconfigure=lambda **k: None)):
            with self.assertRaises(SystemExit):
                cli.main()
        text = "".join(err)
        self.assertIn("「cut」這一步沒做完", text)
        self.assertNotIn("Traceback", text)
        self.assertTrue((self.d / "autoedit-error.log").exists())


# ---------------------------------------------------------------- speech engine fallback (M4)

class TestEngineFallback(unittest.TestCase):
    def test_cuda_failure_falls_back_to_cpu(self):
        made = []

        class FakeModel:
            def __init__(self, name, device, compute_type):
                made.append((name, device, compute_type))
                self.device = device

            def transcribe(self, audio, **kw):
                if self.device == "cuda":
                    raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                return iter([]), None

        fw = types.ModuleType("faster_whisper")
        fw.WhisperModel = FakeModel
        ct2 = types.ModuleType("ctranslate2")
        ct2.get_cuda_device_count = lambda: 1
        with mock.patch.dict(sys.modules, {"faster_whisper": fw, "ctranslate2": ct2}):
            eng = transcribe.Engine(None, "fw")
        self.assertEqual(made[0][1], "cuda")
        self.assertEqual(made[-1], ("medium", "cpu", "int8"))
        self.assertEqual(eng.model, "medium")


# ---------------------------------------------------------------- effects bookkeeping (M10b)

class TestFxBookkeeping(TempDir):
    def test_empty_fx_json_drops_old_fx_video(self):
        from autoedit import fx
        common.save(self.d, "project.json", {"video_for_finish": "fx_video.mp4", "out_w": 640, "out_h": 360})
        common.save(self.d, "timeline.json", {"duration": 1, "frames": 30, "words": [], "spans": []})
        common.save(self.d, "fx.json", [])
        fx.main(self.d)
        self.assertNotIn("video_for_finish", common.load(self.d, "project.json"))


# ---------------------------------------------------------------- installer (M8, R1, R10)

class TestSetup(TempDir):
    def setUp(self):
        super().setUp()
        import setup
        self.setup = setup

    def test_prefers_312(self):
        found = {("python3.13",): (3, 13), ("python3.12",): (3, 12), ("python3.10",): (3, 10)}
        with mock.patch.object(self.setup, "py_version", side_effect=lambda c: found.get(tuple(c))), \
                mock.patch.object(self.setup.shutil, "which", side_effect=lambda c: c if (c,) in found else None):
            py, v = self.setup.find_python()
        self.assertEqual(v, (3, 12))

    def test_onedrive_folder_is_moved_home(self):
        with mock.patch.object(self.setup, "mac_icloud_desktop", return_value=False):
            folder, why, note = self.setup.choose_folder(r"C:\Users\me\OneDrive\Desktop\AI剪片")
            self.assertEqual(folder, Path.home() / "AI剪片")
            self.assertIn("OneDrive", why)
            folder, why, _ = self.setup.choose_folder(r"C:\Users\me\OneDrive\Desktop\AI剪片", allow_cloud=True)
            self.assertIsNone(why)
            folder, why, _ = self.setup.choose_folder(str(self.d / "AI剪片"))
            self.assertEqual(folder, self.d / "AI剪片")

    def test_icloud_desktop_is_moved_home(self):
        with mock.patch.object(self.setup, "mac_icloud_desktop", return_value=True):
            folder, why, _ = self.setup.choose_folder(str(Path.home() / "Desktop" / "AI剪片"))
        self.assertEqual(folder, Path.home() / "AI剪片")
        self.assertIn("iCloud", why)

    def test_agent_files_backed_up_when_changed(self):
        (self.d / "CLAUDE.md").write_text("我自己加的規矩", encoding="utf-8")
        self.setup.write_agent_files(self.d, "新版")
        self.assertEqual((self.d / "CLAUDE.md").read_text(encoding="utf-8"), "新版")
        baks = list(self.d.glob("CLAUDE.md.bak-*"))
        self.assertEqual(len(baks), 1)
        self.assertEqual(baks[0].read_text(encoding="utf-8"), "我自己加的規矩")
        self.assertEqual((self.d / "AGENTS.md").read_text(encoding="utf-8"), "新版")

    def test_template_placeholders_all_filled(self):
        tpl = (REPO / "agent" / "AGENTS.template.md").read_text(encoding="utf-8")
        text = tpl.replace("{{RUN}}", "x").replace("{{REPO}}", "y").replace("{{RUN_POWERSHELL}}", "")
        self.assertNotIn("{{", text)


if __name__ == "__main__":
    unittest.main()
