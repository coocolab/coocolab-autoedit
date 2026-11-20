"""Command line. Run from anywhere:  python -m autoedit <step> ...

  init <影片檔> [--name 名稱]       建一個剪片工作區（在「工作區/<名稱>」）
  transcribe <工作區>              取出人聲、切段、逐段聽打 → transcript.md
  suggest <工作區>                 先猜一版要剪的（cuts.json）→ 剪掉清單.md
  list <工作區>                    cuts.json 改過之後，重新產生 剪掉清單.md
  words <工作區> <段落編號>         列出某一段每個字的編號和時間（要剪半段時用）
  cut <工作區>                     照 cuts.json 剪
  subs <工作區>                    產生字幕（subtitles.txt）並回報語速
  fx <工作區>                      照 fx.json 做動態特效（頭後字、光軌、便條紙…）
  gaps <工作區>                    列出畫面太久沒變化的地方
  finish <工作區>                  上字幕、字卡、加速、調音量 → 成片/
  cover <工作區> <秒數> <主標> [副標] 做封面
  qa <影片檔>                      檢查響度、左右聲道、影音長度

  --version                        顯示版本
"""
import argparse
import re
import sys
import traceback
from pathlib import Path

from . import __version__
from .common import die, load, save

# how many arguments each step needs (after the step name), and how to call it
USAGE = {"init": (1, "init <影片檔>"), "transcribe": (1, "transcribe <工作區>"), "suggest": (1, "suggest <工作區>"),
         "list": (1, "list <工作區>"), "words": (2, "words <工作區> <段落編號>"), "cut": (1, "cut <工作區>"),
         "subs": (1, "subs <工作區>"), "fx": (1, "fx <工作區>"), "gaps": (1, "gaps <工作區>"),
         "finish": (1, "finish <工作區>"), "cover": (3, "cover <工作區> <秒數> <主標> [副標]"), "qa": (1, "qa <影片檔>")}


def safe_name(s):
    """A folder name that cannot point outside 工作區/ ('..', slashes) and works on Windows."""
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", str(s)).strip().strip(".")
    return s or "影片"


def init(video, name=None):
    v = Path(video).expanduser().absolute()
    if not v.exists():
        die(f"找不到影片：{v}")
    if not v.is_file():
        die(f"{v} 是資料夾，不是影片檔。")
    root = v.parent.parent if v.parent.name == "原片" else Path.cwd()
    name = safe_name(name or v.stem)
    base, n = name, 2
    while True:
        wd = root / "工作區" / name
        old = load(wd, "project.json", {}) if (wd / "project.json").exists() else {}
        if not old or Path(old.get("source", "")) == v:
            break
        name = f"{base}_{n}"                     # 'a.mp4' and 'a.mov' must not share a work folder
        n += 1
    if name != base:
        print(f"「{base}」已經是另一支影片的工作區，這支改用「{name}」。")
    wd.mkdir(parents=True, exist_ok=True)
    proj = load(wd, "project.json", {})
    proj.update({"name": name, "source": str(v), "output_dir": str(root / "成片")})
    proj.setdefault("speed", 1.0)
    proj.setdefault("zoom_cuts", True)
    proj.setdefault("max_size", 1080)
    proj.setdefault("keywords", [])
    proj.setdefault("vocab", "")
    save(wd, "project.json", proj)
    print(f"工作區：{wd}")


def _plain_error(step, e):
    """One readable paragraph instead of a traceback; the full traceback goes to a log file for whoever fixes it."""
    tb = traceback.extract_tb(e.__traceback__)
    where = next((f"{Path(f.filename).name}:{f.lineno}" for f in reversed(tb) if "autoedit" in f.filename), "")
    log = Path.cwd() / "autoedit-error.log"
    try:
        log.write_text("".join(traceback.format_exception(e)), encoding="utf-8")
        logmsg = f"完整的技術訊息存在：{log}"
    except OSError:
        logmsg = ""
    if isinstance(e, MemoryError):
        why = "電腦記憶體不夠。先關掉其他程式再試；影片很長的話，可以先切成幾段再剪。"
    elif isinstance(e, (KeyError, IndexError, TypeError, ValueError)):
        why = "工作區裡的設定檔（例如 cuts.json、fx.json、cards.json）內容格式不對，或少了欄位。"
    elif isinstance(e, ModuleNotFoundError):
        why = f"缺少套件 {e.name}。請重新執行安裝（install/setup.py install）。"
    elif isinstance(e, OSError):
        why = "讀寫檔案時出錯：可能是檔案被別的程式開著、硬碟滿了，或路徑不存在。"
    else:
        why = "程式遇到沒預料到的狀況。"
    print(f"\n[錯誤] 「{step}」這一步沒做完。\n原因：{why}\n技術訊息：{type(e).__name__}: {e}"
          f"{f'（{where}）' if where else ''}\n{logmsg}\n請把這一段整段貼給 AI 助理，它會告訴你下一步。",
          file=sys.stderr)


def main():
    for stream in (sys.stdout, sys.stderr):      # Windows pipes default to cp950/cp1252: names and transcripts break
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="autoedit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", nargs="?")
    p.add_argument("args", nargs="*")
    p.add_argument("--name")
    p.add_argument("--model")
    p.add_argument("--engine", choices=["mlx", "fw"])
    p.add_argument("--lang", default="zh")
    p.add_argument("--version", action="version", version=f"Coocolab AutoEdit {__version__}")
    a = p.parse_args()
    if a.step not in USAGE:
        print(__doc__)
        sys.exit(1)
    need_n, usage = USAGE[a.step]
    if len(a.args) < need_n:
        die(f"少給了參數。用法：AE {usage}")
    try:
        run_step(a)
    except KeyboardInterrupt:
        print("\n[中斷] 這一步被停掉了，沒有做完。重跑同一個指令就好。", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # noqa: BLE001 - anything unexpected becomes a plain message
        _plain_error(a.step, e)
        sys.exit(1)


def run_step(a):
    s, x = a.step, a.args
    if s == "init":
        init(x[0], a.name)
    elif s == "transcribe":
        from .transcribe import main as m
        m(x[0], a.model, a.engine, a.lang)
    elif s == "suggest":
        from .suggest import draft
        draft(x[0])
    elif s == "list":
        from .suggest import write_list
        write_list(x[0])
    elif s == "words":
        isl = load(x[0], "transcript.json")["islands"]
        try:
            k = int(x[1])
            s_ = isl[k]
        except (ValueError, IndexError):
            die(f"段落編號要是 0～{len(isl) - 1} 的數字，你給的是「{x[1]}」。")
        for k, w in enumerate(s_["words"]):
            print(f"{k:3d}  {w['t0']:8.2f}–{w['t1']:8.2f}  {w['w']}")
    elif s == "cut":
        from .cut import main as m
        m(x[0])
    elif s == "subs":
        from .finish import subs
        subs(x[0])
    elif s == "fx":
        from .fx import main as m
        m(x[0])
    elif s == "gaps":
        from .fx import gaps
        gaps(x[0])
    elif s == "finish":
        from .finish import finish
        finish(x[0])
    elif s == "cover":
        from .cover import make
        try:
            float(x[1])
        except ValueError:
            die(f"封面的秒數要是數字（成片的第幾秒），你給的是「{x[1]}」。")
        make(x[0], x[1], x[2], x[3] if len(x) > 3 else "")
    elif s == "qa":
        from .finish import qa
        qa(x[0])


if __name__ == "__main__":
    main()
