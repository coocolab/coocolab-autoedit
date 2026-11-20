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
"""
import argparse
import re
import sys
from pathlib import Path

from .common import die, load, save


def init(video, name=None):
    v = Path(video).expanduser().absolute()
    if not v.exists():
        die(f"找不到影片：{v}")
    root = v.parent.parent if v.parent.name == "原片" else Path.cwd()
    name = name or re.sub(r"[\\/:*?\"<>|]", "_", v.stem)
    wd = root / "工作區" / name
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


def main():
    p = argparse.ArgumentParser(prog="autoedit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step")
    p.add_argument("args", nargs="*")
    p.add_argument("--name")
    p.add_argument("--model")
    p.add_argument("--engine", choices=["mlx", "fw"])
    p.add_argument("--lang", default="zh")
    a = p.parse_args()
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
        isl = load(x[0], "transcript.json")["islands"][int(x[1])]
        for k, w in enumerate(isl["words"]):
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
        make(x[0], x[1], x[2], x[3] if len(x) > 3 else "")
    elif s == "qa":
        from .finish import qa
        qa(x[0])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
