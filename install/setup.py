"""Installer / environment check. Uses only the Python standard library, so it runs before anything is installed.

  python install/setup.py check                 看電腦環境，缺什麼會列出來（什麼都不裝）
  python install/setup.py install [--folder 路徑] 建虛擬環境、裝套件、下載字型、建「AI剪片」資料夾

ffmpeg 和 Python 這類系統軟體不會自動裝，只會告訴你要執行哪一行。
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV = REPO / ".venv"
FONT_URL = "https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/TC/NotoSansTC-Bold.otf"
FONT = REPO / "assets" / "fonts" / "NotoSansTC-Bold.otf"
OK_PY = [(3, 10), (3, 11), (3, 12), (3, 13)]
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"


def venv_python():
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def py_version(exe):
    try:
        r = subprocess.run([*exe, "-c", "import sys;print(sys.version_info[0],sys.version_info[1])"],
                           capture_output=True, text=True, timeout=20)
        a, b = r.stdout.split()
        return int(a), int(b)
    except Exception:
        return None


def find_python():
    """A Python 3.10-3.13 (3.14 is too new for some speech packages)."""
    cands = [[sys.executable]]
    if IS_WIN:
        cands += [["py", f"-{a}.{b}"] for a, b in reversed(OK_PY)]
    cands += [[f"python{a}.{b}"] for a, b in reversed(OK_PY)] + [["python3"], ["python"]]
    for c in cands:
        if c[0] != sys.executable and not shutil.which(c[0]):
            continue
        v = py_version(c)
        if v in OK_PY:
            return c, v
    return None, None


def gpu():
    if IS_MAC and platform.machine() == "arm64":
        return "Apple Silicon（用 mlx 加速聽打）"
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return "NVIDIA " + r.stdout.strip().splitlines()[0]
    return "沒有可用的顯示卡加速（用 CPU 聽打，會比較慢，但可以用）"


def ffmpeg_hint():
    if IS_MAC:
        return "brew install ffmpeg（沒有 Homebrew 的話先裝：https://brew.sh）"
    if IS_WIN:
        return "winget install --id Gyan.FFmpeg -e（裝完要關掉終端機再重開）"
    return "sudo apt install ffmpeg"


def python_hint():
    if IS_MAC:
        return "brew install python@3.12"
    if IS_WIN:
        return "winget install --id Python.Python.3.12 -e"
    return "sudo apt install python3.12 python3.12-venv"


def check(verbose=True):
    rep = {"系統": f"{platform.system()} {platform.release()}（{platform.machine()}）"}
    py, v = find_python()
    rep["Python"] = f"{v[0]}.{v[1]}（{' '.join(py)}）" if py else f"缺少：需要 3.10～3.13 → {python_hint()}"
    rep["ffmpeg"] = shutil.which("ffmpeg") or f"缺少 → {ffmpeg_hint()}"
    rep["ffprobe"] = shutil.which("ffprobe") or "缺少（跟 ffmpeg 一起裝）"
    rep["加速"] = gpu()
    free = shutil.disk_usage(str(Path.home())).free / 1e9
    rep["硬碟剩餘"] = f"{free:.0f} GB" + ("（建議至少 10 GB：模型約 1.5 GB，剪片暫存檔約影片的 2～3 倍）" if free < 10 else "")
    rep["剪片工具環境"] = "已建立" if venv_python().exists() else "還沒建立（install 會建）"
    rep["字型"] = "已下載" if FONT.exists() else "還沒下載（install 會下載 Noto Sans TC）"
    missing = [k for k, val in rep.items() if str(val).startswith("缺少")]
    if verbose:
        print("── 環境檢查 ──")
        for k, val in rep.items():
            print(f"{k}：{val}")
        print("結果：", "可以安裝" if not missing else f"還缺 {'、'.join(missing)}，請先裝好再執行 install")
    return rep, missing, py


def default_folder():
    desk = Path.home() / "Desktop"
    return (desk if desk.exists() else Path.home()) / "AI剪片"


def install(folder):
    rep, missing, py = check()
    if missing:
        sys.exit(1)
    if not venv_python().exists():
        print("\n建立剪片工具專用的 Python 環境…")
        subprocess.run([*py, "-m", "venv", str(VENV)], check=True)
    vp = str(venv_python())
    print("安裝套件（第一次約 1～3 分鐘）…")
    subprocess.run([vp, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([vp, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")], check=True)
    if not FONT.exists():
        print("下載字型 Noto Sans TC Bold（開源字型，約 6 MB）…")
        FONT.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(FONT_URL, FONT)
        except Exception as e:
            print(f"[注意] 字型下載失敗（{e}），會改用電腦內建的中文字型。")

    folder = Path(folder).expanduser() if folder else default_folder()
    for sub in ("原片", "成片", "工作區"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    run = f'"{vp}" "{REPO / "ae.py"}"'
    tpl = (REPO / "agent" / "AGENTS.template.md").read_text(encoding="utf-8")
    text = tpl.replace("{{RUN}}", run).replace("{{REPO}}", str(REPO)).replace("{{FOLDER}}", str(folder))
    for name in ("CLAUDE.md", "AGENTS.md"):           # Claude Code reads CLAUDE.md, Codex reads AGENTS.md
        (folder / name).write_text(text, encoding="utf-8")
    (REPO / "install" / "installed.json").write_text(json.dumps(
        {"folder": str(folder), "python": vp, "run": run}, ensure_ascii=False, indent=1), encoding="utf-8")

    # quick self test: the package imports and finds a font
    r = subprocess.run([vp, "-c", "import sys;sys.path.insert(0,r'%s');from autoedit.common import font_path;"
                        "print(font_path())" % REPO], capture_output=True, text=True)
    print("\n── 安裝完成 ──")
    print(f"剪片資料夾：{folder}")
    print(f"  原片／  ← 把要剪的影片放這裡")
    print(f"  成片／  ← 剪好的影片、字幕檔、封面會出現在這裡")
    print(f"  工作區／← 剪片過程的暫存檔，剪完可以刪")
    print(f"字型：{r.stdout.strip() or r.stderr.strip()[-200:]}")
    print("第一次聽打時會自動下載語音模型（約 1.5 GB），之後就不用再下載。")
    print("以後要剪片：在這個資料夾打開 Claude Code 或 Codex，說「幫我剪這支」。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "install"])
    ap.add_argument("--folder")
    a = ap.parse_args()
    check() if a.cmd == "check" else install(a.folder)
