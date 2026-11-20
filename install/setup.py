"""Installer / environment check. Uses only the Python standard library, so it runs before anything is installed.

  python install/setup.py check                 看電腦環境，缺什麼會列出來（什麼都不裝）
  python install/setup.py install [--folder 路徑] 建虛擬環境、裝套件、下載字型、建「AI剪片」資料夾

ffmpeg 和 Python 這類系統軟體不會自動裝，只會告訴你要執行哪一行。
"""
import argparse
import collections
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from autoedit import __version__  # noqa: E402  (standard library only)
from autoedit.download import DownloadError, fetch, is_ok  # noqa: E402
from autoedit.resources import (FONT, FONT_LICENSE, FONT_LICENSE_SHA256, FONT_LICENSE_URL,  # noqa: E402
                                FONT_SHA256, FONT_URL)

VENV = REPO / ".venv"
PREFERRED_PY = [(3, 12), (3, 11), (3, 13), (3, 10)]     # 3.12 first: every package (also MediaPipe) has wheels
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
FFMPEG_MIN = (5, 1)                                     # -fps_mode


def utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


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
    """The best Python on this computer for the tool: 3.12, then 3.11, 3.13, 3.10 (3.14 is too new for some
    speech packages; 3.13 has no MediaPipe on Intel Macs)."""
    cands = [[sys.executable]]
    if IS_WIN:
        cands += [["py", f"-{a}.{b}"] for a, b in PREFERRED_PY]
    cands += [[f"python{a}.{b}"] for a, b in PREFERRED_PY] + [["python3"], ["python"]]
    found = {}
    for c in cands:
        if c[0] != sys.executable and not shutil.which(c[0]):
            continue
        v = py_version(c)
        if v in PREFERRED_PY and v not in found:
            found[v] = c
    for v in PREFERRED_PY:
        if v in found:
            return found[v], v
    return None, None


def ffmpeg_version():
    exe = shutil.which("ffmpeg")
    if not exe:
        return None, None
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=20).stdout
    except Exception:
        return exe, None
    first = out.splitlines()[0] if out else ""
    m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", first)
    if m:
        return exe, (int(m.group(1)), int(m.group(2)))
    m = re.search(r"ffmpeg version (\d{4})-\d\d-\d\d", first)      # git snapshot builds name a date
    if m:
        return exe, (99, 0) if int(m.group(1)) >= 2023 else (0, 0)
    return exe, None


def gpu():
    if IS_MAC and platform.machine() == "arm64":
        return "Apple Silicon（用 mlx 加速聽打）"
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                return "NVIDIA " + r.stdout.strip().splitlines()[0] + "（加速沒開成的話會自動改用 CPU）"
        except Exception:
            pass
    return "沒有可用的顯示卡加速（用 CPU 聽打，會比較慢，但可以用）"


def ffmpeg_hint():
    if IS_MAC:
        return "brew install ffmpeg（沒有 Homebrew 的話先裝：https://brew.sh）"
    if IS_WIN:
        return "winget install --id Gyan.FFmpeg -e（裝完要把 Claude Code／Codex 整個關掉再重開）"
    return ("Ubuntu 24.04 以上：sudo apt install ffmpeg；Ubuntu 22.04 的 apt 版本太舊（4.4），"
            "請改用 sudo snap install ffmpeg，或到 https://johnvansickle.com/ffmpeg/ 下載靜態版")


def python_hint():
    if IS_MAC:
        return "brew install python@3.12"
    if IS_WIN:
        return "winget install --id Python.Python.3.12 -e"
    return "sudo apt install python3 python3-venv（Ubuntu 22.04 會裝到 3.10、24.04 會裝到 3.12，都可以用）"


# ---------------------------------------------------------------- where the AI剪片 folder goes

def _windows_desktop():
    """The Desktop Windows really shows (OneDrive often moves it to C:\\Users\\<me>\\OneDrive\\Desktop)."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders")
        val, _ = winreg.QueryValueEx(key, "Desktop")
        return Path(os.path.expandvars(val))
    except Exception:
        return Path.home() / "Desktop"


def mac_icloud_desktop():
    """macOS 'Desktop & Documents in iCloud' is on: the Desktop is synced to iCloud."""
    return IS_MAC and (Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Desktop").is_dir()


def cloud_reason(folder):
    """Why this folder is synced to the cloud (or None). Gigabytes of raw video would keep uploading there."""
    p = str(Path(folder).expanduser().absolute())
    if "Mobile Documents" in p or "iCloud" in p:
        return "這個位置在 iCloud 雲碟裡"
    if "OneDrive" in p:
        return "這個位置在 OneDrive 裡（Windows 常把桌面搬進 OneDrive）"
    if mac_icloud_desktop():
        home = Path.home()
        for sub in ("Desktop", "Documents"):
            try:
                Path(p).relative_to(home / sub)
                return "這台 Mac 開了「iCloud 桌面與文件」，桌面上的東西都會同步到 iCloud"
            except ValueError:
                pass
    return None


def default_folder():
    desk = _windows_desktop() if IS_WIN else Path.home() / "Desktop"
    return (desk if desk.exists() else Path.home()) / "AI剪片"


def choose_folder(folder, allow_cloud=False):
    """The folder the user asked for (default: Desktop), unless it is synced to iCloud / OneDrive: then
    ~/AI剪片, with a plain explanation."""
    want = Path(os.path.abspath(Path(folder).expanduser() if folder else default_folder()))
    why = None if allow_cloud else cloud_reason(want)
    if not why:
        return want, None, None
    safe = Path.home() / "AI剪片"
    note = (f"{why}。影片和剪片暫存檔動輒好幾 GB，放那裡會一直上傳、佔雲端空間，也會拖慢剪片，"
            f"所以「AI剪片」改放在：{safe}（一樣可以從 Finder／檔案總管打開，也可以拉到側邊欄）。"
            "如果你確定要放在雲端資料夾，安裝時加上 --allow-cloud。")
    return safe, why, note


# ---------------------------------------------------------------- check

def check(verbose=True):
    rep = {"系統": f"{platform.system()} {platform.release()}（{platform.machine()}）"}
    py, v = find_python()
    rep["Python"] = f"{v[0]}.{v[1]}（{' '.join(py)}）" if py else f"缺少：需要 3.10～3.13（建議 3.12）→ {python_hint()}"
    exe, fv = ffmpeg_version()
    if not exe:
        rep["ffmpeg"] = f"缺少 → {ffmpeg_hint()}"
    elif fv and fv < FFMPEG_MIN:
        rep["ffmpeg"] = f"缺少：版本 {fv[0]}.{fv[1]} 太舊，需要 5.1 以上 → {ffmpeg_hint()}"
    else:
        rep["ffmpeg"] = f"{exe}（{'版本 %d.%d' % fv if fv and fv[0] != 99 else '版本 OK'}）"
    rep["ffprobe"] = shutil.which("ffprobe") or "缺少（跟 ffmpeg 一起裝）"
    if IS_WIN and platform.machine().upper() == "ARM64":
        rep["注意"] = "Windows ARM（Snapdragon）的聽打套件沒有 ARM 版，請改裝 x64 版的 Python 3.12（會用模擬執行）"
    rep["加速"] = gpu()
    free = shutil.disk_usage(str(Path.home())).free / 1e9
    rep["硬碟剩餘"] = f"{free:.0f} GB" + ("（建議至少 10 GB：模型約 1.5 GB，剪片暫存檔約影片的 2～3 倍）" if free < 10 else "")
    rep["剪片工具環境"] = "已建立" if venv_python().exists() else "還沒建立（install 會建）"
    rep["字型"] = "已下載" if is_ok(FONT, FONT_SHA256) else "還沒下載（install 會下載 Noto Sans TC，約 6 MB）"
    folder, why, _ = choose_folder(None)
    rep["AI剪片資料夾預設位置"] = str(folder) + (f"（{why}，所以改放家目錄）" if why else "")
    rep["版本"] = __version__
    missing = [k for k, val in rep.items() if str(val).startswith("缺少")]
    if verbose:
        print("── 環境檢查 ──")
        for k, val in rep.items():
            print(f"{k}：{val}")
        print("結果：", "可以安裝" if not missing else f"還缺 {'、'.join(missing)}，請先裝好再執行 install")
    return rep, missing, py


# ---------------------------------------------------------------- install

def pip(vp, args, what):
    """Run pip with its progress visible; on failure print a plain message + the last lines. Returns True if OK."""
    env = {**os.environ, "PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    tail = collections.deque(maxlen=20)
    p = subprocess.Popen([vp, "-m", "pip", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        tail.append(line.rstrip())
        if line.startswith(("Collecting", "Downloading", "Installing", "Successfully", "ERROR", "Requirement already")):
            print("  " + line.split(" (from ")[0].rstrip()[:120], flush=True)
    if p.wait() == 0:
        return True
    print(f"\n[錯誤] {what}沒有裝成功。最後幾行訊息：")
    print("\n".join("  " + t for t in tail))
    print("常見原因：網路斷了（重跑一次就好）、硬碟空間不夠、或這台電腦的 Python 版本沒有對應的套件。"
          "請把上面這段整段貼給 AI 助理。")
    return False


def self_test(vp):
    """Import every package the steps need, one by one, and say plainly what works."""
    code = r"""
import importlib, json, platform, sys
sys.path.insert(0, sys.argv[1])
res = {}
for label, mods in [("基本（numpy、Pillow、簡轉繁）", ["numpy", "PIL", "opencc"]),
                    ("聽打引擎", ["mlx_whisper"] if (platform.system() == "Darwin" and platform.machine() == "arm64")
                                 else ["faster_whisper", "ctranslate2"]),
                    ("動態特效（OpenCV、MediaPipe）", ["cv2", "mediapipe", "mediapipe.tasks.python.vision"])]:
    try:
        for m in mods:
            importlib.import_module(m)
        res[label] = "OK"
    except Exception as e:
        res[label] = "失敗：%s: %s" % (type(e).__name__, str(e)[:160])
try:
    from autoedit.common import font_path
    res["字型"] = font_path()
except BaseException as e:
    res["字型"] = "失敗：%s" % e
print(json.dumps(res, ensure_ascii=False))
"""
    r = subprocess.run([vp, "-c", code, str(REPO)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**os.environ, "PYTHONUTF8": "1"})
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"自我測試": "失敗：" + (r.stderr.strip()[-300:] or "沒有輸出")}


def run_command(vp):
    """The command the AI tools use. Forward slashes work in bash, PowerShell (with &) and cmd alike."""
    py, ae = Path(vp), REPO / "ae.py"
    if IS_WIN:
        return f'"{py.as_posix()}" "{ae.as_posix()}"'
    return f'"{py}" "{ae}"'


def write_agent_files(folder, text):
    """CLAUDE.md (Claude Code) and AGENTS.md (Codex). A copy the user changed is backed up, never just replaced."""
    stamp = datetime.date.today().strftime("%Y%m%d")
    for name in ("CLAUDE.md", "AGENTS.md"):
        p = folder / name
        if p.exists():
            old = p.read_text(encoding="utf-8", errors="replace")
            if old == text:
                continue
            bak = p.with_name(f"{name}.bak-{stamp}")
            if not bak.exists():
                shutil.copy2(p, bak)
                print(f"{name} 跟新版不一樣，舊的備份成 {bak.name}")
        p.write_text(text, encoding="utf-8")


def install(folder, allow_cloud=False, skip_fx=False):
    rep, missing, py = check()
    if missing:
        sys.exit(1)
    if not venv_python().exists():
        print("\n建立剪片工具專用的 Python 環境…", flush=True)
        r = subprocess.run([*py, "-m", "venv", str(VENV)])
        if r.returncode != 0 or not venv_python().exists():
            shutil.rmtree(VENV, ignore_errors=True)       # half a venv would look "already there" next time
            print("[錯誤] 建立 Python 環境失敗。Linux 請先執行 sudo apt install python3-venv，再重跑 install。")
            sys.exit(1)
    vp = str(venv_python())
    print("安裝套件（第一次約 1～5 分鐘，會下載幾百 MB，裝好後約 1.5 GB）…", flush=True)
    if not pip(vp, ["install", "--upgrade", "pip"], "pip 更新"):
        sys.exit(1)
    if not pip(vp, ["install", "-r", str(REPO / "requirements.txt")], "主要套件"):
        sys.exit(1)
    fx_ok = False
    if not skip_fx:
        print("安裝動態特效用的套件（MediaPipe，選用）…", flush=True)
        fx_ok = pip(vp, ["install", "-r", str(REPO / "requirements-fx.txt")], "動態特效套件")
        if not fx_ok:
            print("[注意] 動態特效（字從頭後滑出、光軌等）在這台電腦先不能用，其他功能（聽打、剪片、字幕、字卡、封面）都正常。")
    if not is_ok(FONT, FONT_SHA256):
        print("下載字型 Noto Sans TC Bold（開源字型，約 6 MB）…", flush=True)
        try:
            fetch(FONT_URL, FONT, FONT_SHA256, "字型")
            fetch(FONT_LICENSE_URL, FONT_LICENSE, FONT_LICENSE_SHA256, "字型授權書")
        except DownloadError as e:
            print(f"[注意] {e}\n先改用電腦內建的中文字型，之後重跑 install 會再下載。")

    folder, _, note = choose_folder(folder, allow_cloud)
    if note:
        print(f"\n[說明] {note}\n")
    for sub in ("原片", "成片", "工作區"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    run = run_command(vp)
    tpl = (REPO / "agent" / "AGENTS.template.md").read_text(encoding="utf-8")
    ps = (f"\n- Windows 的 PowerShell（例如 Codex）要在最前面加 `& `，也就是：\n\n```\n& {run}\n```\n"
          if IS_WIN else "")
    text = tpl.replace("{{RUN}}", run).replace("{{REPO}}", str(REPO)).replace("{{RUN_POWERSHELL}}", ps)
    write_agent_files(folder, text)
    (REPO / "install" / "installed.json").write_text(json.dumps(
        {"version": __version__, "folder": str(folder), "python": vp, "run": run, "fx": fx_ok},
        ensure_ascii=False, indent=1), encoding="utf-8")

    res = self_test(vp)
    print("\n── 安裝自我測試 ──")
    bad = False
    for k, val in res.items():
        print(f"{k}：{val}")
        if "libGL" in str(val):
            print("  → Linux 缺少圖形函式庫，執行 sudo apt install libgl1 再重跑 install，就能用動態特效。")
        if str(val).startswith("失敗") and not k.startswith("動態特效"):
            bad = True
    print("\n── 安裝完成 ──" if not bad else "\n── 安裝有問題 ──")
    print(f"剪片資料夾：{folder}")
    print("  原片／  ← 把要剪的影片放這裡")
    print("  成片／  ← 剪好的影片、字幕檔、封面會出現在這裡")
    print("  工作區／← 剪片過程的暫存檔")
    if bad:
        print("上面標「失敗」的項目要先處理。請把這整段貼給 AI 助理。")
        sys.exit(1)
    print("第一次聽打時會自動下載語音模型（約 1.5 GB），之後就不用再下載。")
    print("以後要剪片：在這個資料夾打開 Claude Code 或 Codex，說「幫我剪這支」。")


if __name__ == "__main__":
    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "install"])
    ap.add_argument("--folder")
    ap.add_argument("--allow-cloud", action="store_true", help="允許把 AI剪片 放在 iCloud／OneDrive 同步的資料夾")
    ap.add_argument("--skip-fx", action="store_true", help="不裝動態特效（MediaPipe）")
    a = ap.parse_args()
    check() if a.cmd == "check" else install(a.folder, a.allow_cloud, a.skip_fx)
