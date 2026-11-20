"""Safe downloads, standard library only (the installer uses this before any package is installed).

- Downloads to `<name>.part` first and only renames it to the real name after the SHA-256 matches, so a download
  that was cut off (network drop, the AI tool's command timeout) never leaves a broken file behind.
- A file that is already there but does not match (for example from an older, interrupted install) is downloaded
  again.
- Uses certifi's certificates when available: python.org's macOS Python has none until "Install Certificates" is run.
"""
import hashlib
import os
import ssl
import urllib.request
from pathlib import Path


class DownloadError(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_ok(path, digest):
    p = Path(path)
    return p.is_file() and sha256(p) == digest


def _context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch(url, dest, digest, label="檔案", timeout=60):
    """Make sure `dest` exists with the given SHA-256. Returns the path. Raises DownloadError (plain message)."""
    dest = Path(dest)
    if dest.exists():
        if is_ok(dest, digest):
            return dest
        print(f"{label}之前沒有下載完整（檔案壞了），重新下載…", flush=True)
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "coocolab-autoedit"})
        kw = {"context": _context()} if url.lower().startswith("https:") else {}
        with urllib.request.urlopen(req, timeout=timeout, **kw) as r, open(part, "wb") as f:
            while True:
                block = r.read(1 << 16)
                if not block:
                    break
                f.write(block)
    except Exception as e:
        part.unlink(missing_ok=True)
        msg = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in msg:
            hint = ("電腦的 Python 缺少網路憑證。Mac 用 python.org 安裝的 Python 時，打開「應用程式 → Python 3.x」"
                    "資料夾，雙擊 Install Certificates.command，再重跑一次。")
        else:
            hint = "請確認網路是通的（公司或學校網路可能擋下載），再重跑一次這一步。"
        raise DownloadError(f"{label}下載失敗：{msg}\n{hint}") from None
    got = sha256(part)
    if got != digest:
        part.unlink(missing_ok=True)
        raise DownloadError(f"{label}下載下來的內容不對（可能沒下載完，或網路被改寫）。請再重跑一次這一步。")
    os.replace(part, dest)
    return dest

