"""Entry point that works from any folder:  <python> /path/to/coocolab-autoedit/ae.py <step> ..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoedit.__main__ import main  # noqa: E402

main()
