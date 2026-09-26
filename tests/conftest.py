import sys
from pathlib import Path

source = Path(__file__).resolve().parents[1] / "python"
if any((source / "grayt").glob("_core*.so")) or any((source / "grayt").glob("_core*.pyd")):
    sys.path.insert(0, str(source))
