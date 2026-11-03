"""Supplementary experiments. Legacy sources are loaded without modifying them."""
__version__ = "0.1.0"

# Optional experiment-only dependencies; existing environment packages are not overwritten.
from pathlib import Path as _Path
import sys as _sys
_local_dependencies = _Path(__file__).resolve().parents[1] / "revision_external" / "python"
if _local_dependencies.is_dir():
    _sys.path.insert(0, str(_local_dependencies))
