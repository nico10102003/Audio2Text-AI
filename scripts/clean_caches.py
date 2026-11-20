"""Remove only Python/tool caches from the project source tree.

This helper is invoked by ``make clean``. It deliberately skips the virtual
environments, model weights, vendored code, logs, data and documentation, so
test audio, inference results and docs are never deleted.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".venv-legacy",
    "data",
    "docs",
    "logs",
    "models",
    "vendor",
}
CACHE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".pyright",
}
CACHE_FILES = {".coverage"}


def main() -> int:
    removed = 0
    for dirpath, dirnames, filenames in os.walk(ROOT, topdown=True):
        current = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames if name not in EXCLUDED_DIRS
        )

        for name in list(dirnames):
            if name not in CACHE_DIRS:
                continue
            path = current / name
            try:
                shutil.rmtree(path)
                removed += 1
            except OSError as exc:
                print(f"Aviso: no se pudo eliminar {path}: {exc}")
            dirnames.remove(name)

        for filename in filenames:
            if not (filename.endswith((".pyc", ".pyo")) or filename in CACHE_FILES):
                continue
            path = current / filename
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"Aviso: no se pudo eliminar {path}: {exc}")

    print(f"Limpieza completada: {removed} elemento(s) eliminado(s).")
    print("No se tocaron logs, audios, modelos, resultados ni documentacion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
