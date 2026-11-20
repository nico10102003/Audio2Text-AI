"""Compatibility shim for the upstream DistilWhisper code.

The official repository (https://github.com/naver/multilingual-distilwhisper)
was written for transformers 4.30.1, which has no wheels for Python 3.13.
We keep the upstream code untouched and adapt it at runtime:

1. ``DistilWhisper.py`` imports ``datasets`` at module level although inference
   never uses it.  A tiny stub avoids installing pyarrow/pandas/datasets.
2. ``load_experts`` passes ``resume_download``/``proxies`` to
   ``transformers.utils.hub.cached_file``.  Those arguments were removed from
   huggingface_hub, so they are filtered out.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_MODEL_DIR = PROJECT_ROOT / "vendor" / "multilingual-distilwhisper" / "model"


def _install_datasets_stub() -> None:
    """Provide a no-op ``datasets`` module when the real package is absent."""
    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("datasets")
        stub.Dataset = type("Dataset", (), {})
        sys.modules["datasets"] = stub


def _patch_cached_file() -> None:
    """Drop keyword arguments removed from huggingface_hub."""
    from transformers.utils import hub

    original = hub.cached_file
    if getattr(original, "_audio2text_patched", False):
        return

    def cached_file(*args, **kwargs):
        kwargs.pop("resume_download", None)
        kwargs.pop("proxies", None)
        return original(*args, **kwargs)

    cached_file._audio2text_patched = True
    hub.cached_file = cached_file


def import_distilwhisper():
    """Import the upstream model classes with the compatibility fixes applied."""
    # Import transformers first: its import machinery probes for ``datasets``
    # and a bare stub without a real module spec would break that probe.
    import transformers  # noqa: F401
    import torch  # noqa: F401

    _install_datasets_stub()
    _patch_cached_file()

    vendor = str(VENDOR_MODEL_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)

    import DistilWhisper  # type: ignore[import-not-found]

    # transformers >= 4.36 changed the attention return contract; restore the
    # 4.30 version the upstream classes were written against.
    from .legacy_attention import LegacyWhisperAttention

    DistilWhisper.WhisperAttention = LegacyWhisperAttention

    return DistilWhisper
