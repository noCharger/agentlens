import importlib
import subprocess
import sys

import pytest


def test_no_litellm_importable():
    """Ensure litellm is not installed as a direct or indirect dependency."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("litellm")


def test_no_litellm_in_pip_list():
    """Verify litellm does not appear in pip list."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=columns"],
        capture_output=True,
        text=True,
    )
    installed = result.stdout.lower()
    assert "litellm" not in installed, "litellm found in installed packages"
