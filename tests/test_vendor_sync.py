"""The vendored packer files must stay byte-identical to the packer repo (D31)."""
import hashlib
from pathlib import Path

import pytest

from tests._paths import ROOT, packer_repo

VENDOR = ROOT / "training" / "packer_vendor"
PACKER = packer_repo()

# sha256 of the packer files at commit ccd8fdc (llm-robotic-packer main, 2026-09-21)
PINNED = {
    "prompts.py": ("harness/prompts.py", "bad9c106ec4e5e50c1ee7db94115a3758c8a79de56766de1b8b69a73604d1da0"),
    "state_manager.py": ("envs/state_manager.py", "e8bf06301242a0c7296f3c1e8160b2efc720139dfbcc933fe2d9e38c3ba34cb9"),
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", sorted(PINNED))
def test_vendored_file_matches_pinned_hash(name):
    assert _sha(VENDOR / name) == PINNED[name][1], f"{name} differs from the pinned packer version; re-copy and re-pin"


@pytest.mark.parametrize("name", sorted(PINNED))
def test_vendored_file_matches_packer_checkout(name):
    src = PACKER / PINNED[name][0]
    if not src.exists():
        pytest.skip(f"packer repo not found at {PACKER} (set PACKER_REPO)")
    assert (VENDOR / name).read_bytes() == src.read_bytes(), f"{name}: packer checkout has moved on; re-copy and re-pin"
