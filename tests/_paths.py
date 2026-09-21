import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def packer_repo() -> Path:
    """$PACKER_REPO, else the nearest `llm-robotic-packer` beside this repo or one of its parents (worktrees)."""
    if os.environ.get("PACKER_REPO"):
        return Path(os.environ["PACKER_REPO"])
    for parent in ROOT.parents:
        cand = parent / "llm-robotic-packer"
        if (cand / "harness" / "prompts.py").exists():
            return cand
    return ROOT.parent / "llm-robotic-packer"
