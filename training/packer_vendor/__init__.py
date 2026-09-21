"""
Verbatim copies of two files from the llm-robotic-packer repo (commit ccd8fdc),
so that training prompts are built by the same code the evaluation harness uses
(D24, D31):

    prompts.py        <- llm-robotic-packer/harness/prompts.py
    state_manager.py  <- llm-robotic-packer/envs/state_manager.py

Do not edit these files here. Update them by copying from the packer repo and
running `pytest tests/test_vendor_sync.py`, which checks the copies byte-for-byte
against the packer checkout (when present) and against the pinned hashes.
"""
