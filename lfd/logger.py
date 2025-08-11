import os
import json
from typing import Dict, Any
from datetime import datetime

class DemoLogger:
    def __init__(self, jsonl_path: str):
        self.path = jsonl_path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log_example(self, task: str, state: Dict[str, Any], label: Dict[str, Any], meta: Dict[str, Any]):
        rec = {
            "ts": datetime.utcnow().isoformat(),
            "task": task,
            "state": state,
            "label": label,
            "meta": meta
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
