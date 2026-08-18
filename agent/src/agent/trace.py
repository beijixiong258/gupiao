"""TraceWriter: crash-safe JSONL trace writer.

One JSON record per line; append + flush guarantees no data loss on crash.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.core.run_policy import RunLogPolicy, trace_entry_for_policy


class TraceWriter:
    """JSONL trace writer, one record per line, crash-safe.

    Attributes:
        path: Path to the trace.jsonl file.
    """

    def __init__(self, run_dir: Path, policy: RunLogPolicy | None = None) -> None:
        """Initialize TraceWriter.

        Args:
            run_dir: Run directory; trace.jsonl is written here.
        """
        self.policy = policy or RunLogPolicy.load()
        self.path = run_dir / "trace.jsonl"
        self._file = (
            open(self.path, "a", encoding="utf-8")
            if self.policy.enabled
            else None
        )

    def write(self, entry: Dict[str, Any]) -> None:
        """Write a trace record.

        Args:
            entry: Trace entry; a ts field is added automatically.
        """
        if self._file is None:
            return
        payload = dict(entry)
        if "ts" not in payload:
            payload["ts"] = time.time()
        payload = trace_entry_for_policy(payload, self.policy)
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._file is not None:
            self._file.close()

    @staticmethod
    def read(run_dir: Path) -> List[Dict[str, Any]]:
        """Read trace.jsonl and return records.

        Args:
            run_dir: Run directory.

        Returns:
            List of trace records.
        """
        path = run_dir / "trace.jsonl"
        if not path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries
