"""运行状态存储：遵循隐私策略创建、裁剪和清理运行目录。"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.core.run_policy import (
    RunLogPolicy,
    create_ephemeral_run_dir,
    prune_run_directories,
    redact_for_log,
    response_for_policy,
)


class RunStateStore:
    """运行记录生命周期；默认仅保存脱敏元数据。"""

    def __init__(self, policy: RunLogPolicy | None = None) -> None:
        self.policy = policy or RunLogPolicy.load()
        self._ephemeral_paths: set[Path] = set()
        self.last_pruned: tuple[str, ...] = ()

    def create_run_dir(self, workspace: Path) -> Path:
        if self.policy.enabled:
            self.last_pruned = tuple(prune_run_directories(workspace, self.policy))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
            suffix = uuid.uuid4().hex[:6]
            run_dir = workspace / f"{timestamp}_{suffix}"
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = create_ephemeral_run_dir()
            self._ephemeral_paths.add(run_dir.resolve())
        (run_dir / "code").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        return run_dir

    def public_location(self, run_dir: Path) -> tuple[str, str]:
        if not self.policy.enabled:
            return "", "not_saved"
        return str(run_dir), run_dir.name

    def cleanup_ephemeral(self, run_dir: Path) -> None:
        resolved = run_dir.resolve()
        if resolved not in self._ephemeral_paths:
            return
        shutil.rmtree(resolved, ignore_errors=True)
        self._ephemeral_paths.discard(resolved)

    def save_request(self, run_dir: Path, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.policy.content_mode == "metadata_only":
            payload = {
                "prompt_saved": False,
                "prompt_length": len(str(prompt)),
                "context": redact_for_log(context),
            }
        else:
            payload = redact_for_log({"prompt": prompt, "context": context})
        self._write_json(run_dir / "req.json", payload)
        return payload

    def mark_success(self, run_dir: Path) -> None:
        self.mark_status(run_dir, "success")

    def mark_status(self, run_dir: Path, status: str, reason: str = "") -> None:
        payload = {"status": str(status)}
        if reason:
            if self.policy.content_mode == "metadata_only":
                payload["reason_saved"] = False
                payload["reason_length"] = len(str(reason))
            else:
                payload["reason"] = reason
        self._write_json(run_dir / "state.json", redact_for_log(payload))

    def mark_failure(self, run_dir: Path, reason: str) -> None:
        self.mark_status(run_dir, "failed", reason)

    def save_response(self, run_dir: Path, response: Dict[str, Any]) -> None:
        self._write_json(
            run_dir / "response.json",
            response_for_policy(response, self.policy),
        )

    def _write_json(self, path: Path, data: Any) -> None:
        if not self.policy.enabled:
            return
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


__all__ = ["RunStateStore"]
