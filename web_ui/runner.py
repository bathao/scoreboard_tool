from __future__ import annotations

import threading
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Callable

from wsgiref.simple_server import WSGIServer

from backend.production_jobs import (
    job_json_path_from_id,
    load_match_job,
    update_job_runtime_state,
)
from backend.production_pipeline import ProductionPipelineConfig


class StopRequested(Exception):
    pass


class JobTaskRunner:
    def __init__(self, *, config: ProductionPipelineConfig, jobs_root: Path | None = None):
        self.config = config
        self.jobs_root = jobs_root
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._stop_flags: dict[str, bool] = {}

    def _active_job_id(self) -> str | None:
        for job_id, thread in self._threads.items():
            if thread.is_alive():
                return job_id
        return None

    def start(self, job_id: str, target: Callable[[str], None]) -> tuple[bool, str]:
        with self._lock:
            active_job_id = self._active_job_id()
            if active_job_id and active_job_id != job_id:
                return False, f"Another job is already running: {active_job_id}"

            current = self._threads.get(job_id)
            if current is not None and current.is_alive():
                return False, f"Job {job_id} is already running"

            self._stop_flags[job_id] = False
            thread = threading.Thread(target=self._run_safe, args=(job_id, target), daemon=True)
            self._threads[job_id] = thread
            thread.start()
            return True, f"Started background task for {job_id}"

    def request_stop(self, job_id: str) -> bool:
        if job_id in self._stop_flags:
            self._stop_flags[job_id] = True
            return True
        return False

    def is_stop_requested(self, job_id: str) -> bool:
        return bool(self._stop_flags.get(job_id, False))

    def _run_safe(self, job_id: str, target: Callable[[str], None]) -> None:
        from backend.production_pipeline import _job_log
        try:
            target(job_id)
        except StopRequested:
            job = load_match_job(job_json_path_from_id(job_id, self.jobs_root))
            _job_log(job.artifacts.job_dir, "Pipeline stopped by operator")
            update_job_runtime_state(job, status="failed", current_step="failed", error_message="stopped_by_operator")
        except Exception as exc:
            job = load_match_job(job_json_path_from_id(job_id, self.jobs_root))
            _job_log(job.artifacts.job_dir, f"ERROR: {exc}")
            update_job_runtime_state(job, status="failed", current_step="failed", error_message=str(exc))


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


def _start_heartbeat_watcher(timeout_sec: float = 120.0) -> "list[float]":
    import os
    import time

    last_beat: list[float] = [time.monotonic()]

    def _watch() -> None:
        while True:
            time.sleep(5)
            if time.monotonic() - last_beat[0] > timeout_sec:
                os._exit(0)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return last_beat


def _cleanup_old_logs(max_age_days: int = 3) -> None:
    import time
    from backend.production_pipeline import LOGS_DIR
    if not LOGS_DIR.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    for log_file in LOGS_DIR.glob("*.log"):
        try:
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
        except Exception:
            pass
