#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pricing SQL watcher daemon.

This module is intentionally self-contained so the watcher logic is easy to read,
debug, and maintain in one place.

High-level flow:
1. `Watcher.start()` verifies dependencies and starts a filesystem observer.
2. `SQLFileHandler` receives SQL create/modify events.
3. Files are validated, deduplicated, then queued.
4. A background worker waits for file stability and executes the ML pipeline.
5. Results and metrics are persisted under `runtime/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from pricing__epac.src.config.settings import settings
from pricing__epac.src.shared.logging import configure_logging


IS_WINDOWS = platform.system() == "Windows"

PROJECT_ROOT = settings.PROJECT_ROOT
PRICING_EPAC_ROOT = PROJECT_ROOT / "pricing__epac"
DATA_ROOT = settings.DATA_ROOT
RAW_DATA = DATA_ROOT / "raw"
CONSOLIDATED_DATA = DATA_ROOT / "consolidated"
SQL_FOLDER = RAW_DATA / "dumps" / "sql"
PROCESSED_FOLDER = RAW_DATA / "dumps" / "processed"

PID_FILE = settings.WATCHER_RUNTIME_ROOT / "watcher.pid"
LOG_FILE = settings.RUNTIME_LOGS_ROOT / "pipeline_output.log"
RESULTS_DIR = settings.PIPELINE_RESULTS_ROOT
METRICS_FILE = settings.WATCHER_RUNTIME_ROOT / "watcher_metrics.json"
IGNORED_FILES = ["current_source.sql", "mysql_db_dump.sql"]


def load_env_file() -> bool:
    """Load root .env file and log minimal diagnostics."""
    env_file = settings.PROJECT_ROOT / ".env"
    if not env_file.exists():
        print(f"Warning: .env file not found at: {env_file}")
        return False

    load_dotenv(env_file)
    print(f".env file loaded from: {env_file}")
    print(f"MYSQL_HOST: {os.getenv('MYSQL_HOST', 'localhost')}")
    print(f"MYSQL_PORT: {os.getenv('MYSQL_PORT', '3307')}")
    return True


load_env_file()


@dataclass
class WatcherConfig:
    """Runtime options for watcher behavior."""

    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", 3307))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")

    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    max_retries: int = int(os.getenv("MAX_RETRIES", 3))
    retry_delay: int = int(os.getenv("RETRY_DELAY", 60))
    pipeline_timeout: int = int(os.getenv("PIPELINE_TIMEOUT", 14400))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", 500))
    file_ready_retries: int = 10
    file_ready_delay: int = 1
    max_processed_cache: int = 100
    shutdown_timeout: int = 5


@dataclass
class ProcessingMetrics:
    """Aggregated watcher processing metrics."""

    files_processed: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_execution_time: float = 0.0
    retries_used: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def record_success(self, duration: float, retries: int = 0) -> None:
        self.files_processed += 1
        self.successful_runs += 1
        self.total_execution_time += duration
        self.retries_used += retries

    def record_failure(self, error: str, retries: int = 0) -> None:
        self.files_processed += 1
        self.failed_runs += 1
        self.retries_used += retries
        self.errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "error": error,
                "retries": retries,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": self.successful_runs / max(self.files_processed, 1),
            "average_execution_time": self.total_execution_time / max(self.successful_runs, 1),
            "total_execution_time": self.total_execution_time,
            "retries_used": self.retries_used,
            "recent_errors": self.errors[-5:],
        }


def setup_logging(log_file: Path, log_level: str = "INFO") -> logging.Logger:
    """Configure console + rotating file logging."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = configure_logging(level=level, reset_handlers=True, logger_name=__name__)

    console_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(console_formatter)

    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10_485_760,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as exc:
        logger.warning("Could not create file handler: %s", exc)

    return logger


logger = setup_logging(LOG_FILE, os.getenv("LOG_LEVEL", "INFO"))


def check_pid_file(pid_file: Path) -> Optional[int]:
    """Return running PID if pid file exists and process is alive."""
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return None

    try:
        if IS_WINDOWS:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                shell=True,
                timeout=5,
            )
            return pid if str(pid) in result.stdout else None
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


def cleanup_stale_pid_file(pid_file: Path) -> None:
    if pid_file.exists():
        try:
            pid_file.unlink()
        except Exception as exc:
            logger.warning("Could not remove stale PID file: %s", exc)


def write_pid_file(pid_file: Path) -> bool:
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except Exception as exc:
        logger.error("Failed to create PID file: %s", exc)
        return False


@contextmanager
def pid_lock(pid_file: Path):
    """Create and cleanup watcher PID file around daemon lifecycle."""
    existing_pid = check_pid_file(pid_file)
    if existing_pid:
        logger.error("Watcher already running with PID %s", existing_pid)
        yield False
        return

    cleanup_stale_pid_file(pid_file)
    if not write_pid_file(pid_file):
        yield False
        return

    try:
        yield True
    finally:
        cleanup_stale_pid_file(pid_file)


class DependencyChecker:
    """Check required services/tools before watcher startup."""

    @staticmethod
    def check_mysql() -> bool:
        try:
            import mysql.connector

            mysql_password = os.getenv("MYSQL_PASSWORD", "")
            if not mysql_password:
                logger.error("MYSQL_PASSWORD is not set")
                return False

            conn = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3307")),
                user=os.getenv("MYSQL_USER", "root"),
                password=mysql_password,
                connection_timeout=5,
            )
            conn.close()
            return True
        except Exception as exc:
            logger.error("MySQL connectivity check failed: %s", exc)
            return False

    @staticmethod
    def check_mlflow() -> bool:
        try:
            import requests

            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
            response = requests.get(f"{mlflow_uri}/health", timeout=5)
            if response.status_code == 200:
                return True
            logger.warning("MLflow health returned status %s", response.status_code)
            return True
        except Exception as exc:
            logger.warning("MLflow check skipped/failed: %s", exc)
            return True

    @staticmethod
    def check_poetry() -> bool:
        try:
            result = subprocess.run(
                ["poetry", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                shell=IS_WINDOWS,
            )
            return result.returncode == 0
        except Exception:
            return False

    @classmethod
    def check_all(cls) -> bool:
        return cls.check_mysql() and cls.check_poetry() and cls.check_mlflow()


class SQLFileHandler(FileSystemEventHandler):
    """Receive SQL file events and execute the pipeline serially from a queue."""

    def __init__(self, mode: str = "all", config: Optional[WatcherConfig] = None):
        self.config = config or WatcherConfig()
        self.mode = mode
        self.pending_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.processing = False
        self.stop_monitoring = False
        self.current_process: Optional[subprocess.Popen] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.last_trigger: float = time.time()
        self.queued_or_processing_paths: Set[str] = set()
        self.processed_files: Set[str] = set()
        self.metrics = ProcessingMetrics()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
        self.start_processing_thread()

    def start_processing_thread(self) -> None:
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()

    def  _process_queue(self) -> None:
        while not self.stop_monitoring:
            try:
                sql_file = self.pending_queue.get(timeout=1)
            except queue.Empty:
                continue

            if sql_file is None:
                break

            self.processing = True
            normalized_path = self._normalize_file_path(sql_file)
            try:
                if self._wait_for_file_ready(sql_file):
                    self._run_pipeline_with_retry(sql_file)
            except Exception as exc:
                logger.error("Queue processing failed for %s: %s", sql_file, exc)
            finally:
                self.queued_or_processing_paths.discard(normalized_path)
                self.processing = False
                self.pending_queue.task_done()

    def _wait_for_file_ready(self, file_path: str, max_retries: Optional[int] = None, delay: Optional[int] = None) -> bool:
        retries = max_retries or self.config.file_ready_retries
        wait_delay = delay or self.config.file_ready_delay

        for _ in range(retries):
            try:
                with open(file_path, "rb") as file_handle:
                    file_handle.read(1024)
                size_1 = os.path.getsize(file_path)
                time.sleep(0.5)
                size_2 = os.path.getsize(file_path)
                if size_1 == size_2 and size_1 > 0:
                    size_mb = size_1 / (1024 * 1024)
                    if size_mb > self.config.max_file_size_mb:
                        logger.error("File too large: %.2f MB", size_mb)
                        return False
                    return True
            except Exception:
                time.sleep(wait_delay)
        logger.error("File not ready after retries: %s", file_path)
        return False

    def _normalize_file_path(self, file_path: str) -> str:
        return str(Path(file_path).resolve())

    def _get_file_hash(self, file_path: str) -> str:
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as file_handle:
                for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return f"{file_path}_{os.path.getmtime(file_path)}"

    def on_created(self, event) -> None:
        if not event.is_directory and event.src_path.endswith(".sql"):
            self._handle_new_sql_file(event.src_path)

    def on_modified(self, event) -> None:
        if not event.is_directory and event.src_path.endswith(".sql"):
            now = time.time()
            if now - self.last_trigger > 2:
                self.last_trigger = now
                self._handle_new_sql_file(event.src_path)

    def _handle_new_sql_file(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            return

        file_name = Path(file_path).name
        if file_name in IGNORED_FILES:
            return
        if not self._validate_sql_file(file_path):
            return

        normalized_path = self._normalize_file_path(file_path)
        if normalized_path in self.queued_or_processing_paths:
            return

        self.queued_or_processing_paths.add(normalized_path)
        file_hash = self._get_file_hash(file_path)
        if file_hash in self.processed_files:
            self.queued_or_processing_paths.discard(normalized_path)
            return

        self.processed_files.add(file_hash)
        if len(self.processed_files) > self.config.max_processed_cache:
            self.processed_files.clear()
        self.pending_queue.put(file_path)

    def _validate_sql_file(self, file_path: str) -> bool:
        if not file_path.endswith(".sql"):
            return False
        if not os.access(file_path, os.R_OK):
            return False
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb <= 0 or size_mb > self.config.max_file_size_mb:
            return False
        return True

    def _run_pipeline_with_retry(self, sql_file: str, retry_count: int = 0) -> bool:
        success = self._run_pipeline(sql_file)
        if success:
            return True
        if retry_count >= self.config.max_retries:
            return False
        wait_seconds = self.config.retry_delay * (2 ** retry_count)
        time.sleep(wait_seconds)
        return self._run_pipeline_with_retry(sql_file, retry_count + 1)

    def _run_pipeline(self, sql_file: str) -> bool:
        cmd = [
            "poetry",
            "run",
            "python",
            "-u",
            "-m",
            "pricing__epac.src.machine_learning.flows.pricing_full_pipeline",
            "--mode",
            self.mode,
            "--input",
            str(sql_file),
            "--once",
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["MLFLOW_TRACKING_URI"] = self.config.mlflow_uri
        if self.config.mysql_password:
            env["MYSQL_PASSWORD"] = self.config.mysql_password

        start = time.time()
        output_lines: List[str] = []
        has_errors = False

        try:
            self.current_process = subprocess.Popen(
                cmd,
                cwd=PRICING_EPAC_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding="utf-8",
                bufsize=1,
                universal_newlines=True,
            )

            def read_stream(stream, is_error: bool) -> None:
                nonlocal has_errors
                try:
                    for line in iter(stream.readline, ""):
                        line = line.rstrip()
                        if not line:
                            continue
                        output_lines.append(line)
                        if is_error:
                            has_errors = True
                            logger.error(self._normalize_output_line(line))
                        else:
                            self._display_line(line)
                finally:
                    stream.close()

            stdout_thread = threading.Thread(target=read_stream, args=(self.current_process.stdout, False), daemon=True)
            stderr_thread = threading.Thread(target=read_stream, args=(self.current_process.stderr, True), daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            return_code = self.current_process.wait(timeout=self.config.pipeline_timeout)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            elapsed = time.time() - start
            if return_code == 0 and not has_errors:
                self.metrics.record_success(elapsed)
                self._save_results(output_lines, elapsed, sql_file)
                self._save_metrics()
                self._move_processed_file(sql_file)
                return True

            self.metrics.record_failure(f"Exit code: {return_code}")
            self._save_metrics()
            return False
        except subprocess.TimeoutExpired:
            if self.current_process:
                self.current_process.kill()
            self.metrics.record_failure("Timeout")
            self._save_metrics()
            return False
        except Exception as exc:
            self.metrics.record_failure(str(exc))
            self._save_metrics()
            logger.error("Pipeline launch failed: %s", exc)
            return False
        finally:
            self.current_process = None

    def _move_processed_file(self, sql_file: str) -> None:
        src = Path(sql_file)
        if src.name in IGNORED_FILES or src.parent == PROCESSED_FOLDER:
            return
        dst = PROCESSED_FOLDER / src.name
        shutil.move(str(src), str(dst))

    def _display_line(self, line: str) -> None:
        normalized = self._normalize_output_line(line)
        if "ERROR" in normalized or "FAILED" in normalized:
            logger.error(normalized)
        elif "WARNING" in normalized:
            logger.warning(normalized)
        else:
            logger.info(normalized)

    @staticmethod
    def _normalize_output_line(line: str) -> str:
        normalized_line = line.strip()
        return re.sub(
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s*-\s*",
            "",
            normalized_line,
            count=1,
        )

    def _save_results(self, output_lines: List[str], elapsed_seconds: float, sql_file: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"pipeline_{Path(sql_file).stem}_{timestamp}.txt"
        with open(result_file, "w", encoding="utf-8") as file_handle:
            file_handle.write(f"Executed at: {datetime.now().isoformat()}\n")
            file_handle.write(f"SQL file: {sql_file}\n")
            file_handle.write(f"Mode: {self.mode}\n")
            file_handle.write(f"Duration seconds: {elapsed_seconds:.2f}\n")
            file_handle.write("=" * 80 + "\n")
            for line in output_lines:
                file_handle.write(line + "\n")

    def _save_metrics(self) -> None:
        with open(METRICS_FILE, "w", encoding="utf-8") as file_handle:
            json.dump(self.metrics.to_dict(), file_handle, indent=2, default=str)

    def stop(self) -> None:
        self.stop_monitoring = True
        self.pending_queue.put(None)
        if self.current_process:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=self.config.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=self.config.shutdown_timeout)


class Watcher:
    """
    Main daemon lifecycle wrapper around watchdog + SQLFileHandler.

    Responsibilities:
    1. Validate startup prerequisites.
    2. Start filesystem monitoring.
    3. Keep the daemon alive.
    4. Handle graceful shutdown and final metrics reporting.
    """

    def __init__(self, watch_path: Optional[Path] = None, mode: str = "all", config: Optional[WatcherConfig] = None):
        """
        Step 1: Build watcher runtime context.

        - Load runtime config (or default WatcherConfig).
        - Resolve the folder to monitor.
        - Ensure the watched folder exists.
        - Create watchdog observer and SQL file handler.
        """
        self.config = config or WatcherConfig()
        self.watch_path = watch_path or SQL_FOLDER
        self.mode = mode
        self.watch_path.mkdir(parents=True, exist_ok=True)
        self.observer = Observer()
        self.handler = SQLFileHandler(mode, self.config)
        logger.info("Watcher initialized")

    def start(self) -> None:
        """
        Step 2: Start daemon lifecycle.

        Flow:
        - Validate dependencies (MySQL, Poetry, etc.).
        - Print startup banner (runtime context).
        - Acquire PID lock to prevent multiple watcher instances.
        - Register file handler on the observed directory.
        - Start observer + register signal handlers.
        - Enter keep-alive loop until interruption.
        """
        if not DependencyChecker.check_all():
            logger.error("Critical dependency check failed. Exiting.")
            sys.exit(1)

        self._print_banner()
        with pid_lock(PID_FILE) as acquired:
            if not acquired:
                sys.exit(1)

            # Bind SQL event handler to the monitored folder.
            self.observer.schedule(self.handler, str(self.watch_path), recursive=False)
            self.observer.start()
            # Graceful stop on Ctrl+C / termination.
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            logger.info("Watcher active")
            logger.info("Monitored folder: %s", self.watch_path)
            logger.info("Processed folder: %s", PROCESSED_FOLDER)
            logger.info("Results folder: %s", RESULTS_DIR)
            logger.info("Metrics file: %s", METRICS_FILE)

            try:
                # Keep daemon alive while observer/handler threads do the real work.
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()

    def _signal_handler(self, sig, frame) -> None:
        """Step 3: Handle OS signals and trigger graceful shutdown."""
        logger.info("Received signal %s", sig)
        self.stop()

    def stop(self) -> None:
        """
        Step 4: Stop daemon safely.

        - Stop handler worker thread/process.
        - Stop watchdog observer thread.
        - Print final execution metrics.
        - Exit process.
        """
        logger.info("Stopping watcher...")
        self.handler.stop()
        self.observer.stop()
        self.observer.join()

        metrics = self.handler.metrics.to_dict()
        logger.info("Files processed: %s", metrics["files_processed"])
        logger.info("Successful runs: %s", metrics["successful_runs"])
        logger.info("Failed runs: %s", metrics["failed_runs"])
        logger.info("Success rate: %.1f%%", metrics["success_rate"] * 100)
        logger.info("Average time: %.1f min", metrics["average_execution_time"] / 60)
        logger.info("Retries used: %s", metrics["retries_used"])
        logger.info("Watcher stopped")
        sys.exit(0)

    def _print_banner(self) -> None:
        """Startup diagnostics: print key runtime paths, mode, and environment info."""
        logger.info("=" * 70)
        logger.info("PRICING WATCHER - CONTINUOUS DATA PROCESSING")
        logger.info("Project root: %s", PROJECT_ROOT)
        logger.info("Pricing EPAC root: %s", PRICING_EPAC_ROOT)
        logger.info("Data root: %s", DATA_ROOT)
        logger.info("Monitored: %s", self.watch_path)
        logger.info("Mode: %s", self.mode)
        logger.info("Processed: %s", PROCESSED_FOLDER)
        logger.info("Results: %s", RESULTS_DIR)
        logger.info("Metrics: %s", METRICS_FILE)
        logger.info("Consolidated: %s", CONSOLIDATED_DATA)
        logger.info("Ignored files: %s", IGNORED_FILES)
        logger.info("OS: %s", platform.system())
        logger.info("MySQL: %s:%s as %s", self.config.mysql_host, self.config.mysql_port, self.config.mysql_user)
        logger.info("=" * 70)


def main() -> None:
    """
    CLI entry point.

    - Parse mode and custom watch directory.
    - Instantiate Watcher.
    - Start daemon.
    """
    parser = argparse.ArgumentParser(description="Pricing SQL watcher daemon")
    parser.add_argument("--mode", default="all", choices=["all", "global", "family", "couple", "features"])
    parser.add_argument("--watch", default=None, help="Custom folder to monitor")
    args = parser.parse_args()

    custom_watch = Path(args.watch) if args.watch else None
    watcher = Watcher(watch_path=custom_watch, mode=args.mode)
    watcher.start()


if __name__ == "__main__":
    main()
