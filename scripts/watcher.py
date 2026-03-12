#!/usr/bin/env python
# scripts/watcher.py
import time
import logging
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import os
import json
from datetime import datetime
import signal
import threading
import queue
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_FOLDER = PROJECT_ROOT / "data" / "raw" / "dumps" / "sql"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "raw" / "dumps" / "processed"
PID_FILE = PROJECT_ROOT / "data" / "watcher.pid"
LOG_FILE = PROJECT_ROOT / "data" / "pipeline_output.log"
RESULTS_DIR = PROJECT_ROOT / "data" / "pipeline_results"


class SQLFileHandler(FileSystemEventHandler):
    def __init__(self, mode="all"):
        self.processing = False
        self.pending_queue = queue.Queue()
        self.last_trigger = time.time()
        self.mode = mode  # ← mode attribute is here in the handler
        self.current_process = None
        self.stop_monitoring = False
        self.processing_thread = None

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

        self.start_processing_thread()

    def start_processing_thread(self):
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        logger.info("🔄 Processing thread started")

    def _process_queue(self):
        while not self.stop_monitoring:
            try:
                sql_file = self.pending_queue.get(timeout=1)

                if sql_file is None:
                    break

                self.processing = True
                try:
                    if self.wait_for_file_ready(sql_file):
                        self.run_pipeline(sql_file)
                except Exception as e:
                    logger.error(f"❌ Error: {e}")
                finally:
                    self.processing = False
                    self.pending_queue.task_done()

            except queue.Empty:
                continue

    def wait_for_file_ready(self, file_path, max_retries=10, delay=1):
        for i in range(max_retries):
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)
                size1 = os.path.getsize(file_path)
                time.sleep(0.5)
                size2 = os.path.getsize(file_path)
                if size1 == size2 and size1 > 0:
                    logger.info(f"✅ File ready: {Path(file_path).name}")
                    return True
            except:
                time.sleep(delay)
        return False

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.sql'):
            self.handle_new_sql_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.sql'):
            current_time = time.time()
            if current_time - self.last_trigger > 2:
                self.last_trigger = current_time
                self.handle_new_sql_file(event.src_path)

    def handle_new_sql_file(self, file_path):
        if not os.path.exists(file_path):
            return

        file_name = Path(file_path).name
        logger.info(f"➕ New SQL file: {file_name}")
        self.pending_queue.put(file_path)
        logger.info(f"📊 Queue: {self.pending_queue.qsize()}")

    def run_pipeline(self, sql_file):
        try:
            self._print_separator()
            logger.info(f"📊 START - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"📁 File: {Path(sql_file).name}")
            self._print_separator()

            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
            env['PYTHONUNBUFFERED'] = '1'

            # Command with --input to use the specific file
            cmd = [
                "poetry", "run", "python",
                "-u",
                "-m", "pricing_epac.flows.pricing_full_pipeline",
                "--mode", self.mode,
                "--input", str(sql_file),
                "--once"  # To avoid automatic promotion
            ]

            logger.info(f"📋 Command: {' '.join(cmd)}")

            self.current_process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding='utf-8',
                bufsize=1
            )

            pipeline_output = []
            has_errors = False
            start_time = time.time()

            for line in self.current_process.stdout:
                line = line.rstrip()
                pipeline_output.append(line)
                self._display_line(line)
                if "ERROR" in line or "❌" in line:
                    has_errors = True

            for line in self.current_process.stderr:
                line = line.rstrip()
                pipeline_output.append(f"ERR: {line}")
                logger.error(f"🔴 {line}")
                has_errors = True

            return_code = self.current_process.wait(timeout=14400)
            elapsed = time.time() - start_time

            if return_code == 0 and not has_errors:
                logger.info(f"✅ SUCCESS in {elapsed / 60:.1f} minutes!")
                self._save_results(pipeline_output, elapsed, sql_file)

                # Move to processed
                dest = PROCESSED_FOLDER / Path(sql_file).name
                shutil.move(str(sql_file), str(dest))
                logger.info(f"📁 File moved to: {dest}")
            else:
                logger.error(f"❌ FAILED after {elapsed / 60:.1f} minutes")

        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.current_process = None

    def _display_line(self, line):
        if "ERROR" in line or "❌" in line:
            logger.error(f"❌ {line}")
        elif "WARNING" in line or "⚠️" in line:
            logger.warning(f"⚠️ {line}")
        elif "✅" in line:
            logger.info(f"✅ {line}")
        else:
            logger.info(f"   {line}")

    def _print_separator(self):
        logger.info("=" * 80)

    def _save_results(self, output_lines, elapsed_minutes, sql_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sql_name = Path(sql_file).stem

        result_file = RESULTS_DIR / f"pipeline_{sql_name}_{timestamp}.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline at {datetime.now().isoformat()}\n")
            f.write(f"SQL: {sql_file}\n")
            f.write(f"Duration: {elapsed_minutes:.1f} minutes\n")
            f.write("=" * 80 + "\n")
            for line in output_lines:
                f.write(line + "\n")

        logger.info(f"💾 Results: {result_file}")

    def stop(self):
        self.stop_monitoring = True
        self.pending_queue.put(None)
        if self.current_process:
            self.current_process.terminate()


class Watcher:
    def __init__(self, watch_path=None, mode="all"):
        self.watch_path = watch_path or SQL_FOLDER
        self.mode = mode  # ← Added mode attribute here as well
        self.watch_path.mkdir(parents=True, exist_ok=True)
        self.observer = Observer()
        self.handler = SQLFileHandler(mode)

        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)

    def start(self):
        self._print_banner()
        self.observer.schedule(self.handler, str(self.watch_path), recursive=False)

        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))

        self.observer.start()
        logger.info("✅ Watcher active - Ctrl+C to stop")
        logger.info(f"📁 Monitored folder: {self.watch_path}")
        logger.info(f"📁 Processed folder: {PROCESSED_FOLDER}")
        logger.info(f"📊 Results: {RESULTS_DIR}")
        logger.info("🛑 To stop: Ctrl+C\n")

        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def signal_handler(self, sig, frame):
        self.stop()

    def stop(self):
        logger.info("🛑 Stopping...")
        self.handler.stop()
        self.observer.stop()
        self.observer.join()

        if PID_FILE.exists():
            PID_FILE.unlink()

        logger.info("👋 Stopped")
        sys.exit(0)

    def _print_banner(self):
        banner = f"""
{'=' * 60}
🚀 PRICING WATCHER - CONTINUOUS MODE
{'=' * 60}
📁 Folder: {self.watch_path}
⚡ Mode: {self.mode}
📁 Processed: {PROCESSED_FOLDER}
📊 Results: {RESULTS_DIR}
{'=' * 60}
"""
        logger.info(banner)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "global", "family", "couple", "features"])
    args = parser.parse_args()

    # Check if watcher is already running
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process exists
            os.kill(pid, 0)
            logger.error("❌ Watcher already running")
            sys.exit(1)
        except:
            # PID file exists but process is dead
            PID_FILE.unlink()

    watcher = Watcher(mode=args.mode)
    watcher.start()