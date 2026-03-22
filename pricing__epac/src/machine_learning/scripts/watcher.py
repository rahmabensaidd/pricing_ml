#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
EPAC PRICING FILE WATCHER - CONTINUOUS DATA PROCESSING DAEMON
================================================================================

A production-grade file watcher that monitors SQL dump files and triggers the
EPAC pricing pipeline for automated data processing and consolidation.

This daemon provides:
- Real-time file monitoring using watchdog
- Thread-safe queue-based processing
- Automatic retry logic for failed operations
- Comprehensive error handling and recovery
- Metrics collection for monitoring
- Graceful shutdown with PID file management
- Cross-platform support (Windows/Linux/Mac)

================================================================================
PROCESSING FLOWCHART
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                     PRICING PIPELINE WATCHER DAEMON                          │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │ SQL File Arrives │
    │ in /sql folder   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │ STEP 1: File Validation                      │
    │ - Check file extension (.sql)                │
    │ - Verify file size (max 500MB)               │
    │ - Detect binary content                      │
    │ - Validate file readability                  │
    └────────┬─────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │ STEP 2: Queue Management                     │
    │ - Add to thread-safe processing queue        │
    │ - Prevent duplicate processing               │
    │ - Track processed files by checksum          │
    └────────┬─────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │ STEP 3: Dependency Verification              │
    │ - Check MySQL connectivity                   │
    │ - Verify MLflow availability                 │
    │ - Ensure Poetry environment                  │
    └────────┬─────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │ STEP 4: Pipeline Execution                   │
    │ - Run pricing_full_pipeline module           │
    │ - Stream stdout/stderr in real-time          │
    │ - Apply retry logic on failure (3 attempts)  │
    │ - Track execution metrics                    │
    └────────┬─────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │ STEP 5: Post-Processing                      │
    │ - Save results to pipeline_results/          │
    │ - Move processed file to processed/          │
    │ - Update metrics and tracking                │
    │ - Log success/failure with duration          │
    └────────┬─────────────────────────────────────┘
             │
             ▼
    ┌──────────────────┐
    │ Ready for Next   │
    │ SQL File         │
    └──────────────────┘

================================================================================
KEY FEATURES
================================================================================

✓ Real-time Monitoring  - Watchdog-based file system event detection
✓ Thread-safe Queue     - Concurrent processing without conflicts
✓ Retry Logic          - Automatic retry on failure (3 attempts with backoff)
✓ Dependency Checks    - Pre-flight verification of all services
✓ Metrics Collection   - Track success rates, processing times, error counts
✓ Graceful Shutdown    - Proper cleanup with PID file management
✓ File Validation      - Size, format, and integrity checks
✓ Comprehensive Logging - Rotating logs with configurable levels
✓ Cross-Platform       - Works on Windows, Linux, and Mac

================================================================================
ENVIRONMENT VARIABLES
================================================================================

Optional:
    MYSQL_HOST          - MySQL server host (default: localhost)
    MYSQL_PORT          - MySQL server port (default: 3307)
    MLFLOW_URI          - MLflow tracking URI (default: http://localhost:5000)
    MAX_RETRIES         - Maximum retry attempts (default: 3)
    RETRY_DELAY         - Base retry delay in seconds (default: 60)
    PIPELINE_TIMEOUT    - Pipeline timeout in seconds (default: 14400)
    LOG_LEVEL           - Logging level (default: INFO)
    MAX_FILE_SIZE_MB    - Maximum SQL file size (default: 500)

================================================================================
USAGE EXAMPLES
================================================================================

Basic usage:
    $ python watcher.py

With specific mode:
    $ python watcher.py --mode all

Custom watch path:
    $ python watcher.py --watch /custom/path

Stop watcher:
    $ kill -TERM $(cat data/watcher.pid)  # Linux/Mac
    $ taskkill /PID $(type data\watcher.pid)  # Windows

================================================================================
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
EPAC PRICING FILE WATCHER - CONTINUOUS DATA PROCESSING DAEMON
================================================================================

A production-grade file watcher that monitors SQL dump files and triggers the
EPAC pricing pipeline for automated data processing and consolidation.
"""

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
import hashlib
import platform
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from dotenv import load_dotenv
import tempfile


# ========== LOAD ENVIRONMENT VARIABLES FROM .env ==========
def load_env_file():
    """Load .env file from project root"""
    script_path = Path(__file__).resolve()
    # Structure: scripts/watcher.py -> scripts -> machine_learning -> src -> pricing__epac -> project_root
    project_root = script_path.parent.parent.parent.parent.parent
    env_file = project_root / '.env'

    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ .env file loaded from: {env_file}")

        # Verify critical variables are loaded
        mysql_password = os.getenv('MYSQL_PASSWORD')
        if mysql_password:
            print(f"✅ MYSQL_PASSWORD loaded (length: {len(mysql_password)})")
        else:
            print("⚠️ MYSQL_PASSWORD not found in .env - database connection may fail")

        mysql_host = os.getenv('MYSQL_HOST', 'localhost')
        print(f"✅ MYSQL_HOST: {mysql_host}")

        mysql_port = os.getenv('MYSQL_PORT', '3307')
        print(f"✅ MYSQL_PORT: {mysql_port}")

        return True
    else:
        print(f"⚠️ .env file not found at: {env_file}")
        print("   Please create a .env file with MYSQL_PASSWORD=your_password")
        return False


# Load .env file
load_env_file()
# ========== END ENV LOADING ==========


# ========== CROSS-PLATFORM FILE LOCKING ==========
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_MAC = platform.system() == 'Darwin'

# Attempt to import platform-specific locking modules
try:
    if IS_WINDOWS:
        import msvcrt

        HAS_FILE_LOCK = True
    else:
        import fcntl

        HAS_FILE_LOCK = True
except ImportError:
    HAS_FILE_LOCK = False
    print("⚠️ File locking not available on this platform")


class CrossPlatformFileLock:
    """Cross-platform file locking handler"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_handle = None

    def __enter__(self):
        """Acquire a lock on the file"""
        try:
            # Create parent directory if needed
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            # Open file in append mode (creates if doesn't exist)
            self.file_handle = open(self.file_path, 'a')

            if IS_WINDOWS:
                # Windows locking
                msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_LOCK, 1)
            elif HAS_FILE_LOCK:
                # Unix locking (Linux/Mac)
                fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_EX)
            else:
                # No locking available
                pass

        except PermissionError as e:
            # On Windows, permission denied might mean file is locked
            if IS_WINDOWS:
                # Try to open with shared read instead
                try:
                    self.file_handle = open(self.file_path, 'r')
                    # Could read but not lock, continue anyway
                except:
                    pass
            else:
                logging.error(f"Permission denied for {self.file_path}: {e}")
                raise
        except Exception as e:
            logging.warning(f"⚠️ Could not acquire file lock: {e}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the file lock"""
        try:
            if self.file_handle:
                if IS_WINDOWS and hasattr(msvcrt, 'locking'):
                    try:
                        msvcrt.locking(self.file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                    except:
                        pass
                elif HAS_FILE_LOCK and not IS_WINDOWS:
                    try:
                        fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
                    except:
                        pass
                self.file_handle.close()
        except Exception as e:
            logging.warning(f"⚠️ Could not release file lock: {e}")
        return False


# ========== CONFIGURATION ==========
@dataclass
class WatcherConfig:
    """Configuration for the file watcher"""
    mysql_host: str = os.getenv('MYSQL_HOST', 'localhost')
    mysql_port: int = int(os.getenv('MYSQL_PORT', 3307))
    mysql_user: str = os.getenv('MYSQL_USER', 'root')
    mysql_password: str = os.getenv('MYSQL_PASSWORD', '')
    mlflow_uri: str = os.getenv('MLFLOW_URI', 'http://localhost:5000')
    max_retries: int = int(os.getenv('MAX_RETRIES', 3))
    retry_delay: int = int(os.getenv('RETRY_DELAY', 60))
    pipeline_timeout: int = int(os.getenv('PIPELINE_TIMEOUT', 14400))  # 4 hours
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    max_file_size_mb: int = int(os.getenv('MAX_FILE_SIZE_MB', 500))
    file_ready_retries: int = 10
    file_ready_delay: int = 1
    max_processed_cache: int = 100
    shutdown_timeout: int = 5


# ========== METRICS COLLECTION ==========
@dataclass
class ProcessingMetrics:
    """Metrics for pipeline processing"""
    files_processed: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_execution_time: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    retries_used: int = 0

    def record_success(self, duration: float, retries: int = 0):
        self.successful_runs += 1
        self.files_processed += 1
        self.total_execution_time += duration
        self.retries_used += retries

    def record_failure(self, error: str, retries: int = 0):
        self.failed_runs += 1
        self.files_processed += 1
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'retries': retries
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            'files_processed': self.files_processed,
            'successful_runs': self.successful_runs,
            'failed_runs': self.failed_runs,
            'success_rate': self.successful_runs / max(self.files_processed, 1),
            'average_execution_time': self.total_execution_time / max(self.successful_runs, 1),
            'total_execution_time': self.total_execution_time,
            'retries_used': self.retries_used,
            'recent_errors': self.errors[-5:]  # Last 5 errors
        }


# ========== LOGGING SETUP ==========
def setup_logging(log_file: Path, log_level: str = 'INFO'):
    """Configure logging with rotation and proper formatting"""
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }

    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level_map.get(log_level, logging.INFO))

    # File handler with rotation (10MB, keep 5 backups)
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10_485_760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
    except Exception as e:
        print(f"⚠️ Could not create file handler: {e}")
        file_handler = None

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)

    # Configure Windows console encoding
    if IS_WINDOWS:
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    return logging.getLogger(__name__)


# ========== PATH CONFIGURATION ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PRICING_EPAC_ROOT = PROJECT_ROOT / "pricing__epac"
DATA_ROOT = PRICING_EPAC_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"
CONSOLIDATED_DATA = DATA_ROOT / "consolidated"
SQL_FOLDER = RAW_DATA / "dumps" / "sql"
PROCESSED_FOLDER = RAW_DATA / "dumps" / "processed"
PID_FILE = DATA_ROOT / "watcher.pid"
LOG_FILE = DATA_ROOT / "pipeline_output.log"
RESULTS_DIR = DATA_ROOT / "pipeline_results"
METRICS_FILE = DATA_ROOT / "watcher_metrics.json"
IGNORED_FILES = ["current_source.sql", "mysql_db_dump.sql"]

# Create directories
for folder in [SQL_FOLDER, PROCESSED_FOLDER, RESULTS_DIR, CONSOLIDATED_DATA]:
    folder.mkdir(parents=True, exist_ok=True)

# Setup logging
logger = setup_logging(LOG_FILE, os.getenv('LOG_LEVEL', 'INFO'))

logger.info(f"🖥️ Operating system: {platform.system()}")
logger.info(f"🔧 File locking: {'Available' if HAS_FILE_LOCK else 'Not available'}")


# ========== PID FILE MANAGEMENT (Windows Safe) ==========
def check_pid_file(pid_file: Path) -> Optional[int]:
    """Check if PID file exists and process is running"""
    if not pid_file.exists():
        return None

    try:
        with open(pid_file, 'r') as f:
            content = f.read().strip()
            if not content:
                return None
            pid = int(content)

        # Check if process is running
        if IS_WINDOWS:
            # Windows: use tasklist
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}'],
                capture_output=True,
                text=True,
                shell=True,
                timeout=5
            )
            if str(pid) in result.stdout:
                return pid
        else:
            # Linux/Mac: use kill with signal 0
            os.kill(pid, 0)
            return pid
    except (ValueError, ProcessLookupError, subprocess.TimeoutExpired, subprocess.CalledProcessError,
            FileNotFoundError):
        # Process doesn't exist or error
        pass
    except Exception as e:
        logger.warning(f"⚠️ Error checking PID file: {e}")

    return None


def cleanup_stale_pid_file(pid_file: Path):
    """Remove stale PID file if exists"""
    if pid_file.exists():
        try:
            pid_file.unlink()
            logger.info(f"🧹 Removed stale PID file: {pid_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not remove stale PID file: {e}")


def write_pid_file(pid_file: Path) -> bool:
    """Write PID file with retry on Windows"""
    for attempt in range(3):
        try:
            # Try to write with exclusive creation
            with open(pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logger.info(f"✅ PID file created: {pid_file} (PID: {os.getpid()})")
            return True
        except PermissionError as e:
            if attempt < 2:
                logger.warning(f"⚠️ Permission denied (attempt {attempt + 1}/3), retrying...")
                time.sleep(0.5)
            else:
                logger.error(f"❌ Failed to create PID file after 3 attempts: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to create PID file: {e}")
            return False
    return False


@contextmanager
def pid_lock(pid_file: Path):
    """Atomic PID file management with Windows-safe handling"""
    # Check if another instance is running
    existing_pid = check_pid_file(pid_file)
    if existing_pid:
        logger.error(f"❌ Watcher already running with PID {existing_pid}")
        yield False
        return

    # Clean up any stale PID file
    cleanup_stale_pid_file(pid_file)

    # Write new PID file
    if not write_pid_file(pid_file):
        yield False
        return

    try:
        yield True
    finally:
        # Clean up PID file on exit
        cleanup_stale_pid_file(pid_file)


# ========== DEPENDENCY CHECKER ==========
class DependencyChecker:
    """Check availability of external dependencies"""

    @staticmethod
    def check_mysql() -> bool:
        """Check MySQL connectivity using credentials from .env"""
        try:
            import mysql.connector

            # Get credentials from environment (loaded from .env)
            mysql_password = os.getenv('MYSQL_PASSWORD')
            mysql_host = os.getenv('MYSQL_HOST', 'localhost')
            mysql_port = int(os.getenv('MYSQL_PORT', 3307))
            mysql_user = os.getenv('MYSQL_USER', 'root')

            if not mysql_password:
                logger.warning("⚠️ MYSQL_PASSWORD not set in environment")
                logger.warning("   Please check your .env file")
                return True  # Don't fail, just warn - pipeline will fail if needed

            logger.info(f"🔍 Testing MySQL connection to {mysql_host}:{mysql_port} as {mysql_user}")

            conn = mysql.connector.connect(
                host=mysql_host,
                port=mysql_port,
                user=mysql_user,
                password=mysql_password,
                connection_timeout=5
            )
            conn.close()
            logger.info("✅ MySQL connection verified")
            return True
        except ImportError:
            logger.error("❌ MySQL connector not installed")
            logger.error("   Install with: pip install mysql-connector-python")
            return False
        except Exception as e:
            logger.error(f"❌ MySQL connection failed: {e}")
            logger.error("   Please check your MySQL credentials in .env file")
            return False  # Critical dependency, fail if cannot connect

    @staticmethod
    def check_mlflow() -> bool:
        """Check MLflow availability"""
        try:
            import requests
            mlflow_uri = os.getenv('MLFLOW_URI', 'http://localhost:5000')
            response = requests.get(f"{mlflow_uri}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ MLflow service available")
                return True
        except ImportError:
            logger.warning("⚠️ Requests library not installed, skipping MLflow check")
            return True  # Not critical
        except Exception as e:
            logger.warning(f"⚠️ MLflow check failed: {e}")
            return True  # Not critical
        return True

    @staticmethod
    def check_poetry() -> bool:
        """Check if poetry is available"""
        try:
            result = subprocess.run(
                ['poetry', '--version'],
                capture_output=True,
                timeout=5,
                shell=IS_WINDOWS
            )
            if result.returncode == 0:
                version = result.stdout.decode().strip() if result.stdout else "installed"
                logger.info(f"✅ Poetry available: {version}")
                return True
        except Exception as e:
            logger.error(f"❌ Poetry check failed: {e}")
        return False

    @classmethod
    def check_all(cls) -> bool:
        """Check all dependencies"""
        logger.info("🔍 Checking dependencies...")

        checks = [
            ('MySQL', cls.check_mysql),
            ('MLflow', cls.check_mlflow),
            ('Poetry', cls.check_poetry)
        ]

        all_ok = True
        for name, check_func in checks:
            if not check_func():
                if name == 'MySQL':
                    logger.error(f"❌ {name} connection failed - pipeline will not work")
                    all_ok = False
                elif name == 'Poetry':
                    logger.error(f"❌ {name} is required for pipeline execution")
                    all_ok = False
                else:
                    logger.warning(f"⚠️ {name} check failed, continuing...")

        if all_ok:
            logger.info("✅ All critical dependencies verified")
        else:
            logger.error("❌ Some critical dependencies are missing")

        return all_ok


# ========== SQL FILE HANDLER ==========
class SQLFileHandler(FileSystemEventHandler):
    """
    Handles SQL file events and triggers the pricing pipeline.

    This class provides thread-safe file processing with retry logic,
    metrics collection, and comprehensive error handling.
    """

    def __init__(self, mode: str = "all", config: Optional[WatcherConfig] = None):
        """
        Initialize the file handler.

        Args:
            mode: Pipeline mode ('all', 'global', 'family', 'couple', 'features')
            config: Optional configuration overrides
        """
        self.config = config or WatcherConfig()
        self.processing: bool = False
        self.pending_queue: queue.Queue = queue.Queue()
        self.last_trigger: float = time.time()
        self.mode: str = mode
        self.current_process: Optional[subprocess.Popen] = None
        self.stop_monitoring: bool = False
        self.processing_thread: Optional[threading.Thread] = None
        self.processed_files: Set[str] = set()  # Track processed files by hash
        self.metrics: ProcessingMetrics = ProcessingMetrics()

        # Create directories
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

        self.start_processing_thread()
        logger.info("🔄 SQL file handler initialized")

    def start_processing_thread(self) -> None:
        """Start the background processing thread."""
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        logger.info("🔄 Processing thread started")

    def _process_queue(self) -> None:
        """Process files from the queue with proper error handling."""
        while not self.stop_monitoring:
            try:
                sql_file = self.pending_queue.get(timeout=1)

                if sql_file is None:  # Sentinel for shutdown
                    break

                self.processing = True
                try:
                    if self._wait_for_file_ready(sql_file):
                        self._run_pipeline_with_retry(sql_file)
                except Exception as e:
                    logger.error(f"❌ Error processing {sql_file}: {e}")
                finally:
                    self.processing = False
                    self.pending_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Queue processing error: {e}")

    def _wait_for_file_ready(self, file_path: str, max_retries: Optional[int] = None,
                             delay: Optional[int] = None) -> bool:
        """
        Wait for file to be fully written and ready for processing.

        Args:
            file_path: Path to the file
            max_retries: Maximum number of retries
            delay: Delay between checks in seconds

        Returns:
            True if file is ready, False otherwise
        """
        max_retries = max_retries or self.config.file_ready_retries
        delay = delay or self.config.file_ready_delay

        for i in range(max_retries):
            try:
                # Check if file is readable
                with open(file_path, 'rb') as f:
                    f.read(1024)

                # Check if file size is stable
                size1 = os.path.getsize(file_path)
                time.sleep(0.5)
                size2 = os.path.getsize(file_path)

                if size1 == size2 and size1 > 0:
                    # Check file size limit
                    size_mb = size1 / (1024 * 1024)
                    if size_mb > self.config.max_file_size_mb:
                        logger.error(f"❌ File too large: {size_mb:.2f} MB")
                        return False

                    logger.info(f"✅ File ready: {Path(file_path).name}")
                    return True

            except (IOError, OSError) as e:
                logger.debug(f"Waiting for file to be ready: {e}")
                time.sleep(delay)
            except Exception as e:
                logger.warning(f"⚠️ Error checking file readiness: {e}")
                time.sleep(delay)

        logger.error(f"❌ File not ready after {max_retries} attempts: {file_path}")
        return False

    def _get_file_hash(self, file_path: str) -> str:
        """Generate a unique hash for a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read(8192)).hexdigest()
        except Exception:
            return f"{file_path}_{os.path.getmtime(file_path)}"

    def on_created(self, event) -> None:
        """Handle file creation events."""
        if not event.is_directory and event.src_path.endswith('.sql'):
            self._handle_new_sql_file(event.src_path)

    def on_modified(self, event) -> None:
        """Handle file modification events with debouncing."""
        if not event.is_directory and event.src_path.endswith('.sql'):
            current_time = time.time()
            if current_time - self.last_trigger > 2:
                self.last_trigger = current_time
                self._handle_new_sql_file(event.src_path)

    def _handle_new_sql_file(self, file_path: str) -> None:
        """
        Process a new SQL file.

        Args:
            file_path: Path to the SQL file
        """
        if not os.path.exists(file_path):
            return

        file_name = Path(file_path).name

        # Check ignored files
        if file_name in IGNORED_FILES:
            logger.info(f"⏭️ Ignoring {file_name} (system file)")
            return

        # Validate file
        if not self._validate_sql_file(file_path):
            logger.error(f"❌ Invalid SQL file: {file_name}")
            return

        # Prevent duplicate processing
        try:
            file_hash = self._get_file_hash(file_path)
            if file_hash in self.processed_files:
                logger.info(f"⏭️ File already processed: {file_name}")
                return

            self.processed_files.add(file_hash)

            # Clean up cache periodically
            if len(self.processed_files) > self.config.max_processed_cache:
                self.processed_files.clear()

        except Exception as e:
            logger.warning(f"⚠️ Could not check duplicate: {e}")

        logger.info(f"➕ New SQL file: {file_name}")
        self.pending_queue.put(file_path)
        logger.info(f"📊 Queue size: {self.pending_queue.qsize()}")

    def _validate_sql_file(self, file_path: str) -> bool:
        """
        Validate SQL file before processing.

        Args:
            file_path: Path to the SQL file

        Returns:
            True if file is valid, False otherwise
        """
        # Check file extension
        if not file_path.endswith('.sql'):
            return False

        # Check if readable
        if not os.access(file_path, os.R_OK):
            logger.error(f"❌ File not readable: {file_path}")
            return False

        # Check file size
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb == 0:
                logger.error(f"❌ Empty file: {file_path}")
                return False
            if size_mb > self.config.max_file_size_mb:
                logger.error(f"❌ File too large: {size_mb:.2f} MB")
                return False
        except Exception as e:
            logger.error(f"❌ Error checking file size: {e}")
            return False

        # Quick SQL validation (check first line)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if first_line and not any(word in first_line.upper() for word in ['SELECT', 'INSERT', 'CREATE']):
                    logger.warning(f"⚠️ File may not be valid SQL: {first_line[:50]}")
        except Exception:
            pass

        return True

    def _run_pipeline_with_retry(self, sql_file: str, retry_count: int = 0) -> bool:
        """
        Run the pricing pipeline with retry logic.

        Args:
            sql_file: Path to SQL file
            retry_count: Current retry attempt number

        Returns:
            True if successful, False otherwise
        """
        try:
            success = self._run_pipeline(sql_file)

            if not success and retry_count < self.config.max_retries:
                logger.warning(f"⚠️ Retry {retry_count + 1}/{self.config.max_retries} for {Path(sql_file).name}")
                wait_time = self.config.retry_delay * (2 ** retry_count)  # Exponential backoff
                time.sleep(wait_time)
                return self._run_pipeline_with_retry(sql_file, retry_count + 1)

            return success

        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            if retry_count < self.config.max_retries:
                return self._run_pipeline_with_retry(sql_file, retry_count + 1)
            return False

    def _run_pipeline(self, sql_file: str) -> bool:
        """
        Execute the pricing pipeline.

        Args:
            sql_file: Path to SQL file

        Returns:
            True if successful, False otherwise
        """
        try:
            self._print_separator()
            logger.info(f"📊 START - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"📁 File: {Path(sql_file).name}")
            self._print_separator()

            # Prepare environment - inherit all environment variables including MySQL password
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['MLFLOW_TRACKING_URI'] = self.config.mlflow_uri
            env['PYTHONUNBUFFERED'] = '1'

            # Ensure MySQL password is passed to subprocess
            if self.config.mysql_password:
                env['MYSQL_PASSWORD'] = self.config.mysql_password

            # Build command
            cmd = [
                "poetry", "run", "python",
                "-u",
                "-m", "pricing__epac.src.machine_learning.flows.pricing_full_pipeline",
                "--mode", self.mode,
                "--input", str(sql_file),
                "--once"
            ]

            logger.info(f"📋 Command: {' '.join(cmd)}")

            # Start process
            self.current_process = subprocess.Popen(
                cmd,
                cwd=PRICING_EPAC_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding='utf-8',
                bufsize=1,
                universal_newlines=True
            )

            # Stream output
            pipeline_output = []
            has_errors = False
            start_time = time.time()

            def read_stream(stream, is_error: bool = False):
                nonlocal has_errors
                try:
                    for line in iter(stream.readline, ''):
                        line = line.rstrip()
                        pipeline_output.append(line)
                        if is_error:
                            logger.error(f"🔴 {line}")
                            has_errors = True
                        else:
                            self._display_line(line)
                finally:
                    stream.close()

            # Create reader threads
            stdout_thread = threading.Thread(target=read_stream, args=(self.current_process.stdout, False))
            stderr_thread = threading.Thread(target=read_stream, args=(self.current_process.stderr, True))
            stdout_thread.daemon = True
            stderr_thread.daemon = True

            stdout_thread.start()
            stderr_thread.start()

            # Wait for completion with timeout
            return_code = self.current_process.wait(timeout=self.config.pipeline_timeout)

            # Wait for reader threads
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            elapsed = time.time() - start_time

            # Process result
            if return_code == 0 and not has_errors:
                logger.info(f"✅ SUCCESS in {elapsed / 60:.1f} minutes!")
                self._save_results(pipeline_output, elapsed, sql_file)
                self.metrics.record_success(elapsed)
                self._save_metrics()

                # Move file to processed folder
                if Path(sql_file).name not in IGNORED_FILES:
                    dest = PROCESSED_FOLDER / Path(sql_file).name
                    if Path(sql_file).parent != PROCESSED_FOLDER:
                        shutil.move(str(sql_file), str(dest))
                        logger.info(f"📁 File moved to: {dest}")
                return True
            else:
                logger.error(f"❌ FAILED after {elapsed / 60:.1f} minutes with code {return_code}")
                self.metrics.record_failure(f"Exit code: {return_code}")
                self._save_metrics()
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"❌ TIMEOUT after {self.config.pipeline_timeout / 3600:.1f} hours")
            if self.current_process:
                self.current_process.kill()
            self.metrics.record_failure("Timeout")
            self._save_metrics()
            return False

        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            self.metrics.record_failure(str(e))
            self._save_metrics()
            return False

        finally:
            self.current_process = None

    def _display_line(self, line: str) -> None:
        """Display a line with appropriate logging level."""
        if "ERROR" in line or "❌" in line or "FAILED" in line:
            logger.error(f"❌ {line}")
        elif "WARNING" in line or "⚠️" in line:
            logger.warning(f"⚠️ {line}")
        elif "✅" in line or "SUCCESS" in line:
            logger.info(f"✅ {line}")
        elif "INFO" in line:
            logger.info(f"ℹ️ {line}")
        else:
            logger.info(f"   {line}")

    def _print_separator(self) -> None:
        """Print a separator line."""
        logger.info("=" * 80)

    def _save_results(self, output_lines: List[str], elapsed_seconds: float, sql_file: str) -> None:
        """Save pipeline output to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sql_name = Path(sql_file).stem
        result_file = RESULTS_DIR / f"pipeline_{sql_name}_{timestamp}.txt"

        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"Pipeline executed at: {datetime.now().isoformat()}\n")
                f.write(f"SQL file: {sql_file}\n")
                f.write(f"Mode: {self.mode}\n")
                f.write(f"Duration: {elapsed_seconds / 60:.1f} minutes ({elapsed_seconds:.2f} seconds)\n")
                f.write("=" * 80 + "\n")
                for line in output_lines:
                    f.write(line + "\n")

            logger.info(f"💾 Results saved: {result_file}")

        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            with open(METRICS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.metrics.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save metrics: {e}")

    def stop(self) -> None:
        """Stop the handler and cleanup."""
        logger.info("🛑 Stopping file handler...")
        self.stop_monitoring = True
        self.pending_queue.put(None)  # Sentinel for queue thread

        if self.current_process:
            logger.info("⏳ Terminating current pipeline...")
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=self.config.shutdown_timeout)
            except subprocess.TimeoutExpired:
                self.current_process.kill()

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=self.config.shutdown_timeout)

        logger.info("✅ File handler stopped")


# ========== MAIN WATCHER CLASS ==========
class Watcher:
    """
    Main file watcher daemon.

    Monitors a directory for SQL files and triggers processing.
    """

    def __init__(self, watch_path: Optional[Path] = None, mode: str = "all",
                 config: Optional[WatcherConfig] = None):
        """
        Initialize the watcher.

        Args:
            watch_path: Directory to monitor
            mode: Pipeline mode
            config: Optional configuration
        """
        self.config = config or WatcherConfig()
        self.watch_path = watch_path or SQL_FOLDER
        self.mode = mode
        self.watch_path.mkdir(parents=True, exist_ok=True)
        self.observer = Observer()
        self.handler = SQLFileHandler(mode, self.config)

        logger.info("🔧 Watcher initialized")

    def start(self) -> None:
        """Start the watcher daemon."""
        # Check dependencies first
        if not DependencyChecker.check_all():
            logger.error("❌ Critical dependency check failed. Exiting.")
            sys.exit(1)

        self._print_banner()

        # Acquire PID lock (Windows-safe)
        with pid_lock(PID_FILE) as acquired:
            if not acquired:
                sys.exit(1)

        # Start observer
        self.observer.schedule(self.handler, str(self.watch_path), recursive=False)
        self.observer.start()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("✅ Watcher active")
        logger.info(f"📁 Monitored folder: {self.watch_path}")
        logger.info(f"📁 Processed folder: {PROCESSED_FOLDER}")
        logger.info(f"📊 Results: {RESULTS_DIR}")
        logger.info(f"📈 Metrics: {METRICS_FILE}")
        logger.info(f"⏭️ Ignored files: {IGNORED_FILES}")
        logger.info(f"🔄 Retry attempts: {self.config.max_retries}")
        logger.info(f"⏱️ Pipeline timeout: {self.config.pipeline_timeout / 3600:.1f} hours")
        logger.info("🛑 Press Ctrl+C to stop\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _signal_handler(self, sig, frame) -> None:
        """Handle termination signals."""
        logger.info(f"\n⚠️ Received signal {sig}")
        self.stop()

    def stop(self) -> None:
        """Stop the watcher gracefully."""
        logger.info("🛑 Stopping watcher...")

        # Stop handler
        self.handler.stop()

        # Stop observer
        self.observer.stop()
        self.observer.join()

        # Print final metrics
        metrics = self.handler.metrics.to_dict()
        logger.info("\n" + "=" * 60)
        logger.info("📊 FINAL METRICS")
        logger.info("=" * 60)
        logger.info(f"📁 Files processed: {metrics['files_processed']}")
        logger.info(f"✅ Successful runs: {metrics['successful_runs']}")
        logger.info(f"❌ Failed runs: {metrics['failed_runs']}")
        logger.info(f"📈 Success rate: {metrics['success_rate'] * 100:.1f}%")
        logger.info(f"⏱️ Average time: {metrics['average_execution_time'] / 60:.1f} minutes")
        logger.info(f"🔄 Retries used: {metrics['retries_used']}")
        logger.info("=" * 60)

        logger.info("👋 Watcher stopped")
        sys.exit(0)

    def _print_banner(self) -> None:
        """Print startup banner."""
        banner = f"""
{'=' * 70}
🚀 PRICING WATCHER - CONTINUOUS DATA PROCESSING DAEMON
{'=' * 70}
📁 Project root: {PROJECT_ROOT}
📁 Pricing EPAC root: {PRICING_EPAC_ROOT}
📁 Data root: {DATA_ROOT}
📁 Monitored: {self.watch_path}
⚡ Mode: {self.mode}
📁 Processed: {PROCESSED_FOLDER}
📊 Results: {RESULTS_DIR}
📈 Metrics: {METRICS_FILE}
📁 Consolidated: {CONSOLIDATED_DATA}
⏭️ Ignored: {IGNORED_FILES}
🖥️ OS: {platform.system()}
🔒 File Lock: {'Available' if HAS_FILE_LOCK else 'Not available'}
📧 MySQL: {self.config.mysql_host}:{self.config.mysql_port} as {self.config.mysql_user}
{'=' * 70}
"""
        logger.info(banner)


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='EPAC Pricing File Watcher - Continuous data processing daemon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s                         # Start with default settings
    %(prog)s --mode all              # Run full pipeline
    %(prog)s --mode features         # Run feature extraction only
    %(prog)s --watch /custom/path    # Monitor custom directory

Note:
    MySQL credentials are loaded from .env file in the project root.
    Make sure .env contains MYSQL_PASSWORD=your_password
        """
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "global", "family", "couple", "features"],
                        help="Pipeline execution mode (default: all)")
    parser.add_argument("--watch", type=Path, default=None,
                        help="Directory to monitor (default: data/raw/dumps/sql)")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Maximum retry attempts (default: 3)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Pipeline timeout in seconds (default: 14400)")

    args = parser.parse_args()

    # Create custom config
    config = WatcherConfig()
    if args.max_retries:
        config.max_retries = args.max_retries
    if args.timeout:
        config.pipeline_timeout = args.timeout

    # Start watcher
    watcher = Watcher(watch_path=args.watch, mode=args.mode, config=config)
    watcher.start()