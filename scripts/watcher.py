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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_FOLDER = PROJECT_ROOT / "data" / "raw" / "dumps" / "sql"
PID_FILE = PROJECT_ROOT / "data" / "watcher.pid"
MODE_FILE = PROJECT_ROOT / "data" / "watcher_mode.txt"


class SQLFileHandler(FileSystemEventHandler):
    def __init__(self, mode="all"):
        self.processing = False
        self.pending_files = set()
        self.last_trigger = time.time()
        self.mode = mode

    def wait_for_file_ready(self, file_path, max_retries=10, delay=1):
        for i in range(max_retries):
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)
                size1 = os.path.getsize(file_path)
                time.sleep(0.5)
                size2 = os.path.getsize(file_path)
                if size1 == size2:
                    logger.info(f"✅ Fichier prêt: {Path(file_path).name} ({size1} bytes)")
                    return True
            except (IOError, OSError) as e:
                logger.debug(f"⏳ Fichier pas encore prêt ({i + 1}/{max_retries}): {e}")
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
        if self.processing:
            self.pending_files.add(file_path)
            logger.info(f"⏳ Pipeline en cours, fichier mis en attente: {Path(file_path).name}")
            return

        self.processing = True
        try:
            logger.info(f"🎯 Nouveau fichier SQL détecté: {Path(file_path).name}")
            if not self.wait_for_file_ready(file_path):
                logger.error(f"❌ Impossible d'accéder au fichier {Path(file_path).name}")
                return
            self.run_pipeline()
            while self.pending_files:
                time.sleep(2)
                self.pending_files.clear()
                self.run_pipeline()
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
        finally:
            self.processing = False

    def run_pipeline(self):
        try:
            logger.info(f"🚀 Lancement du pipeline en mode: {self.mode} (séquentiel)...")
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            env['MLFLOW_TRACKING_URI'] = str(PROJECT_ROOT / 'mlruns')

            # Appeler le pipeline séquentiel
            cmd = [
                "poetry", "run", "python",
                "-m", "pricing_epac.flows.pricing_full_pipeline",
                "--mode", self.mode,
                "--once"
            ]

            logger.info(f"📋 Commande: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding='utf-8',
                bufsize=1
            )

            for line in process.stdout:
                logger.info(f"[PIPELINE] {line.rstrip()}")
            for line in process.stderr:
                logger.error(f"[PIPELINE-ERR] {line.rstrip()}")

            return_code = process.wait(timeout=14400)  # 4 heures max

            if return_code == 0:
                logger.info(f"✅ Pipeline ({self.mode}) terminé avec succès")
                logger.info(f"📊 Résultats: poetry run mlflow ui --port 5000")
            else:
                logger.error(f"❌ Pipeline échoué (code {return_code})")

        except Exception as e:
            logger.error(f"❌ Erreur: {e}")


class Watcher:
    def __init__(self, watch_path=None, mode="all"):
        self.watch_path = watch_path or SQL_FOLDER
        self.watch_path.mkdir(parents=True, exist_ok=True)
        self.observer = Observer()
        self.handler = SQLFileHandler(mode)
        self.mode = mode

    def start(self):
        logger.info("=" * 60)
        logger.info("🚀 DÉMARRAGE DU WATCHER (MODE SÉQUENTIEL)")
        logger.info(f"📁 Dossier: {self.watch_path}")
        logger.info(f"⚡ Mode: {self.mode}")
        logger.info("=" * 60)

        if sys.platform == 'win32':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleOutputCP(65001)
                kernel32.SetConsoleCP(65001)
            except:
                pass

        self.observer.schedule(self.handler, str(self.watch_path), recursive=False)

        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        with open(MODE_FILE, 'w') as f:
            f.write(self.mode)

        self.observer.start()
        logger.info("✅ Watcher actif - en attente de fichiers SQL...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.observer.stop()
        self.observer.join()
        for f in [PID_FILE, MODE_FILE]:
            if f.exists():
                f.unlink()
        logger.info("👋 Watcher arrêté")


def is_running():
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return True
        except:
            PID_FILE.unlink()
            if MODE_FILE.exists():
                MODE_FILE.unlink()
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watcher pour pipeline pricing (mode séquentiel)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "global", "family"],
                        help="Mode d'exécution")
    args = parser.parse_args()

    # Vérifier si on veut exécuter seulement le mode all
    if args.mode != "all":
        logger.error(f"❌ Ce script ne peut être exécuté qu'avec --mode all")
        logger.info("👉 Utilisation: python scripts/watcher.py --mode all")
        sys.exit(1)

    if is_running():
        logger.error("❌ Un watcher est déjà en cours")
        if MODE_FILE.exists():
            with open(MODE_FILE, 'r') as f:
                current_mode = f.read().strip()
            logger.info(f"📌 Mode actuel: {current_mode}")
        sys.exit(1)

    watcher = Watcher(mode=args.mode)
    watcher.start()