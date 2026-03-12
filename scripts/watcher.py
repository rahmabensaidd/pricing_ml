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
LOG_FILE = PROJECT_ROOT / "data" / "pipeline_output.log"
RESULTS_DIR = PROJECT_ROOT / "data" / "pipeline_results"


class SQLFileHandler(FileSystemEventHandler):
    def __init__(self, mode="all"):
        self.processing = False
        self.pending_files = set()
        self.last_trigger = time.time()
        self.mode = mode
        self.last_results = {}

        # Créer le dossier des résultats
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

            # Afficher un séparateur visuel
            self._print_separator()
            logger.info("📊 DÉBUT DE L'EXÉCUTION DU PIPELINE")
            self._print_separator()

            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            env['MLFLOW_TRACKING_URI'] = str(PROJECT_ROOT / 'mlruns')

            # Désactiver le buffering pour voir les logs en temps réel
            env['PYTHONUNBUFFERED'] = '1'

            # Appeler le pipeline séquentiel
            cmd = [
                "poetry", "run", "python",
                "-u",  # Unbuffered output
                "-m", "pricing_epac.flows.pricing_full_pipeline",
                "--mode", self.mode
            ]

            logger.info(f"📋 Commande: {' '.join(cmd)}")
            logger.info("⏳ Exécution en cours...\n")

            # Capturer la sortie en temps réel
            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding='utf-8',
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Variables pour capturer les résultats
            pipeline_output = []
            has_errors = False

            # Lire stdout en temps réel
            for line in process.stdout:
                line = line.rstrip()
                pipeline_output.append(line)

                # Afficher avec un préfixe visuel
                if "ERROR" in line or "❌" in line:
                    logger.error(f"❌ {line}")
                    has_errors = True
                elif "WARNING" in line or "⚠️" in line:
                    logger.warning(f"⚠️ {line}")
                elif "✅" in line or "✔️" in line:
                    logger.info(f"✅ {line}")
                elif "🏆" in line or "🏁" in line:
                    logger.info(f"🏆 {line}")
                elif "📊" in line:
                    logger.info(f"📊 {line}")
                else:
                    logger.info(f"   {line}")  # Indentation pour les logs normaux

            # Lire stderr
            for line in process.stderr:
                line = line.rstrip()
                pipeline_output.append(f"ERR: {line}")
                logger.error(f"🔴 {line}")
                has_errors = True

            return_code = process.wait(timeout=14400)  # 4 heures max

            self._print_separator()

            if return_code == 0 and not has_errors:
                logger.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS!")

                # Extraire et afficher les résultats du pipeline
                self._display_pipeline_results(pipeline_output)

                # Sauvegarder les résultats dans un fichier
                self._save_results(pipeline_output)

            else:
                logger.error(f"❌ PIPELINE ÉCHOUÉ (code {return_code})")
                if has_errors:
                    logger.error("   Des erreurs ont été détectées pendant l'exécution")

            self._print_separator()

        except subprocess.TimeoutExpired:
            logger.error("❌ Pipeline timeout après 4 heures")
            process.kill()
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _print_separator(self):
        """Affiche un séparateur visuel"""
        separator = "=" * 80
        logger.info(separator)

    def _display_pipeline_results(self, output_lines):
        """Extrait et affiche les résultats importants du pipeline"""

        # Patterns à rechercher dans les logs
        important_patterns = [
            "🏆 MEILLEUR MODÈLE",
            "📋 CLASSEMENT",
            "✅ Modèle sauvegardé",
            "📦 ENREGISTREMENT DANS MLFLOW",
            "📊 EXEMPLES DE PRÉDICTIONS",
            "📈 STATISTIQUES",
            "🎯 Modèle global entraîné",
            "✅ Tous les modèles ont été entraînés"
        ]

        logger.info("\n📋 RÉSULTATS DU PIPELINE:")
        self._print_separator()

        results_found = False

        for line in output_lines:
            for pattern in important_patterns:
                if pattern in line:
                    logger.info(f"   {line}")
                    results_found = True
                    break

        if not results_found:
            logger.info("   Aucun résultat spécifique trouvé dans les logs")

        # Chercher le meilleur modèle et ses métriques
        best_model = None
        best_rmse = None

        for line in output_lines:
            if "🏆 MEILLEUR MODÈLE" in line:
                best_model = line
            if "RMSE final" in line:
                best_rmse = line

        if best_model and best_rmse:
            logger.info("\n🎯 RÉSUMÉ:")
            logger.info(f"   {best_model}")
            logger.info(f"   {best_rmse}")

    def _save_results(self, output_lines):
        """Sauvegarde les résultats dans un fichier"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"pipeline_result_{timestamp}.txt"

        # Sauvegarder tous les logs
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline execution at {datetime.now().isoformat()}\n")
            f.write(f"Mode: {self.mode}\n")
            f.write("=" * 80 + "\n")
            for line in output_lines:
                f.write(line + "\n")

        logger.info(f"💾 Logs complets sauvegardés dans: {result_file}")

        # Extraire et sauvegarder les métriques dans un JSON
        metrics = self._extract_metrics(output_lines)
        if metrics:
            metrics_file = RESULTS_DIR / f"metrics_{timestamp}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"📊 Métriques sauvegardées dans: {metrics_file}")

    def _extract_metrics(self, output_lines):
        """Extrait les métriques des logs"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "best_model": None,
            "rmse": None,
            "r2": None,
            "models": []
        }

        for line in output_lines:
            if "🏆 MEILLEUR MODÈLE" in line:
                metrics["best_model"] = line.split(":")[-1].strip() if ":" in line else line
            elif "RMSE final" in line:
                try:
                    metrics["rmse"] = float(line.split("=")[-1].strip().replace("€", "").strip())
                except:
                    pass
            elif "R²" in line and ":" in line:
                try:
                    if metrics["r2"] is None:
                        metrics["r2"] = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "📋 CLASSEMENT" in line:
                # On va capturer les modèles après
                pass
            elif any(m in line for m in ["RandomForest", "XGBoost", "LightGBM"]):
                if "RMSE" in line:
                    metrics["models"].append(line.strip())

        return metrics


class Watcher:
    def __init__(self, watch_path=None, mode="all"):
        self.watch_path = watch_path or SQL_FOLDER
        self.watch_path.mkdir(parents=True, exist_ok=True)
        self.observer = Observer()
        self.handler = SQLFileHandler(mode)
        self.mode = mode

        # Configuration du logging pour le fichier
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)

    def start(self):
        self._print_banner()

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
        logger.info("📁 Les résultats seront affichés ici en temps réel")
        logger.info("💾 Logs sauvegardés dans: %s", LOG_FILE)

        # Configuration du handler pour Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def signal_handler(self, sig, frame):
        logger.info("🛑 Arrêt demandé...")
        self.stop()

    def stop(self):
        self.observer.stop()
        self.observer.join()
        for f in [PID_FILE, MODE_FILE]:
            if f.exists():
                f.unlink()

        self._print_banner()
        logger.info("👋 Watcher arrêté")
        logger.info("📁 Derniers résultats disponibles dans: %s", RESULTS_DIR)
        logger.info("📊 Logs complets: %s", LOG_FILE)
        sys.exit(0)

    def _print_banner(self):
        banner = f"""
{'=' * 60}
🚀 WATCHER PIPELINE PRICING (MODE SÉQUENTIEL)
{'=' * 60}
📁 Dossier surveillé: {self.watch_path}
⚡ Mode: {self.mode}
📊 Résultats: {RESULTS_DIR}
📝 Logs: {LOG_FILE}
{'=' * 60}
"""
        logger.info(banner)


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
        logger.info("👉 Utilisation: poetry run python scripts/watcher.py --mode all")
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