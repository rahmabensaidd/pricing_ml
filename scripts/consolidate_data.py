# scripts/consolidate_data.py
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
from pathlib import Path
import logging
import json
import shutil
import os
from datetime import datetime
import re
import time
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SQL_DUMPS_DIR = PROJECT_ROOT / "data" / "raw" / "dumps" / "sql"
STATIC_DATA = PROJECT_ROOT / "data" / "raw" / "static" / "epac_historiquee.csv"
CONSOLIDATED_FILE = PROJECT_ROOT / "data" / "consolidated" / "dataset_complet.xlsx"
TRACKING_FILE = PROJECT_ROOT / "data" / "dumps_tracking.json"


class DataConsolidator:
    """
    Version avec MySQL Docker - Fiable et robuste
    Utilise mysql-connector-python, sqlalchemy et pymysql
    """

    def __init__(self):
        # Créer les dossiers si nécessaires
        SQL_DUMPS_DIR.mkdir(parents=True, exist_ok=True)
        CONSOLIDATED_FILE.parent.mkdir(parents=True, exist_ok=True)

        self.tracking = self._load_tracking()
        self.static_df = None

        # Configuration MySQL pour Docker Compose
        self.mysql_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'root',
            'database': 'temp_epac',
            'charset': 'utf8mb4',
            'port': 3307,  # Correspond au port exposé dans docker-compose
            'use_pure': True
        }

    def _load_tracking(self):
        """Charge l'historique des fichiers déjà traités"""
        if TRACKING_FILE.exists():
            try:
                with open(TRACKING_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "last_consolidation": None,
            "last_sql_files": [],
            "stats": {
                "total_conversions": 0,
                "total_rows_processed": 0
            }
        }

    def _save_tracking(self):
        """Sauvegarde le tracking"""
        try:
            with open(TRACKING_FILE, 'w') as f:
                json.dump(self.tracking, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Impossible de sauvegarder le tracking: {e}")

    def _clean_folder(self, folder: Path, pattern: str = "*"):
        """Vide complètement un dossier"""
        if not folder.exists():
            return 0

        files = list(folder.glob(pattern))
        count = len(files)

        for file in files:
            try:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            except Exception as e:
                logger.warning(f"⚠️ Impossible de supprimer {file}: {e}")

        logger.info(f"🧹 Dossier vidé: {folder} ({count} élément(s))")
        return count

    def _check_port_open(self, host, port, timeout=2):
        """Vérifie si un port est ouvert"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False

    def _check_mysql_connection(self, retries=2, delay=3):
        """
        Vérifie la connexion MySQL avec plusieurs tentatives
        """
        # D'abord vérifier si le port est ouvert
        if not self._check_port_open(self.mysql_config['host'], self.mysql_config['port']):
            logger.warning(f"⚠️ Port {self.mysql_config['port']} fermé sur {self.mysql_config['host']}")

        for attempt in range(retries):
            try:
                conn = mysql.connector.connect(
                    host=self.mysql_config['host'],
                    user=self.mysql_config['user'],
                    password=self.mysql_config['password'],
                    port=self.mysql_config['port'],
                    connection_timeout=10,
                    use_pure=True
                )
                conn.close()
                logger.info("✅ Connexion MySQL Docker établie")
                return True
            except mysql.connector.Error as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"⏳ Tentative {attempt + 1}/{retries} - Connexion MySQL... ({e.errno}: {e.msg[:50]})")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Impossible de se connecter à MySQL Docker après {retries} tentatives: {e}")
                    logger.error("   Vérifiez que Docker est en cours d'exécution avec:")
                    logger.error("   docker-compose ps")
                    logger.error("   docker-compose logs mysql")
                    logger.error("   Test manuel: docker exec -it mysql-epac mysql -u root -p")
                    return False
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(f"⏳ Tentative {attempt + 1}/{retries} - Connexion MySQL... ({str(e)[:50]})")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Erreur inattendue: {e}")
                    return False
        return False

    def _get_sql_query(self):
        """
        Retourne la requête SQL complète pour extraire les données
        """
        return """
        SELECT
            pp.*,
            o.order_id,
            o.order_num,
            o.delivery_date,
            o.expected_date,
            o.requester_name,
            o.reception_date,
            o.quantity AS quantity,
            o.qty_min AS quantity_min,
            o.qty_max AS quantity_max,
            o.priority_level,
            c.client_id,
            c.name AS client_name,
            c.shipping_country,
            c.siren,
            c.tva,
            dc.coil_type,
            pc.cover_paper_type,
            pc.double_sided_cover,
            pc.cover_color,
            pc.head_and_tail,
            pc.cover_size,
            pi.active AS insert_active,
            pi.insert_lamination,
            pi.insert_paper_type,
            pi.insert_color,
            pi.insert_size,
            pt.active AS tab_active,
            pt.tab_page_number,
            pt.trim_size,
            pt.tab_color,
            pt.tab_lamination,
            pt.tab_size,
            pt.tab_paper_type,
            pcc.case_finish_type,
            pcc.case_paper_type,
            pcc.cover_case_color,
            pcc.back_cover_flat_size,
            pcc.spine_type
        FROM production_parts pp
        INNER JOIN orders o ON o.order_id = pp.order_id
        INNER JOIN clients c ON c.client_id = o.client_id
        LEFT JOIN part_coils dc ON dc.part_id = REPLACE(pp.part_id, 'PZ', '') AND dc.active = 1
        LEFT JOIN part_covers pc ON pc.part_id = REPLACE(pp.part_id, 'PZ', '') AND pc.active = 1
        LEFT JOIN part_tabs pt ON pt.part_id = REPLACE(pp.part_id, 'PZ', '') AND pt.active = 1
        LEFT JOIN part_inserts pi ON pi.part_id = REPLACE(pp.part_id, 'PZ','') AND pi.active = 1
        LEFT JOIN part_covers_cases pcc ON pcc.part_id = REPLACE(pp.part_id, 'PZ','') AND pcc.active = 1
        WHERE pp.split = 0
          AND NOT EXISTS (
              SELECT 1
              FROM kits k
              WHERE k.order_id = pp.order_id
                AND k.active = 1
          )
        ORDER BY o.order_id DESC;
        """

    def import_sql_to_mysql(self, sql_path: Path) -> bool:
        """
        Importe un dump SQL dans MySQL Docker
        """
        logger.info(f"🔄 Import MySQL Docker: {sql_path.name}")

        try:
            # Connexion à MySQL Docker
            conn = mysql.connector.connect(
                host=self.mysql_config['host'],
                user=self.mysql_config['user'],
                password=self.mysql_config['password'],
                port=self.mysql_config['port']
            )
            cursor = conn.cursor()

            # Désactiver les vérifications de clés étrangères
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")

            # Créer et utiliser la base temporaire
            cursor.execute(f"DROP DATABASE IF EXISTS {self.mysql_config['database']}")
            cursor.execute(f"CREATE DATABASE {self.mysql_config['database']} CHARACTER SET utf8mb4")
            cursor.execute(f"USE {self.mysql_config['database']}")

            # Lire le fichier SQL
            with open(sql_path, 'r', encoding='utf-8', errors='ignore') as f:
                sql_script = f.read()

            # Remplacer les collationnements problématiques
            sql_script = sql_script.replace('utf8mb4_0900_ai_ci', 'utf8mb4_unicode_ci')
            sql_script = sql_script.replace('utf8mb4_unicode_ci', 'utf8mb4_general_ci')

            # Supprimer les contraintes FOREIGN KEY problématiques
            sql_script = re.sub(r',\s*CONSTRAINT\s+\w+\s+FOREIGN KEY\s*\([^)]+\)\s+REFERENCES\s+\w+\s*\([^)]+\)', '',
                                sql_script)
            sql_script = re.sub(r'FOREIGN KEY\s*\([^)]+\)\s+REFERENCES\s+\w+\s*\([^)]+\)', '', sql_script)

            # Supprimer les commentaires
            sql_script = re.sub(r'--.*$', '', sql_script, flags=re.MULTILINE)

            # Supprimer les instructions DROP TABLE si présentes
            sql_script = re.sub(r'DROP TABLE IF EXISTS .*?;', '', sql_script, flags=re.IGNORECASE)

            # Diviser en instructions
            statements = []
            current_stmt = []
            in_string = False
            escape_next = False

            for char in sql_script:
                if char == "'" and not escape_next:
                    in_string = not in_string
                elif char == '\\':
                    escape_next = True
                else:
                    escape_next = False

                if char == ';' and not in_string:
                    stmt = ''.join(current_stmt).strip()
                    if stmt:
                        statements.append(stmt)
                    current_stmt = []
                else:
                    current_stmt.append(char)

            logger.info(f"📊 {len(statements)} instructions SQL trouvées")

            # Exécuter chaque instruction
            tables_created = 0
            successful_queries = 0
            failed_queries = 0

            for i, stmt in enumerate(statements):
                if not stmt:
                    continue

                try:
                    cursor.execute(stmt)
                    if stmt.upper().startswith('CREATE TABLE'):
                        tables_created += 1
                    successful_queries += 1

                    if (i + 1) % 50 == 0:
                        logger.info(f"   Exécution {i + 1}/{len(statements)} (succès: {successful_queries})")

                except Exception as e:
                    failed_queries += 1
                    if 'already exists' not in str(e).lower() and 'duplicate' not in str(e).lower():
                        logger.debug(f"   ⚠️ Instruction {i} ignorée: {str(e)[:100]}")

            # Réactiver les vérifications de clés étrangères
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"✅ Import MySQL Docker réussi: {sql_path.name}")
            logger.info(f"   📊 Tables créées: {tables_created}")
            logger.info(f"   ✅ Requêtes réussies: {successful_queries}")
            logger.info(f"   ⚠️ Requêtes ignorées: {failed_queries}")

            return True

        except Exception as e:
            logger.error(f"❌ Erreur import MySQL Docker: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def extract_from_mysql(self) -> pd.DataFrame:
        """
        Extrait les données depuis MySQL Docker avec SQLAlchemy
        """
        try:
            # Créer l'engine SQLAlchemy avec pymysql
            connection_string = (
                f"mysql+pymysql://{self.mysql_config['user']}:{self.mysql_config['password']}@"
                f"{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}"
                f"?charset=utf8mb4"
            )

            engine = create_engine(connection_string)

            # Vérifier les tables disponibles
            with engine.connect() as conn:
                result = conn.execute(text("SHOW TABLES"))
                tables = [row[0] for row in result.fetchall()]
                logger.info(f"📋 Tables disponibles après import: {tables}")

                if not tables:
                    logger.error("❌ Aucune table trouvée dans la base")
                    return None

                # Vérifier les tables essentielles
                required_tables = ['production_parts', 'orders', 'clients']
                missing_tables = [t for t in required_tables if t not in tables]

                if missing_tables:
                    logger.error(f"❌ Tables essentielles manquantes: {missing_tables}")
                    logger.error("   Le dump SQL n'a pas créé les tables nécessaires")
                    return None

            # Exécuter la requête principale
            query = self._get_sql_query()
            df = pd.read_sql(query, engine)
            logger.info(f"📊 {len(df)} lignes extraites de MySQL Docker")

            return df

        except Exception as e:
            logger.error(f"❌ Erreur extraction MySQL Docker: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def convert_sql_to_dataframe(self, sql_path: Path) -> pd.DataFrame:
        """
        Convertit un SQL en DataFrame via MySQL Docker
        """
        logger.info(f"🔄 Conversion via MySQL Docker: {sql_path.name}")

        if self.import_sql_to_mysql(sql_path):
            df = self.extract_from_mysql()
            if df is not None and not df.empty:
                self.tracking["stats"]["total_conversions"] += 1
                self.tracking["stats"]["total_rows_processed"] += len(df)
                return df
        return None

    def convert_all_sql_to_dataframes(self):
        """
        Convertit TOUS les fichiers SQL en DataFrames avec MySQL Docker
        """
        sql_files = list(SQL_DUMPS_DIR.glob("*.sql"))

        if not sql_files:
            logger.info("ℹ️ Aucun fichier SQL à convertir")
            return []

        logger.info(f"🎯 {len(sql_files)} fichier(s) SQL trouvé(s)")

        dataframes = []
        total_rows = 0

        for sql_file in sql_files:
            df = self.convert_sql_to_dataframe(sql_file)
            if df is not None and not df.empty:
                dataframes.append(df)
                total_rows += len(df)

        logger.info(f"✅ Conversion MySQL Docker terminée: {len(dataframes)} DataFrames, {total_rows} lignes totales")

        # Sauvegarder la liste des fichiers traités
        self.tracking["last_sql_files"] = [f.name for f in sql_files]
        self._save_tracking()

        return dataframes

    def load_static_data(self):
        """Charge la base statique (CSV)"""
        if not STATIC_DATA.exists():
            logger.error(f"❌ Fichier statique non trouvé: {STATIC_DATA}")
            return False

        logger.info(f"📦 Chargement base statique: {STATIC_DATA}")
        self.static_df = pd.read_csv(STATIC_DATA, encoding='utf-8')
        logger.info(f"✅ Base statique: {len(self.static_df)} lignes")
        return True

    def _diagnose_dataframes(self, dataframes):
        """Diagnostic détaillé des DataFrames"""
        logger.info("🔍 DIAGNOSTIC DÉTAILLÉ DES DATAFRAMES:")

        for i, df in enumerate(dataframes):
            if df is None:
                logger.info(f"  DataFrame {i}: None")
                continue

            logger.info(f"  DataFrame {i}:")
            logger.info(f"    - Shape: {df.shape}")
            logger.info(f"    - Index unique: {df.index.is_unique}")
            logger.info(f"    - Colonnes dupliquées: {df.columns.duplicated().any()}")
            if df.columns.duplicated().any():
                logger.warning(f"    ⚠️ Colonnes dupliquées: {df.columns[df.columns.duplicated()].tolist()}")
            logger.info(f"    - Types: {df.dtypes.value_counts().to_dict()}")
            logger.info(f"    - Valeurs nulles: {df.isnull().sum().sum()}")

    def consolidate(self) -> Path:
        """
        Consolidation avec MySQL Docker
        """
        logger.info("\n" + "=" * 60)
        logger.info("🚀 DÉMARRAGE DE LA CONSOLIDATION (MYSQL DOCKER)")
        logger.info("=" * 60)

        # Vérifier la connexion MySQL Docker d'abord
        if not self._check_mysql_connection():
            logger.error("❌ MySQL Docker n'est pas accessible. Veuillez:")
            logger.error("   1. Démarrer Docker Desktop")
            logger.error("   2. Exécuter: docker-compose up -d")
            logger.error("   3. Vérifier: docker-compose ps")
            logger.error("   4. Tester: docker exec -it mysql-epac mysql -u root -p")
            return None

        try:
            # Étape 1: Convertir tous les SQL en DataFrames avec MySQL
            logger.info("\n📥 ÉTAPE 1: Conversion SQL → DataFrames (MySQL Docker)")
            dataframes = self.convert_all_sql_to_dataframes()

            # Diagnostiquer les DataFrames
            self._diagnose_dataframes(dataframes)

            # Étape 2: Charger les données statiques (CSV)
            logger.info("\n📂 ÉTAPE 2: Chargement des données statiques (CSV)")
            self.load_static_data()

            # Étape 3: Fusionner toutes les données
            logger.info("\n🔄 ÉTAPE 3: Fusion des données")
            all_dfs = []

            # Ajouter la base statique
            if self.static_df is not None and not self.static_df.empty:
                # Réinitialiser l'index pour éviter les conflits
                static_df_reset = self.static_df.reset_index(drop=True)
                all_dfs.append(static_df_reset)
                logger.info(f"   + Statique (CSV): {len(static_df_reset)} lignes")

            # Ajouter tous les DataFrames des dumps
            for i, df in enumerate(dataframes):
                if df is not None and not df.empty:
                    # Nettoyer les colonnes dupliquées si nécessaire
                    if df.columns.duplicated().any():
                        logger.warning(f"   ⚠️ Dump SQL {i + 1} a des colonnes dupliquées, nettoyage...")
                        df = df.loc[:, ~df.columns.duplicated()]

                    # Réinitialiser l'index pour chaque DataFrame
                    df_reset = df.reset_index(drop=True)
                    all_dfs.append(df_reset)
                    logger.info(f"   + Dump SQL {i + 1}: {len(df_reset)} lignes, {len(df_reset.columns)} colonnes")

            if not all_dfs:
                logger.error("❌ Aucune donnée disponible")
                return None

            # Concaténer tous les DataFrames avec gestion des erreurs
            logger.info("   🔄 Concaténation des DataFrames...")

            # Filtrer les DataFrames vides
            valid_dfs = [df for df in all_dfs if not df.empty]

            if not valid_dfs:
                logger.error("❌ Aucun DataFrame valide après filtrage")
                return None

            # Vérifier la compatibilité des colonnes
            all_columns = set()
            for df in valid_dfs:
                all_columns.update(df.columns)
            logger.info(f"   📋 Total colonnes uniques: {len(all_columns)}")

            # Concaténer avec gestion d'erreur
            try:
                # Première tentative avec options standard
                consolidated = pd.concat(valid_dfs, ignore_index=True, sort=False)
                logger.info("   ✅ Concaténation réussie avec méthode standard")
            except Exception as e:
                logger.warning(f"   ⚠️ Erreur lors de la concaténation standard: {e}")
                logger.info("   🔄 Tentative avec méthode alternative...")

                # Alternative: concaténer un par un avec alignement des colonnes
                consolidated = valid_dfs[0].copy()
                for df in valid_dfs[1:]:
                    # Aligner les colonnes
                    for col in df.columns:
                        if col not in consolidated.columns:
                            consolidated[col] = None
                    for col in consolidated.columns:
                        if col not in df.columns:
                            df[col] = None

                    # Concaténer
                    consolidated = pd.concat([consolidated, df], ignore_index=True, sort=False)

                logger.info("   ✅ Concaténation réussie avec méthode alternative")

            # Supprimer les colonnes dupliquées si nécessaire
            if consolidated.columns.duplicated().any():
                logger.warning("⚠️ Colonnes dupliquées détectées dans le résultat final, nettoyage...")
                consolidated = consolidated.loc[:, ~consolidated.columns.duplicated()]

            logger.info(f"✅ Total fusionné: {len(consolidated)} lignes, {len(consolidated.columns)} colonnes")

            # Étape 4: Déduplication
            if 'order_id' in consolidated.columns and 'part_id' in consolidated.columns:
                initial_count = len(consolidated)
                # Gérer les valeurs nulles avant déduplication
                subset_cols = ['order_id', 'part_id']
                for col in subset_cols:
                    if consolidated[col].isna().any():
                        logger.warning(f"⚠️ Des valeurs nulles dans {col}, remplissage temporaire")
                        consolidated[col] = consolidated[col].fillna(f'MISSING_{col}')

                consolidated = consolidated.drop_duplicates(subset=subset_cols, keep='first')
                if initial_count > len(consolidated):
                    logger.info(f"🧹 Déduplication: {initial_count} → {len(consolidated)} lignes")
            else:
                missing_cols = [col for col in ['order_id', 'part_id'] if col not in consolidated.columns]
                logger.warning(f"⚠️ Colonnes manquantes pour déduplication: {missing_cols}")

            # Étape 5: Ajout des colonnes manquantes
            logger.info("\n🔧 ÉTAPE 4: Ajout des colonnes manquantes")

            default_columns = {
                'insert_lamination': 'NONE',
                'insert_paper_type': 'NONE',
                'insert_color': 'NONE',
                'insert_size': -1,
                'insert_active': 0,
                'tab_active': 0,
                'tab_page_number': -1,
                'trim_size': -1,
                'tab_color': 'NONE',
                'tab_lamination': 'NONE',
                'tab_size': -1,
                'tab_paper_type': 'NONE',
                'coil_type': 'NONE',
                'cover_paper_type': 'NONE',
                'double_sided_cover': 0,
                'cover_color': 'NONE',
                'head_and_tail': 'NONE',
                'cover_size': 'NONE',
                'case_finish_type': 'NONE',
                'case_paper_type': 'NONE',
                'cover_case_color': 'NONE',
                'back_cover_flat_size': -1,
                'spine_type': 'NONE',
                'has_coil': 0,
                'has_insert': 0,
                'has_tab': 0,
                'has_backcover': 0,
                'security_label': 0,
                'perf': 0,
                'shrinkwrap': 0,
                'three_hole_drill': 0
            }

            columns_added = 0
            existing_cols = set(consolidated.columns)

            for col, default_value in default_columns.items():
                if col not in existing_cols:
                    logger.info(f"   ➕ Ajout: {col} = {default_value}")
                    consolidated[col] = default_value
                    columns_added += 1

            if columns_added > 0:
                logger.info(f"✅ {columns_added} colonnes ajoutées")
            else:
                logger.info("✅ Toutes les colonnes sont déjà présentes")

            # Étape 6: Conversion des types
            logger.info("\n🔧 ÉTAPE 5: Conversion des types")

            string_cols = [
                'insert_lamination', 'insert_paper_type', 'insert_color',
                'tab_color', 'tab_lamination', 'tab_paper_type',
                'coil_type', 'cover_paper_type', 'cover_color', 'head_and_tail',
                'cover_size', 'case_finish_type', 'case_paper_type',
                'cover_case_color', 'spine_type', 'text_paper_type', 'text_color',
                'cover_finish_type', 'priority_level', 'binding_type',
                'label_type', 'siren'
            ]

            for col in string_cols:
                if col in consolidated.columns:
                    consolidated[col] = consolidated[col].fillna('NONE').astype(str)

            numeric_cols = [
                'insert_size', 'tab_page_number', 'trim_size', 'tab_size',
                'back_cover_flat_size', 'security_label', 'has_coil', 'has_insert',
                'has_tab', 'has_backcover', 'perf', 'double_sided_cover',
                'shrinkwrap', 'three_hole_drill', 'quantity', 'production_page',
                'height', 'thickness', 'width', 'quantity_min', 'quantity_max',
                'insert_active', 'tab_active'
            ]

            for col in numeric_cols:
                if col in consolidated.columns:
                    consolidated[col] = pd.to_numeric(consolidated[col], errors='coerce').fillna(0)

            # Étape 7: Sauvegarde en EXCEL
            logger.info(f"\n💾 ÉTAPE 6: Sauvegarde en EXCEL - {CONSOLIDATED_FILE}")

            CONSOLIDATED_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Sauvegarder avec gestion d'erreur
            try:
                with pd.ExcelWriter(CONSOLIDATED_FILE, engine='openpyxl') as writer:
                    consolidated.to_excel(writer, sheet_name='Données consolidées', index=False)

                logger.info(f"✅ Fichier Excel créé: {CONSOLIDATED_FILE}")
                logger.info(f"   Lignes: {len(consolidated)}")
                logger.info(f"   Colonnes: {len(consolidated.columns)}")

                if CONSOLIDATED_FILE.exists():
                    size_mb = os.path.getsize(CONSOLIDATED_FILE) / (1024 * 1024)
                    logger.info(f"   Taille: {size_mb:.2f} MB")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la sauvegarde Excel: {e}")
                # Tentative de sauvegarde en CSV comme fallback
                csv_file = CONSOLIDATED_FILE.with_suffix('.csv')
                consolidated.to_csv(csv_file, index=False)
                logger.info(f"✅ Sauvegarde de secours en CSV: {csv_file}")

            self.tracking["last_consolidation"] = datetime.now().isoformat()
            self._save_tracking()

            logger.info("\n" + "=" * 60)
            logger.info("✅ CONSOLIDATION MYSQL DOCKER TERMINÉE AVEC SUCCÈS")
            logger.info("=" * 60)

            return CONSOLIDATED_FILE

        except Exception as e:
            logger.error(f"❌ Erreur lors de la consolidation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def get_consolidated_file(self) -> Path:
        """Retourne le chemin du fichier consolidé Excel"""
        return CONSOLIDATED_FILE if CONSOLIDATED_FILE.exists() else None


# Instance singleton
_consolidator = None


def get_consolidator():
    global _consolidator
    if _consolidator is None:
        _consolidator = DataConsolidator()
    return _consolidator


def run_consolidation():
    """Fonction appelée par pricing_full_pipeline.py"""
    consolidator = get_consolidator()
    return consolidator.consolidate()


def get_consolidated_file():
    """Retourne le chemin du fichier consolidé"""
    consolidator = get_consolidator()
    return consolidator.get_consolidated_file()


def clean_temp_folders():
    """Nettoie les dossiers temporaires"""
    consolidator = get_consolidator()
    sql_count = consolidator._clean_folder(SQL_DUMPS_DIR, "*.sql")
    return sql_count


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--clean":
            count = clean_temp_folders()
            print(f"🧹 Nettoyage terminé: {count} fichier(s) supprimé(s)")
        elif sys.argv[1] == "--status":
            sql_files = list(SQL_DUMPS_DIR.glob('*.sql'))
            print(f"📁 Dossier sql/ : {len(sql_files)} fichier(s)")
            print(f"📁 Fichier consolidé Excel: {'✅' if CONSOLIDATED_FILE.exists() else '❌'} {CONSOLIDATED_FILE}")
            if CONSOLIDATED_FILE.exists():
                size_mb = os.path.getsize(CONSOLIDATED_FILE) / (1024 * 1024)
                print(f"   Taille: {size_mb:.2f} MB")
        else:
            print("Usage: python consolidate_data.py [--clean|--status]")
    else:
        path = run_consolidation()
        if path:
            print(f"\n✅ Fichier Excel consolidé prêt: {path}")
        else:
            print("\n❌ Échec de la consolidation")