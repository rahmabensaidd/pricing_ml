# tests/conftest.py
"""
Configuration pytest pour les tests du module de prétraitement.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


def pytest_configure(config):
    """Configuration pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marque les tests lents (à exécuter occasionnellement)"
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Crée un répertoire temporaire pour les données de test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cleanup_test_files():
    """Nettoie les fichiers créés pendant les tests."""
    files_to_cleanup = []
    yield files_to_cleanup
    for file_path in files_to_cleanup:
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)


def pytest_runtest_setup(item):
    """Configuration avant chaque test."""
    pass


def pytest_runtest_teardown(item):
    """Nettoyage après chaque test."""
    pass