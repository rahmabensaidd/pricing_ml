# pricing_epac/openssl_patch.py
"""
Patch central pour l'erreur OpenSSL
À importer dans TOUS les fichiers qui utilisent MLflow/boto3
"""

import os
import sys
import warnings

# 1. Variable d'environnement (la plus importante)
os.environ['URLLIB3_USE_PYOPENSSL'] = '0'

# 2. Bloquer complètement pyOpenSSL
class BlockOpenSSL:
    """Remplace le module OpenSSL pour bloquer son import"""
    def __getattr__(self, name):
        raise ImportError("pyOpenSSL est désactivé - ce module est bloqué intentionnellement")

# 3. Remplacer OpenSSL dans sys.modules AVANT tout import
if 'OpenSSL' not in sys.modules:
    sys.modules['OpenSSL'] = BlockOpenSSL()
if 'urllib3.contrib.pyopenssl' not in sys.modules:
    sys.modules['urllib3.contrib.pyopenssl'] = BlockOpenSSL()

# 4. Ignorer les warnings
warnings.filterwarnings('ignore', module='urllib3.contrib.pyopenssl')
warnings.filterwarnings('ignore', module='OpenSSL')

# 5. Vérification que le patch est actif
print("🔧 [Patch OpenSSL] Appliqué avec succès - L'erreur X509_V_FLAG_NOTIFY_POLICY est contournée")