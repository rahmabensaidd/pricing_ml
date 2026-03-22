# main.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# IMPORTANT: Patch OpenSSL en premier !
try:
    from pricing__epac import openssl_patch

    print("✅ OpenSSL patch applied successfully")
except ImportError as e:
    print(f"⚠️ Could not import openssl_patch: {e}")
except Exception as e:
    print(f"⚠️ Error applying openssl_patch: {e}")

import argparse
import uvicorn
from config.settings import settings
from config import NUM_COLS, CAT_COLS, ALL_FEATURES
from src.utils.logging_config import setup_logging


def start():
    """Function to start the API"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=settings.API_HOST)
    parser.add_argument("--port", type=int, default=settings.API_PORT)
    parser.add_argument("--mlflow-uri", type=str, default=settings.MLFLOW_TRACKING_URI)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Setup logging
    setup_logging()

    print("\n" + "=" * 60)
    print("🚀 PRICING MLOPS API V7".center(60))
    print("=" * 60)
    print(f"\n📡 MLflow: {args.mlflow_uri}")
    print(f"🌐 API: http://{args.host}:{args.port}")
    print(f"📚 Docs: http://{args.host}:{args.port}/docs")
    print(f"\n📊 Model features:")
    print(f"   - Numeric (including booleans): {len(NUM_COLS)}")
    print(f"   - Categorical: {len(CAT_COLS)}")
    print(f"   - Total: {len(ALL_FEATURES)}")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    start()