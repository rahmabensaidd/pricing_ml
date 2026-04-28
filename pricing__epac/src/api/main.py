import argparse

import uvicorn

# IMPORTANT: Patch OpenSSL en premier !
try:
    import pricing__epac.openssl_patch  # noqa: F401

    print("OpenSSL patch applied successfully")
except ImportError as e:
    print(f"Could not import openssl_patch: {e}")
except Exception as e:
    print(f"Error applying openssl_patch: {e}")

from pricing__epac.src.api.utils.logging_config import setup_logging
from pricing__epac.src.config.feature_config import ALL_FEATURES, CAT_COLS, NUM_COLS
from pricing__epac.src.config.settings import settings


def start():
    """Function to start the API."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=settings.API_HOST)
    parser.add_argument("--port", type=int, default=settings.API_PORT)
    parser.add_argument("--mlflow-uri", type=str, default=settings.MLFLOW_TRACKING_URI)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    setup_logging()

    print("\n" + "=" * 60)
    print("PRICING MLOPS API V7".center(60))
    print("=" * 60)
    print(f"\nMLflow: {args.mlflow_uri}")
    print(f"API: http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    print(f"\nModel features:")
    print(f"   - Numeric (including booleans): {len(NUM_COLS)}")
    print(f"   - Categorical: {len(CAT_COLS)}")
    print(f"   - Total: {len(ALL_FEATURES)}")
    print("=" * 60)

    if args.reload:
        uvicorn.run(
            "pricing__epac.src.api.pricing_controller:app",
            host=args.host,
            port=args.port,
            reload=True,
        )
        return

    from pricing__epac.src.api.pricing_controller import app

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    start()
