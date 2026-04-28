import logging

from pricing__epac.src.shared.logging import configure_logging


def setup_logging(level=logging.INFO):
    """Setup API logging through the shared project logger."""
    return configure_logging(level=level, reset_handlers=True)
