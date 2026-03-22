import logging
import sys
from datetime import datetime


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    # file_handler = logging.FileHandler(f'logs/pricing_api_{datetime.now().strftime("%Y%m%d")}.log')
    # file_handler.setFormatter(formatter)
    # root_logger.addHandler(file_handler)

    return root_logger