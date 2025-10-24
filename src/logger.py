import logging
import sys

def initialize_logger(level: int = logging.INFO) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%m-%d-%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger()
