import logging


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = 0
        logger.setLevel((logging.DEBUG))
        console = logging.StreamHandler()
        logger.addHandler(console)
        formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"
        )
        console.setFormatter(formatter)
    return logger


# For testing
base_log = get_logger(__name__).addHandler(logging.StreamHandler())

# Silenced
# logging.getLogger(__name__).addHandler(logging.NullHandler())
