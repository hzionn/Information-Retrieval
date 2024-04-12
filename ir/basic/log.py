import logging


def setup_logger(filename, classname, level):
    logger = logging.getLogger(f"{filename}.{classname}")
    if not logger.handlers:
        logger.setLevel(level.upper())
        logger.propagate = False  # to not process the log message in the root logger
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
