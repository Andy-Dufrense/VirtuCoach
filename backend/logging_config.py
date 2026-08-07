"""Shared logging setup for VirtuCoach backend.

Usage:
    from logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("something happened")
"""

import logging
import sys

ROOT_LOGGER: logging.Logger | None = None


def setup_logging(*, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger once. Called from main.py on startup."""
    global ROOT_LOGGER

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)-24s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)

    root = logging.getLogger("virtucoach")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    # Quiet down noisy third-party loggers
    for noisy in ("chromadb", "sentence_transformers", "urllib3", "httpx", "openai",
                  "basic_pitch", "tensorflow", "absl", "h5py", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Suppress basic-pitch's module-level import warnings (CoremlTools, tflite, etc.)
    # and MediaPipe protobuf noise (MessageFactory/GetPrototype)
    logging.getLogger().setLevel(logging.WARNING)
    _root_handler = logging.StreamHandler(sys.stdout)
    _root_handler.addFilter(lambda record: not any(
        kw in record.getMessage() for kw in
        ("Coremltools", "tflite-runtime", "CoreML", "TFLite",
         "MessageFactory", "GetPrototype")
    ))
    logging.getLogger().handlers = [_root_handler]

    ROOT_LOGGER = root
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the virtucoach namespace."""
    if ROOT_LOGGER is None:
        setup_logging()
    return logging.getLogger(f"virtucoach.{name}")
