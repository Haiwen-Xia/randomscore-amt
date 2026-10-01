"""Small rank-zero logger for training and evaluation runs."""
import contextlib
import logging
import os
from pathlib import Path


class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


@contextlib.contextmanager
def capture_output(directory):
    """Configure one console/file logger on rank zero and capture warnings."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    logger = logging.getLogger()
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    file_handler = None
    if rank == 0:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [rank=%(rank)s] %(name)s: %(message)s"
        )
        file_handler = logging.FileHandler(directory / "run.log", mode="a")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RankFilter(rank))
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(RankFilter(rank))
        logger.handlers = [file_handler, stream_handler]
        logger.setLevel(logging.INFO)
        logging.LoggerAdapter(logger, {"rank": rank}).info("Process started")
    else:
        logger.handlers = []
        logger.setLevel(logging.CRITICAL)
    logging.captureWarnings(True)
    try:
        yield
    except BaseException:
        if rank == 0:
            logging.exception("Run failed")
        raise
    finally:
        logging.captureWarnings(False)
        if file_handler is not None:
            file_handler.flush()
            file_handler.close()
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
