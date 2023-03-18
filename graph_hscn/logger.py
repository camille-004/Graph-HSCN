import logging
import sys
import time
from pathlib import Path


class CustomLogger:
    def __init__(self, log_file: Path, metric_name: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s | %(message)s"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)
        if metric_name == "ap":
            self.metric_name = "AP"
        elif metric_name == "mae":
            self.metric_name = "MAE"
        else:
            self.metric_name = metric_name

    def log_train(
        self, epoch: int, loss: float, metric_val: float, start_time: float
    ) -> None:
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(
            f"Epoch: {epoch} -- Loss: {loss:.4f}, "
            f"{self.metric_name}: {metric_val:.4f}, "
            f"Duration: {duration:.4f} seconds"
        )

    def log_eval(self, loss: float, metric_val: float, split: str) -> None:
        self.logger.info(
            f"{split} -- Loss: {loss:.4f}, {self.metric_name}: "
            f"{metric_val:.4f}"
        )

    def info(self, message: str) -> None:
        self.logger.info(message)
