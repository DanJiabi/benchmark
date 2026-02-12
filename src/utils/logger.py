import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch


class Config:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_dataset_config(self) -> Dict[str, Any]:
        return self.config.get("dataset", {})

    def get_models_config(self) -> List[Dict[str, Any]]:
        return self.config.get("models", [])

    def get_evaluation_config(self) -> Dict[str, Any]:
        return self.config.get("evaluation", {})

    def get_output_config(self) -> Dict[str, Any]:
        return self.config.get("output", {})

    def get_logging_config(self) -> Dict[str, Any]:
        return self.config.get("logging", {})


def setup_logger(config: Config) -> logging.Logger:
    log_config = config.get_logging_config()
    log_level = getattr(logging, log_config.get("level", "INFO").upper())

    logger = logging.getLogger("benchmark")
    logger.setLevel(log_level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_config.get("save_logs", False):
            log_dir = Path(log_config.get("log_dir", "outputs/logs"))
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_dir / "benchmark.log")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_device(device_config: str = "auto") -> torch.device:
    if device_config != "auto":
        return torch.device(device_config)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def download_model_weights(url: str, output_path: Path) -> None:
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def save_predictions(predictions: List[Dict[str, Any]], output_path: Path) -> None:
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)


def load_predictions(input_path: Path) -> List[Dict[str, Any]]:
    import json

    with open(input_path, "r") as f:
        return json.load(f)
