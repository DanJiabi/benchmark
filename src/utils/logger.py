import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any
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

    def has(self, key: str) -> bool:
        """检查配置中是否存在指定键"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return False

        return True


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

    logger = logging.getLogger("benchmark")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return

    try:
        logger.info(f"开始下载: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = downloaded / total_size * 100
                    if int(percent) % 10 == 0:
                        logger.info(
                            f"    下载进度: {percent:.0f}% ({downloaded / 1024 / 1024:.1f}MB)"
                        )

        logger.info(
            f"✅ 下载完成: {output_path.name} ({total_size / 1024 / 1024:.1f}MB)"
        )

    except requests.exceptions.Timeout:
        logger.error(f"❌ 下载超时: {url}")
        logger.error("   请检查网络连接或稍后重试")
        raise

    except requests.exceptions.ConnectionError:
        logger.error(f"❌ 网络连接错误: {url}")
        logger.error("   请检查网络连接")
        raise

    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ HTTP错误: {e}")
        logger.error(f"   URL: {url}")
        logger.error("   请检查URL是否正确")
        raise

    except IOError as e:
        logger.error(f"❌ 文件写入错误: {e}")
        logger.error(f"   目标路径: {output_path}")
        logger.error("   请检查磁盘空间和权限")
        raise

    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        logger.error(f"   URL: {url}")
        logger.error(f"   目标: {output_path}")
        raise


def save_predictions(predictions: List[Dict[str, Any]], output_path: Path) -> None:
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)


def load_predictions(input_path: Path) -> List[Dict[str, Any]]:
    import json

    with open(input_path, "r") as f:
        return json.load(f)
