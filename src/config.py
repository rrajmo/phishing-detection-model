from pathlib import Path
import yaml

def load_config() -> dict:
    BASE_DIR = Path(__file__).resolve().parents[1]
    CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config
