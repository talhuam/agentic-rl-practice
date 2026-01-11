from modelscope import snapshot_download
from pathlib import Path
from common.global_config import model_root_path, model_name
import os


def download_model():
    path = Path(os.path.join(model_root_path, model_name))
    if path.exists():
        print("model exists, exit!")
    else:
        snapshot_download(model_name, cache_dir=path.absolute())


if __name__ == '__main__':
    download_model()