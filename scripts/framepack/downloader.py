from pathlib import Path
from huggingface_hub import snapshot_download

class FramepackDownloader:
    """Download FramePack F1 repositories using huggingface_hub"""

    REPOS = [
        "hunyuanvideo-community/HunyuanVideo",
        "lllyasviel/flux_redux_bfl",
        "lllyasviel/FramePack_F1_I2V_HY_20250503",
    ]

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def download_all_models(self):
        for repo in self.REPOS:
            print(f"Downloading {repo}...")
            snapshot_download(repo_id=repo, cache_dir=self.cache_dir, local_dir_use_symlinks=False)
            print(f"Finished downloading {repo}")
