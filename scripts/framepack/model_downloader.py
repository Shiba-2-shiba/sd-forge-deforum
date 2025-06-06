from pathlib import Path
from huggingface_hub import snapshot_download

class ModelDownloader:
    """Utility class to download FramePack F1 related models"""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _download_repo(self, repo: str):
        snapshot_download(repo_id=repo, cache_dir=self.cache_dir, local_dir_use_symlinks=False)

    def download_f1(self):
        """Download repositories required for FramePack F1"""
        repos = [
            "hunyuanvideo-community/HunyuanVideo",
            "lllyasviel/flux_redux_bfl",
            "lllyasviel/FramePack_F1_I2V_HY_20250503",
        ]
        for repo in repos:
            print(f"Downloading {repo}...")
            self._download_repo(repo)
            print(f"Finished downloading {repo}")

