import os
from pathlib import Path

class FramepackDiscovery:
    """
    Checks for the existence of FramePack F1 models and resolves their local paths.
    """
    REQUIRED_COMPONENTS = {
        "transformer": {
            "repo_id": "lllyasviel/FramePack_F1_I2V_HY_20250503",
            "check_file": "diffusion_pytorch_model.safetensors.index.json"
        },
        "text_encoder": {
            "repo_id": "hunyuanvideo-community/HunyuanVideo",
            "check_file": "text_encoder/config.json"
        },
        "vae": {
            "repo_id": "hunyuanvideo-community/HunyuanVideo",
            "check_file": "vae/config.json"
        },
        "flux_bfl": {
            "repo_id": "lllyasviel/flux_redux_bfl",
            "check_file": "model_index.json"
        }
    }

    def __init__(self, cache_dir: str | None = None):
        # --- ★★★ 最終修正箇所：ここから ★★★ ---
        # HuggingFaceのキャッシュディレクトリを特定
        base_cache_dir = Path(cache_dir) if cache_dir else Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface"))

        # Forge環境では .../diffusers が、標準では .../huggingface/hub が探索対象となる
        # 'hub' ディレクトリが存在する場合のみパスに追加し、それ以外はベースパスを直接使用する
        if (base_cache_dir / "hub").is_dir():
            self.hub_cache = base_cache_dir / "hub"
        else:
            self.hub_cache = base_cache_dir
        
        print(f"Discovery using final resolved cache directory: {self.hub_cache}")
        # --- ★★★ 最終修正箇所：ここまで ★★★ ---

    def _get_snapshot_path(self, repo_id: str) -> Path | None:
        """指定されたリポジトリの最新のスナップショットディレクトリパスを取得する"""
        repo_path = self.hub_cache / f"models--{repo_id.replace('/', '--')}"
        if not repo_path.is_dir():
            return None
        
        snapshots_dir = repo_path / "snapshots"
        if not snapshots_dir.is_dir():
            return None
            
        try:
            return max(snapshots_dir.iterdir(), key=os.path.getmtime)
        except ValueError:
            return None

    def check_models_exist(self) -> tuple[bool, list[str]]:
        """全ての必須モデルが存在するかチェックする。"""
        missing_repos = []
        all_found = True
        for component, info in self.REQUIRED_COMPONENTS.items():
            repo_id = info["repo_id"]
            snapshot_path = self._get_snapshot_path(repo_id)
            
            if snapshot_path is None or not (snapshot_path / info["check_file"]).exists():
                all_found = False
                if repo_id not in missing_repos:
                    missing_repos.append(repo_id)
        return all_found, missing_repos

    def get_local_path(self, component_name: str) -> str | None:
        """指定されたコンポーネントのローカルパス（snapshotディレクトリ）を取得する"""
        if component_name not in self.REQUIRED_COMPONENTS:
            return None
        
        repo_id = self.REQUIRED_COMPONENTS[component_name]["repo_id"]
        snapshot_path = self._get_snapshot_path(repo_id)
        return str(snapshot_path) if snapshot_path and snapshot_path.is_dir() else None
