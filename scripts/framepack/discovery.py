import os
from pathlib import Path

class FramepackDiscovery:
    """
    Checks for the existence of FramePack F1 models and resolves their local paths.
    """
    # 必須コンポーネントと、それに対応するリポジトリIDおよび検証用ファイル
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
        # HuggingFaceのキャッシュディレクトリを特定
        self.cache_dir = Path(cache_dir) if cache_dir else Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface"))
        self.hub_cache = self.cache_dir / "hub"
        print(f"Discovery using cache directory: {self.hub_cache}")

    def _get_snapshot_path(self, repo_id: str) -> Path | None:
        """指定されたリポジトリの最新のスナップショットディレクトリパスを取得する"""
        repo_path = self.hub_cache / f"models--{repo_id.replace('/', '--')}"
        if not repo_path.is_dir():
            return None
        
        snapshots_dir = repo_path / "snapshots"
        if not snapshots_dir.is_dir():
            return None
            
        try:
            # 更新日時が最新のスナップショットディレクトリを返す
            return max(snapshots_dir.iterdir(), key=os.path.getmtime)
        except ValueError:
            return None # スナップショットが存在しない

    def check_models_exist(self) -> tuple[bool, list[str]]:
        """
        全ての必須モデルが存在するかチェックする。
        戻り値: (全てのモデルが存在するかどうか, 見つからなかったリポジトリのリスト)
        """
        missing_repos = []
        all_found = True
        for component, info in self.REQUIRED_COMPONENTS.items():
            repo_id = info["repo_id"]
            snapshot_path = self._get_snapshot_path(repo_id)
            
            # スナップショットパス自体、またはその中のチェック用ファイルが存在しない場合
            if snapshot_path is None or not (snapshot_path / info["check_file"]).exists():
                print(f"Component '{component}' not found. Expected repo: {repo_id}")
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
