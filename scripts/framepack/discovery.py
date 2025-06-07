import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

class FramepackDiscovery:
    """Locate FramePack F1 models inside the HuggingFace cache"""

    REQUIRED = {
        # --- 修正箇所：コンポーネント名をキーにし、リポジトリIDを値にする ---
        "transformer": "lllyasviel/FramePack_F1_I2V_HY_20250503",
        "text_encoder": "hunyuanvideo-community/HunyuanVideo",
        "vae": "hunyuanvideo-community/HunyuanVideo",
        # flux_redux_bflは直接モデルとして使われないため、ここでは不要
    }

    def __init__(self, cache_dir: str | None = None):
        # HuggingFaceのキャッシュディレクトリを特定する
        self.cache_dir = Path(cache_dir) if cache_dir else Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface"))
        self.hub_cache = self.cache_dir / "hub"

    def get_local_path(self, component_name: str) -> str | None:
        """
        指定されたコンポーネントのローカルパス（snapshotディレクトリ）を取得する
        """
        if component_name not in self.REQUIRED:
            return None
            
        repo_id = self.REQUIRED[component_name]
        
        # models--namespace--repo-name 形式のディレクトリ名を作成
        repo_path = self.hub_cache / f"models--{repo_id.replace('/', '--')}"
        
        if not repo_path.is_dir():
            print(f"Warning: Repo path does not exist: {repo_path}")
            return None
        
        # snapshotsディレクトリ内の最新のスナップショット（ハッシュ）のパスを返す
        snapshots_dir = repo_path / "snapshots"
        if not snapshots_dir.is_dir():
            print(f"Warning: Snapshots directory does not exist for {repo_id}")
            return None
        
        # 更新日時が最新のスナップショットディレクトリを取得
        try:
            latest_snapshot = max(snapshots_dir.iterdir(), key=os.path.getmtime)
            return str(latest_snapshot)
        except ValueError:
            print(f"Warning: No snapshots found in {snapshots_dir}")
            return None

    # discover_models と validator はダウンロードと検証が完了しているので、
    # この後の工程では直接使われない。念のため残しておく。
    def discover_models(self) -> dict[str, bool]:
        # (この関数の内容は変更なし)
        results: dict[str, bool] = {"all_found": True}
        for name, rel in self.REQUIRED.items():
            path = self.cache_dir / ("models--" + rel.replace("/", "--"))
            exists = path.exists()
            results[name] = exists
            if not exists:
                results["all_found"] = False
        # ...
        pass
        
