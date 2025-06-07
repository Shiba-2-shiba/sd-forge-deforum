import os
from pathlib import Path

class FramepackDiscovery:
    """
    Checks for the existence of FramePack F1 models and resolves their local paths.
    Includes enhanced logging for debugging purposes.
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
        print("\n" + "="*50)
        print("--- [Discovery] Initializing... ---")

        # ステップ1：ベースとなるキャッシュディレクトリを決定
        base_cache_dir = Path(cache_dir) if cache_dir else Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface"))
        print(f"[DEBUG] Base cache directory determined as: {base_cache_dir}")

        # ステップ2：最終的な探索ディレクトリを決定
        # Forge環境と標準環境の両方に対応するため、'hub'サブディレクトリの存在を確認
        if (base_cache_dir / "hub").is_dir():
            self.hub_cache = base_cache_dir / "hub"
            print(f"[DEBUG] 'hub' directory found. Using standard cache structure.")
        else:
            self.hub_cache = base_cache_dir
            print(f"[DEBUG] 'hub' directory not found. Using flat cache structure (Forge environment).")
        
        print(f"[INFO] Final resolved cache directory to be searched: {self.hub_cache}")
        print(f"[INFO] Does this directory exist? -> {self.hub_cache.is_dir()}")
        print("="*50 + "\n")


    def _get_snapshot_path(self, repo_id: str) -> Path | None:
        """指定されたリポジトリの最新のスナップショットディレクトリパスを取得する"""
        repo_dir_name = f"models--{repo_id.replace('/', '--')}"
        repo_path = self.hub_cache / repo_dir_name
        
        print(f"  [sub-check] Attempting to find repo directory: {repo_path}")
        if not repo_path.is_dir():
            print(f"  [sub-check] FAIL: Repo directory not found.")
            return None
        print(f"  [sub-check] OK: Repo directory found.")

        snapshots_dir = repo_path / "snapshots"
        print(f"  [sub-check] Attempting to find snapshots directory: {snapshots_dir}")
        if not snapshots_dir.is_dir():
            print(f"  [sub-check] FAIL: Snapshots directory not found.")
            return None
        print(f"  [sub-check] OK: Snapshots directory found.")
            
        try:
            latest_snapshot = max(snapshots_dir.iterdir(), key=os.path.getmtime)
            print(f"  [sub-check] OK: Found latest snapshot: {latest_snapshot.name}")
            return latest_snapshot
        except ValueError:
            print(f"  [sub-check] FAIL: No snapshot directories found inside {snapshots_dir}.")
            return None

    def check_models_exist(self) -> tuple[bool, list[str]]:
        """全ての必須モデルが存在するかチェックする。"""
        missing_repos = []
        all_found = True
        
        for component, info in self.REQUIRED_COMPONENTS.items():
            repo_id = info["repo_id"]
            check_file = info["check_file"]
            
            print(f"\n--- Checking component: '{component}' (Repo: {repo_id}) ---")
            
            snapshot_path = self._get_snapshot_path(repo_id)
            
            if snapshot_path:
                full_check_file_path = snapshot_path / check_file
                print(f"  [Final Check] Verifying existence of file: {full_check_file_path}")
                
                if full_check_file_path.exists():
                    print(f"  [Result] SUCCESS: Required file found.")
                else:
                    print(f"  [Result] FAIL: Required file NOT found.")
                    all_found = False
                    if repo_id not in missing_repos:
                        missing_repos.append(repo_id)
            else:
                print(f"  [Result] FAIL: Snapshot directory for this component not found.")
                all_found = False
                if repo_id not in missing_repos:
                    missing_repos.append(repo_id)

        print("\n--- All checks complete. ---")
        return all_found, missing_repos

    def get_local_path(self, component_name: str) -> str | None:
        """指定されたコンポーネントのローカルパス（snapshotディレクトリ）を取得する"""
        if component_name not in self.REQUIRED_COMPONENTS:
            return None
        
        repo_id = self.REQUIRED_COMPONENTS[component_name]["repo_id"]
        snapshot_path = self._get_snapshot_path(repo_id)
        return str(snapshot_path) if snapshot_path and snapshot_path.is_dir() else None
