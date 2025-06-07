"""Entry point for FramePack F1 rendering using integration layer."""

# --- 修正箇所：ここから ---
import sys
from pathlib import Path

# このファイルの場所を基準に、拡張機能の'scripts'ディレクトリへのパスを解決
# Path(__file__) -> このファイルのパス (.../render_framepack_f1.py)
# .parent -> .../deforum_helpers/
# .parent -> .../scripts/
scripts_path = Path(__file__).parent.parent.resolve()
# --- 修正箇所：ここまで ---

# FramepackIntegrationをインポートする前に、パスを挿入する必要がある
# sys.path.insert(0, str(scripts_path)) を実行することで、
# Pythonはモジュールを探す際に、まずこの拡張機能の'scripts'ディレクトリ内を検索するようになります。
# これにより、システムにインストールされたdiffusersより、ここにあるdiffusersが優先されます。
sys.path.insert(0, str(scripts_path))

from scripts.framepack.integration import FramepackIntegration

def render_animation_f1(args, anim_args, video_args, framepack_f1_args, root):
    """Generate video using FramePack F1 via the integration helper."""
    integration = FramepackIntegration(device=root.device)
    try:
        integration.setup_environment()
        integration.generate_video(args, anim_args, video_args, framepack_f1_args, root)
    finally:
        integration.cleanup_environment()
        
        # --- 修正箇所：ここから ---
        # 処理が終了したら、追加したパスをクリーンアップして、他の拡張機能への影響を防ぐ
        if str(scripts_path) in sys.path:
            sys.path.remove(str(scripts_path))
        # --- 修正箇所：ここまで ---
