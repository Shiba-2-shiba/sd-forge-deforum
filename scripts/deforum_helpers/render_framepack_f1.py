"""Entry point for FramePack F1 rendering using integration layer."""

import os  # osモジュールをインポート
from scripts.framepack.integration import FramepackIntegration


def render_animation_f1(args, anim_args, video_args, framepack_f1_args, root):
    """Generate video using FramePack F1 via the integration helper."""
    
    # --- ★★★ 修正箇所：ここから ★★★ ---
    # Hugging Faceライブラリを強制的にオフラインモードにする
    # これにより、ローカルパスがオンラインのリポジトリIDとして誤解釈されるのを防ぐ
    print("Forcing Hugging Face Hub to OFFLINE mode for local model loading.")
    os.environ['HF_HUB_OFFLINE'] = '1'
    # --- ★★★ 修正箇所：ここまで ★★★ ---
    
    integration = FramepackIntegration(device=root.device)
    try:
        integration.setup_environment()
        integration.generate_video(args, anim_args, video_args, framepack_f1_args, root)
    finally:
        integration.cleanup_environment()
        
        # --- ★★★ 修正箇所：ここから ★★★ ---
        # 処理が終了したら、必ず環境変数を元に戻す（削除する）
        # これをしないと、WebUI全体の他の機能がオンラインで動作しなくなる可能性がある
        if 'HF_HUB_OFFLINE' in os.environ:
            print("Restoring Hugging Face Hub to ONLINE mode.")
            del os.environ['HF_HUB_OFFLINE']
        # --- ★★★ 修正箇所：ここまで ★★★ ---
