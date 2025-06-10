# render_framepack_f1.py (修正後)

"""Entry point for FramePack F1 rendering using integration layer."""

import os
from scripts.framepack.integration import FramepackIntegration


def render_animation_f1(args, anim_args, video_args, framepack_f1_args, root):
    """Generate video using FramePack F1 via the integration helper."""
    
    # --- Hugging Faceのオフラインモード設定は変更なし ---
    print("Forcing Hugging Face Hub to OFFLINE mode for local model loading.")
    os.environ['HF_HUB_OFFLINE'] = '1'
    
    integration = FramepackIntegration(device=root.device)
    try:
        # ★★★ 修正箇所: root.sd_model を setup_environment に渡す ★★★
        integration.setup_environment(root.sd_model)
        integration.generate_video(args, anim_args, video_args, framepack_f1_args, root)
    finally:
        integration.cleanup_environment()
        
        # --- 環境変数を元に戻す処理も変更なし ---
        if 'HF_HUB_OFFLINE' in os.environ:
            print("Restoring Hugging Face Hub to ONLINE mode.")
            del os.environ['HF_HUB_OFFLINE']
