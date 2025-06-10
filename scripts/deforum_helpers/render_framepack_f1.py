# render_framepack_f1.py (修正後の完全なコード)

import os
from scripts.framepack.integration import FramepackIntegration
from modules import shared

def render_animation_f1(args, anim_args, video_args, framepack_f1_args, root):
    """Generate video using FramePack F1 via the integration helper."""
    
    print("Forcing Hugging Face Hub to OFFLINE mode for local model loading.")
    os.environ['HF_HUB_OFFLINE'] = '1'
    
    # integration.pyのsetup_environmentはsd_modelを要求するため、pオブジェクトから渡します。
    sd_model = root.p.sd_model
    integration = FramepackIntegration(device=root.device)
    
    # 返り値を格納する変数を初期化します
    returned_images = None
    
    try:
        integration.setup_environment(sd_model)
        
        # ★★★ 修正点1: integration.generate_videoの返り値を受け取ります ★★★
        returned_images = integration.generate_video(args, anim_args, video_args, framepack_f1_args, root)
        
    finally:
        integration.cleanup_environment()
        
        if 'HF_HUB_OFFLINE' in os.environ:
            print("Restoring Hugging Face Hub to ONLINE mode.")
            del os.environ['HF_HUB_OFFLINE']
            
    # ★★★ 修正点2: 受け取った画像を返します。これがUIに表示されます。★★★
    return returned_images
