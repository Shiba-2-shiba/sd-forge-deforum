"""Entry point for FramePack F1 rendering using integration layer."""

from scripts.framepack.integration import FramepackIntegration


def render_animation_f1(args, anim_args, video_args, framepack_f1_args, root):
    """Generate video using FramePack F1 via the integration helper."""
    integration = FramepackIntegration(device=root.device)
    try:
        integration.setup_environment()
        integration.generate_video(args, anim_args, video_args, framepack_f1_args, root)
    finally:
        integration.cleanup_environment()

