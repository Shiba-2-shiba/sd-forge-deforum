import os
import torch
import numpy as np
from PIL import Image

from modules import shared, sd_models

from ..framepack.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from ..framepack.k_diffusion_hunyuan import sample_hunyuan
from ..framepack.hunyuan import vae_encode, vae_decode, encode_prompt_conds
from ..framepack.utils import resize_and_center_crop, save_bcthw_as_mp4

F1_TRANSFORMER = None


def load_f1_model(root):
    global F1_TRANSFORMER
    if F1_TRANSFORMER is None:
        model_path = os.path.join(root.models_path, "framepack", "f1_model_weights.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FramePack F1 model not found at: {model_path}")
        print("Loading FramePack F1 model...")
        # Placeholder load - actual model loading depends on repo structure
        F1_TRANSFORMER = HunyuanVideoTransformer3DModelPacked()
        state_dict = torch.load(model_path, map_location=root.device)
        F1_TRANSFORMER.load_state_dict(state_dict)
        F1_TRANSFORMER.eval().to(root.device)
        print("FramePack F1 model loaded.")
    return F1_TRANSFORMER


def render_animation_f1(args, anim_args, video_args, framepack_f1_args, root):
    """Simplified placeholder implementation for FramePack F1 rendering."""
    print("Starting FramePack F1 rendering process...")
    model = load_f1_model(root)

    init_image = Image.open(args.init_image).convert("RGB")
    init_image = resize_and_center_crop(init_image, (args.W, args.H))

    start_latent = vae_encode(init_image, shared.sd_model.first_stage_model, root.device)

    llama_vec, clip_l_pooler = encode_prompt_conds(
        tokenizer=shared.sd_model.cond_stage_model.wrapped.tokenizer,
        text_encoder=shared.sd_model.cond_stage_model.wrapped.transformer,
        text=anim_args.animation_prompts
    )

    history_latents = start_latent
    total_sections = int(max(round((anim_args.max_frames) / (framepack_f1_args.f1_generation_latent_size * 4 - 3)), 1))

    for i_section in range(total_sections):
        shared.state.job = f"FramePack F1: Section {i_section + 1}/{total_sections}"
        shared.state.job_no = i_section + 1
        if shared.state.interrupted:
            break

        generated_latents = sample_hunyuan(
            transformer=model,
            initial_latent=history_latents[:, :, -1:],
            strength=framepack_f1_args.f1_image_strength,
            steps=framepack_f1_args.f1_generation_latent_size,
            llama_vec=llama_vec,
            clip_l_pooler=clip_l_pooler,
        )

        history_latents = torch.cat([generated_latents.flip(dims=[2]), history_latents], dim=2)

    final_video_frames = vae_decode(history_latents.flip(dims=[2]), shared.sd_model.first_stage_model)

    output_path = os.path.join(args.outdir, f"{root.timestring}_framepack_f1.mp4")
    save_bcthw_as_mp4(final_video_frames, output_path, video_args.fps)
    print(f"FramePack F1 video saved to {output_path}")

