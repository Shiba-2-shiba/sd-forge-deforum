import os
import torch
import numpy as np
from PIL import Image

from modules import shared

from scripts.framepack.memory import (
    offload_model_from_device_for_memory_preservation,
    move_model_to_device_with_memory_preservation,
    model_on_device,
)

# Import FramePack modules from the scripts package so that they resolve
# correctly regardless of the current working directory
from scripts.framepack.hunyuan_video_packed import (
    HunyuanVideoTransformer3DModelPacked,
)
from scripts.framepack.k_diffusion_hunyuan import sample_hunyuan
from scripts.framepack.hunyuan import vae_encode, vae_decode, encode_prompt_conds
from scripts.framepack.utils import resize_and_center_crop, save_bcthw_as_mp4

F1_TRANSFORMER = None


def load_f1_model(root):
    global F1_TRANSFORMER
    if F1_TRANSFORMER is None:
        model_base = os.path.expanduser(
            "~/.cache/huggingface/hub/models--lllyasviel--FramePack_F1_I2V_HY_20250503"
        )
        if not os.path.isdir(model_base):
            raise FileNotFoundError(
                f"FramePack F1 model directory not found at: {model_base}"
            )

        snapshots_dir = os.path.join(model_base, "snapshots")
        snapshots = (
            [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if os.path.isdir(snapshots_dir)
            else []
        )
        if not snapshots:
            raise FileNotFoundError(f"No snapshot found in {snapshots_dir}")

        model_directory = os.path.join(snapshots_dir, snapshots[0])
        print(f"Loading sharded FramePack F1 model from: {model_directory}")

        F1_TRANSFORMER = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            model_directory,
            torch_dtype=torch.float16,
            map_location="cpu",
        )
        F1_TRANSFORMER.eval()
        print("FramePack F1 model loaded to CPU.")
    return F1_TRANSFORMER


def render_animation_f1(args, anim_args, video_args, framepack_f1_args, root):
    """Render video with FramePack F1 while carefully managing GPU memory."""
    print("Starting FramePack F1 rendering process with memory management...")

    unet = shared.sd_model.model.diffusion_model
    vae = shared.sd_model.first_stage_model

    # 1. Encode the initial image with the VAE on GPU
    with model_on_device(vae, root.device):
        init_image = np.array(Image.open(args.init_image).convert("RGB"))
        init_image = resize_and_center_crop(init_image, args.W, args.H)
        start_latent = vae_encode(init_image, vae)

    llama_vec, clip_l_pooler = encode_prompt_conds(
        anim_args.animation_prompts,
        shared.sd_model.cond_stage_model.wrapped.transformer,
        getattr(shared.sd_model.cond_stage_model.wrapped, "transformer_2", shared.sd_model.cond_stage_model.wrapped.transformer),
        shared.sd_model.cond_stage_model.wrapped.tokenizer,
        getattr(shared.sd_model.cond_stage_model.wrapped, "tokenizer_2", shared.sd_model.cond_stage_model.wrapped.tokenizer),
    )

    # Free memory used by UNet before loading F1
    offload_model_from_device_for_memory_preservation(unet, root.device)

    model = load_f1_model(root)
    history_latents = start_latent
    total_sections = int(max(round((anim_args.max_frames) / (framepack_f1_args.f1_generation_latent_size * 4 - 3)), 1))

    with model_on_device(model, root.device):
        history_latents = history_latents.to(root.device)

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

    # 3. Decode with VAE back on GPU
    with model_on_device(vae, root.device):
        final_video_frames = vae_decode(history_latents.flip(dims=[2]), vae)

    output_path = os.path.join(args.outdir, f"{root.timestring}_framepack_f1.mp4")
    save_bcthw_as_mp4(final_video_frames, output_path, video_args.fps)
    print(f"FramePack F1 video saved to {output_path}")

    # Restore UNet for any further processing
    move_model_to_device_with_memory_preservation(unet, root.device)
