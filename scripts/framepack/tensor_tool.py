# tensor_tool.py (修正反映版)

import os
import torch
import einops
import numpy as np
import re
from PIL import Image

# Framepack F1のコア機能とヘルパー関数をインポート
from .k_diffusion_hunyuan import sample_hunyuan
from .utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    soft_append_bcthw,
    resize_and_center_crop,
    generate_timestamp,
)
from .hunyuan import encode_prompt_conds, vae_encode
from .clip_vision import hf_clip_vision_encode
from .bucket_tools import find_nearest_bucket
from .vae_cache import vae_decode_cache

# メモリ管理ユーティリティ
from .memory import (
    cpu,
    gpu,
    load_model_as_complete,
    unload_complete_models,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
)

# スケジュール文字列から数値を抽出するヘルパー関数
def parse_schedule_string(schedule_str: str) -> float:
    """ "0: (1.23)" のような文字列から数値部分を抽出する """
    match = re.search(r'\((.*?)\)', schedule_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0  # デフォルト値
    return 0.0

@torch.no_grad()
def execute_generation(managers: dict, device, args, anim_args, video_args, framepack_f1_args, root):
    """
    Deforumから呼び出される動画生成のメイン関数。
    UI関連のコードを排除し、ロジックのみに特化。
    """
    print("[tensor_tool] Starting video generation process...")

    # --- 1. マネージャーからモデルを取得 ---
    transformer_manager = managers["transformer"]
    text_encoder_manager = managers["text_encoder"]
    vae_manager = managers["vae"]
    image_encoder_manager = managers["image_encoder"]
    image_processor_manager = managers["image_processor"]
    tokenizer_manager = managers["tokenizers"]
    
    # ★★★ 修正箇所1: モデルを取得する前に、各マネージャーにロードを指示する ★★★
    transformer_manager.ensure_transformer_state()
    text_encoder_manager.ensure_text_encoder_state()
    
    transformer = transformer_manager.get_transformer()
    text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()
    vae = vae_manager.get_model()
    image_encoder = image_encoder_manager.get_model()
    image_processor = image_processor_manager.get_processor()
    tokenizer, tokenizer_2 = tokenizer_manager.get_tokenizers()
    
    # 状態が確定した後にVRAMモードを取得
    high_vram = transformer_manager.current_state['high_vram']

    # --- 2. パラメータの準備 ---
    prompt = args.positive_prompts
    seed = args.seed
    steps = args.steps
    width, height = args.W, args.H

    cfg = parse_schedule_string(anim_args.strength_schedule)
    gs = parse_schedule_string(anim_args.distilled_cfg_scale_schedule)
    rs = getattr(framepack_f1_args, 'guidance_rescale', 0.0)
    latent_window_size = framepack_f1_args.f1_generation_latent_size
    use_vae_cache = getattr(framepack_f1_args, 'use_vae_cache', False)

    job_id = generate_timestamp()
    # ★★★ 修正箇所2: 正しい出力ディレクトリ変数を参照する ★★★
    output_path = os.path.join(args.outdir, f"{job_id}.mp4")

    # --- 3. プロンプトエンコード ---
    print("[tensor_tool] Encoding prompts...")
    if not high_vram:
        load_model_as_complete(text_encoder, target_device=device)
        load_model_as_complete(text_encoder_2, target_device=device)

    prompt_embeds, prompt_poolers = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    
    n_prompt = args.negative_prompts if hasattr(args, 'negative_prompts') else ""
    n_prompt_embeds, n_prompt_poolers = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

    prompt_embeds, prompt_embeds_mask = crop_or_pad_yield_mask(prompt_embeds, length=256)
    n_prompt_embeds, n_prompt_embeds_mask = crop_or_pad_yield_mask(n_prompt_embeds, length=256)
    
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2)

    # --- 4. 初期画像の準備とエンコード ---
    print("[tensor_tool] Processing initial image...")
    init_image_path = args.init_image
    if not init_image_path or not os.path.exists(init_image_path):
        raise FileNotFoundError(f"Initial image not found at: {init_image_path}")

    input_image_pil = Image.open(init_image_path).convert("RGB")
    
    bucket_w, bucket_h = find_nearest_bucket(width, height)
    input_image_np = resize_and_center_crop(np.array(input_image_pil), bucket_w, bucket_h)

    # VAEエンコード
    if not high_vram: load_model_as_complete(vae, target_device=device)
    
    img_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1.0
    img_pt = img_pt.permute(2, 0, 1).unsqueeze(0).to(device)
    img_pt = img_pt.unsqueeze(2) 
    
    initial_latent = vae_encode(img_pt, vae)
    if not high_vram: unload_complete_models(vae)

    # Image Encoder
    if not high_vram: load_model_as_complete(image_encoder, target_device=device)
    
    image_embeddings_output = hf_clip_vision_encode(input_image_np, image_processor, image_encoder)
    image_embeddings = image_embeddings_output.last_hidden_state 
    
    if not high_vram: unload_complete_models(image_encoder)

    # --- 5. サンプリングの実行 ---
    print("[tensor_tool] Starting sampling loop...")
    if not high_vram:
        preserved_memory = getattr(framepack_f1_args, 'preserved_memory', 8.0)
        move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=preserved_memory)

    rnd = torch.Generator(device=device).manual_seed(seed)

    sampler_kwargs = dict(
        transformer=transformer,
        sampler="unipc",
        width=bucket_w,
        height=bucket_h,
        frames=latent_window_size,
        real_guidance_scale=cfg,
        distilled_guidance_scale=gs,
        guidance_rescale=rs,
        num_inference_steps=steps,
        generator=rnd,
        prompt_embeds=prompt_embeds.to(transformer.dtype),
        prompt_embeds_mask=prompt_embeds_mask,
        prompt_poolers=prompt_poolers.to(transformer.dtype),
        negative_prompt_embeds=n_prompt_embeds.to(transformer.dtype),
        negative_prompt_embeds_mask=n_prompt_embeds_mask,
        negative_prompt_poolers=n_prompt_poolers.to(transformer.dtype),
        image_embeddings=image_embeddings.to(transformer.dtype),
        latent_indices=None,
        clean_latents=None,
        clean_latent_indices=None,
        device=device,
        dtype=torch.bfloat16,
    )
    
    generated_latents = sample_hunyuan(**sampler_kwargs)

    if not high_vram:
        offload_model_from_device_for_memory_preservation(transformer, target_device=device, preserved_memory_gb=8.0)

    # --- 6. VAEデコードと動画保存 ---
    print("[tensor_tool] Decoding latents and saving video...")
    if not high_vram: load_model_as_complete(vae, target_device=device)

    if use_vae_cache:
        print("[tensor_tool] Using VAE cache for decoding.")
        pixels = vae_decode_cache(generated_latents, vae)
    else:
        pixels = vae.decode(generated_latents.to(vae.device, vae.dtype) / vae.config.scaling_factor).sample

    if not high_vram: unload_complete_models(vae)

    pixels = torch.nn.functional.interpolate(pixels, size=(height, width), mode='bilinear', align_corners=False)

    save_bcthw_as_mp4(pixels, output_path, fps=video_args.fps, crf=18)
    
    print(f"[tensor_tool] Video generation finished. Output saved to: {output_path}")

    return output_path
