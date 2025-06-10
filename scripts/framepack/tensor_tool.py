# tensor_tool.py (最終修正・安定版)

import os
import torch
import numpy as np
import re
from PIL import Image

# Framepack F1のコア機能とヘルパー関数をインポート
from .k_diffusion_hunyuan import sample_hunyuan
from .utils import (
    crop_or_pad_yield_mask,
    resize_and_center_crop,
)
from .hunyuan import encode_prompt_conds, vae_encode
from .clip_vision import hf_clip_vision_encode
from .bucket_tools import find_nearest_bucket
from .vae_cache import vae_decode_cache
from .vae_settings import apply_vae_settings

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
            return 0.0
    return 0.0

@torch.no_grad()
def execute_generation(managers: dict, device, args, anim_args, video_args, framepack_f1_args, root):
    """
    Deforumから呼び出される動画生成のメイン関数。
    Latent補間とファイル保存機能を追加。
    """
    print("[tensor_tool] Starting video generation process...")

    # --- 1. マネージャーからモデルを取得 ---
    transformer_manager = managers["transformer"]
    text_encoder_manager = managers["text_encoder"]
    vae_manager = managers["vae"]
    image_encoder_manager = managers["image_encoder"]
    image_processor_manager = managers["image_processor"]
    tokenizer_manager = managers["tokenizers"]
    
    transformer_manager.ensure_transformer_state()
    text_encoder_manager.ensure_text_encoder_state()
    
    transformer = transformer_manager.get_transformer()
    text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()
    vae = vae_manager.get_model()
    image_encoder = image_encoder_manager.get_model()
    image_processor = image_processor_manager.get_processor()
    tokenizer, tokenizer_2 = tokenizer_manager.get_tokenizers()
    
    high_vram = transformer_manager.current_state['high_vram']

    # --- 2. パラメータの準備 ---
    prompt = args.positive_prompts
    seed = args.seed
    steps = args.steps
    width, height = args.W, args.H

    # ★★★ 修正箇所 ★★★
    # Strength scheduleからstrength値を取得。これが初期画像のノイズ強度を決定します。
    strength = parse_schedule_string(anim_args.strength_schedule)
    # Deforumでは通常、cfg_scale_scheduleからCFG値を取得します。
    # 元のコードではstrength_scheduleが使われていましたが、これは誤りである可能性が高いです。
    # getattrで安全に取得し、見つからない場合は一般的なデフォルト値7.0を使用します。
    cfg = parse_schedule_string(getattr(anim_args, 'cfg_scale_schedule', "0: (7.0)"))
    gs = parse_schedule_string(anim_args.distilled_cfg_scale_schedule)
    rs = getattr(framepack_f1_args, 'guidance_rescale', 0.0)
    latent_window_size = framepack_f1_args.f1_generation_latent_size
    
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

    if hasattr(transformer, 'initialize_teacache'):
        print(f"[tensor_tool] Initializing Teacache for acceleration. Steps: {steps}, Threshold: 0.15")
        transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=steps,
            rel_l1_thresh=0.15
        )

    rnd = torch.Generator(device=device).manual_seed(seed)
    
    # ★★★ 修正箇所 ★★★
    # サンプラーが内部で (frames + 3) // 4 の計算を行うため、
    # 期待するLatentフレーム数(latent_window_size)から逆算した最終フレーム数を渡す。
    frames_to_generate = latent_window_size * 4 - 3
    print(f"[tensor_tool] Requesting sampler to generate latents for {frames_to_generate} final frames (expecting {latent_window_size} keyframes).")

    clean_latents = initial_latent
    clean_latent_indices = torch.tensor([[0]], device=device)

    sampler_kwargs = dict(
        transformer=transformer,
        sampler="unipc",
        strength=strength,  # ★★★ 修正箇所 ★★★: 初期ノイズ強度を渡す
        width=bucket_w,
        height=bucket_h,
        frames=frames_to_generate, # ★★★ 修正箇所 ★★★: latent_window_size から変更
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
        clean_latents=clean_latents,
        clean_latent_indices=clean_latent_indices,
        device=device,
        dtype=torch.bfloat16,
    )
    
    generated_latents = sample_hunyuan(**sampler_kwargs)
    print(f"[tensor_tool] Sampler generated {generated_latents.shape[2]} latent frames.")

    if not high_vram:
        offload_model_from_device_for_memory_preservation(transformer, target_device=device, preserved_memory_gb=8.0)

    # --- 6. Latent補間、VAEデコード、ファイル保存 ---
    
    # ★★★ 修正箇所 ★★★
    # サンプラーが期待通りのキーフレーム数(9)を返すようになったため、
    # それを最終フレーム数(33)に補間するこの処理は、当初の設計通り必須となります。
    print(f"[tensor_tool] Interpolating {generated_latents.shape[2]} latent frames to {frames_to_generate} frames...")
    if generated_latents.shape[2] != frames_to_generate:
        original_dtype = generated_latents.dtype
        generated_latents = torch.nn.functional.interpolate(
            generated_latents.to(torch.float32),
            size=(frames_to_generate, generated_latents.shape[3], generated_latents.shape[4]),
            mode='nearest'
        )
        generated_latents = generated_latents.to(original_dtype)
        print(f"[tensor_tool] Interpolation complete. New latent shape: {generated_latents.shape}")

    print(f"[tensor_tool] Decoding {generated_latents.shape[2]} latent frames...")
    if not high_vram: load_model_as_complete(vae, target_device=device)

    apply_vae_settings(vae)
    pixels = vae_decode_cache(generated_latents, vae)

    if not high_vram: unload_complete_models(vae)

    # フレームごとにリサイズ
    frame_list = list(pixels.split(1, dim=2))
    resized_frames = []
    for frame in frame_list:
        frame_4d = frame.squeeze(2)
        resized_frame = torch.nn.functional.interpolate(frame_4d, size=(height, width), mode='bilinear', align_corners=False)
        resized_frames.append(resized_frame)

    # テンソルをNumpy配列に変換
    resized_frames_tensor = torch.cat(resized_frames, dim=0).cpu() 
    resized_frames_tensor = (resized_frames_tensor + 1.0) / 2.0
    resized_frames_tensor = resized_frames_tensor.clamp(0, 1) * 255.0
    frames_np = resized_frames_tensor.to(torch.float32).permute(0, 2, 3, 1).numpy().astype(np.uint8)

    # Deforumは指定されたフォルダにファイルが保存されることを期待する
    output_dir = args.outdir
    start_frame_idx = anim_args.extract_from_frame if hasattr(anim_args, 'extract_from_frame') else 0

    saved_count = 0
    # ★★★ 修正箇所 ★★★: Gradioに返すためファイルパスのリストを初期化
    saved_filepaths = []
    for i, frame_np in enumerate(frames_np):
        current_frame_number = start_frame_idx + i
        filename = f"{root.timestring}_{current_frame_number:09d}.png"
        filepath = os.path.join(output_dir, filename)
        
        try:
            image = Image.fromarray(frame_np)
            image.save(filepath)
            saved_count += 1
            saved_filepaths.append(filepath) # ★★★ 修正箇所 ★★★: 保存したパスをリストに追加
        except Exception as e:
            print(f"[ERROR] Failed to save frame {filename}: {e}")

    print(f"[tensor_tool] Process complete. Saved {saved_count} frames to {output_dir}")

    # ★★★ 新しい修正案 ★★★
    # ユーザーの要望に応え、UIに最後のフレーム画像を表示する。
    # エラーを回避するため、ファイルパスではなくPIL.Imageオブジェクトを返す。
    # また、Deforum UIが期待する形式（画像のリスト）に合わせる。
    last_frame_image = None
    if 'image' in locals() and isinstance(image, Image.Image):
        last_frame_image = image  # forループの最後に保存されたPILイメージ
        print(f"[tensor_tool] Returning the last generated frame as a PIL Image to be displayed in the UI.")
        # Deforum UIは通常、画像の「リスト」を期待するため、リストに格納して返す
        return [last_frame_image]
    else:
        # 万が一画像が生成されなかった場合
        print("[tensor_tool] No valid last image found to return. Returning None.")
        return None
