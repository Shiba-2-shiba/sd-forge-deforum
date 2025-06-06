# -*- coding: utf-8 -*-
import os
import torch
import gc
import numpy as np
from PIL import Image

# WebUIの共有オブジェクトとヘルパー関数をインポート
from modules import shared
# 修正点: 'gpu'を直接インポートする代わりに、'device'を'gpu'としてインポートします
from modules.devices import cpu, device as gpu

# -------------------------------------------------------------------------
# FramePack F1専用のヘルパーとマネージャークラスをインポート
# これらのファイルは 'scripts/framepack/' ディレクトリに配置されている必要があります。
# -------------------------------------------------------------------------

# メモリ管理ユーティリティ
from .framepack.memory import (
    offload_model_from_device_for_memory_preservation,
    move_model_to_device_with_memory_preservation,
    model_on_device,
    get_cuda_free_memory_gb
)

# モデル管理クラス (シングルトンとして使用)
from .framepack.transformer_manager import TransformerManager
from .framepack.text_encoder_manager import TextEncoderManager

# VAEとトークナイザーをロードするためのインポート
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaTokenizerFast, CLIPTokenizer

# FramePackのコア機能
from .framepack.k_diffusion_hunyuan import sample_hunyuan
from .framepack.hunyuan import vae_encode, vae_decode, encode_prompt_conds
from .framepack.utils import resize_and_center_crop, save_bcthw_as_mp4


# -------------------------------------------------------------------------
# グローバルなモデルマネージャ (シングルトンインスタンス)
# -------------------------------------------------------------------------
# これらのマネージャはWebUIセッション中に一度だけ初期化され、モデルをキャッシュします。
# これにより、2回目以降の生成が高速になります。

_F1_MANAGERS = {
    "transformer": None,
    "text_encoder": None,
    "vae": None,
    "tokenizers": None
}

def _initialize_managers():
    """
    シングルトンマネージャとトークナイザーを初期化する。
    """
    global _F1_MANAGERS

    if _F1_MANAGERS["transformer"] is None:
        # VRAMサイズに基づいて high_vram モードを決定
        free_mem_gb = get_cuda_free_memory_gb(gpu)
        high_vram = free_mem_gb > 16  # 16GBを閾値とする
        print(f"[FramePack F1] Free VRAM: {free_mem_gb:.2f} GB. High VRAM mode: {high_vram}")

        # 各マネージャを初期化
        # F1モデルを使用するため `use_f1_model=True` を指定
        _F1_MANAGERS["transformer"] = TransformerManager(device=gpu, high_vram_mode=high_vram, use_f1_model=True)
        _F1_MANAGERS["text_encoder"] = TextEncoderManager(device=gpu, high_vram_mode=high_vram)
        
        # VAEとトークナイザーも同様に管理
        # このスクリプトでは、単純なクラスの属性として保持
        class VaeManager:
            def __init__(self, device, high_vram):
                self.model = None
                self.device = device
                self.high_vram = high_vram
            def get(self):
                if self.model is None:
                    self.model = AutoencoderKLHunyuanVideo.from_pretrained(
                        "hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16
                    ).cpu()
                    self.model.eval()
                return self.model

        class TokenizerManager:
            def __init__(self):
                self.tokenizer = None
                self.tokenizer_2 = None
            def get(self):
                if self.tokenizer is None:
                    self.tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
                    self.tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
                return self.tokenizer, self.tokenizer_2

        _F1_MANAGERS["vae"] = VaeManager(device=gpu, high_vram=high_vram)
        _F1_MANAGERS["tokenizers"] = TokenizerManager()
        
    return _F1_MANAGERS


# -------------------------------------------------------------------------
# WebUIモデル操作のヘルパー関数
# -------------------------------------------------------------------------

def _get_sdxl_components(sd_model):
    """
    現在ロードされているWebUIのモデルから主要コンポーネントを取得する。
    """
    components = {
        "unet": None,
        "vae": None,
        "text_encoders": None
    }
    if hasattr(sd_model, "model") and hasattr(sd_model.model, "diffusion_model"):
        components["unet"] = sd_model.model.diffusion_model
    elif hasattr(sd_model, "unet"):
         components["unet"] = sd_model.unet
         
    if hasattr(sd_model, "first_stage_model"):
        components["vae"] = sd_model.first_stage_model
        
    if hasattr(sd_model, "cond_stage_model"):
        components["text_encoders"] = sd_model.cond_stage_model
        
    return components


# -------------------------------------------------------------------------
# メインのレンダリング関数
# -------------------------------------------------------------------------

def render_animation_f1(args, anim_args, video_args, framepack_f1_args, root):
    """
    FramePack F1モデルを自己完結的にロードし、動画を生成する。
    WebUIの既存モデルとVRAMを動的に入れ替える。
    """
    print("--- [FramePack F1] Start Rendering (Self-Contained Mode) ---")

    # シングルトンマネージャを初期化・取得
    managers = _initialize_managers()

    # WebUIの既存モデルコンポーネントを取得
    sdxl_components = _get_sdxl_components(shared.sd_model)
    
    # 既存モデルをVRAMから退避させ、F1モデルの実行環境を確保する
    # try...finallyブロックで、処理中にエラーが発生しても必ず元の状態に復元する
    try:
        print("[FramePack F1] Step 1: Offloading existing WebUI model from VRAM...")
        if sdxl_components["unet"]:
            offload_model_from_device_for_memory_preservation(sdxl_components["unet"], root.device)
        if sdxl_components["vae"]:
            offload_model_from_device_for_memory_preservation(sdxl_components["vae"], root.device)
        if sdxl_components["text_encoders"]:
            offload_model_from_device_for_memory_preservation(sdxl_components["text_encoders"], root.device)
        gc.collect()
        torch.cuda.empty_cache()
        print("[FramePack F1] Existing model offloaded successfully.")

        # -------------------------------------------------------------
        # ここからFramePack F1専用モデルを使用した処理
        # -------------------------------------------------------------

        # F1のVAEとトークナイザーを取得
        f1_vae = managers["vae"].get()
        f1_tokenizer, f1_tokenizer_2 = managers["tokenizers"].get()

        # F1のVAEを使用して初期画像をエンコード
        print("[FramePack F1] Step 2: Encoding initial image with F1 VAE...")
        with model_on_device(f1_vae, root.device):
            init_image = np.array(Image.open(args.init_image).convert("RGB"))
            init_image = resize_and_center_crop(init_image, args.W, args.H)
            start_latent = vae_encode(init_image, f1_vae)
        print("[FramePack F1] Initial image encoded.")

        # F1のテキストエンコーダーを取得・使用してプロンプトをエンコード
        print("[FramePack F1] Step 3: Encoding prompts with F1 Text Encoders...")
        f1_text_encoder, f1_text_encoder_2 = managers["text_encoder"].get_text_encoders() # ここでロードが発生する可能性がある
        with model_on_device(f1_text_encoder, root.device), model_on_device(f1_text_encoder_2, root.device):
            llama_vec, clip_l_pooler = encode_prompt_conds(
                anim_args.animation_prompts,
                f1_text_encoder,
                f1_text_encoder_2,
                f1_tokenizer,
                f1_tokenizer_2,
            )
        # 使用後すぐにCPUに戻してVRAMを確保
        f1_text_encoder.to(cpu)
        f1_text_encoder_2.to(cpu)
        print("[FramePack F1] Prompts encoded.")
        
        # F1のTransformerを取得して動画を生成
        print("[FramePack F1] Step 4: Generating video frames with F1 Transformer...")
        managers["transformer"].ensure_transformer_state() # LoRA設定などを反映
        f1_transformer = managers["transformer"].get_transformer()
        history_latents = start_latent.clone()
        total_sections = int(max(round((anim_args.max_frames) / (framepack_f1_args.f1_generation_latent_size * 4 - 3)), 1))

        with model_on_device(f1_transformer, root.device):
            history_latents = history_latents.to(root.device)
            llama_vec = llama_vec.to(root.device)
            clip_l_pooler = clip_l_pooler.to(root.device)

            for i_section in range(total_sections):
                shared.state.job = f"FramePack F1: Section {i_section + 1}/{total_sections}"
                shared.state.job_no = i_section + 1
                if shared.state.interrupted:
                    print("[FramePack F1] Generation interrupted by user.")
                    break

                # strengthとstepsはDeforumのUIから渡される引数を使用
                generated_latents = sample_hunyuan(
                    transformer=f1_transformer,
                    initial_latent=history_latents[:, :, -1:],
                    strength=framepack_f1_args.f1_image_strength,
                    steps=framepack_f1_args.f1_generation_latent_size,
                    llama_vec=llama_vec,
                    clip_l_pooler=clip_l_pooler,
                )
                history_latents = torch.cat([history_latents, generated_latents], dim=2)
                # Note: F1モデルは順方向で生成するため .flip は不要
        
        print(f"[FramePack F1] Video frames generated. Total latents shape: {history_latents.shape}")

        # F1のVAEで最終的な動画フレームをデコード
        print("[FramePack F1] Step 5: Decoding final video with F1 VAE...")
        with model_on_device(f1_vae, root.device):
            final_video_frames = vae_decode(history_latents, f1_vae)
        print("[FramePack F1] Final video decoded.")

        # 動画をファイルに保存
        output_path = os.path.join(args.outdir, f"{root.timestring}_framepack_f1.mp4")
        save_bcthw_as_mp4(final_video_frames, output_path, fps=video_args.fps)
        print(f"[FramePack F1] Video saved to {output_path}")

    except Exception as e:
        print(f"!!! [FramePack F1] An error occurred during rendering: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # -------------------------------------------------------------
        # 処理の終了、またはエラー発生時に必ず実行される後処理
        # -------------------------------------------------------------
        print("[FramePack F1] Step 6: Finalizing and restoring WebUI state...")
        # F1モデル群を明示的に解放（ただし、マネージャはインスタンスを保持）
        if 'f1_transformer' in locals() and f1_transformer is not None:
             f1_transformer.to(cpu)
        if 'f1_vae' in locals() and f1_vae is not None:
             f1_vae.to(cpu)
        if 'f1_text_encoder' in locals() and f1_text_encoder is not None:
             f1_text_encoder.to(cpu)
        if 'f1_text_encoder_2' in locals() and f1_text_encoder_2 is not None:
             f1_text_encoder_2.to(cpu)
        
        gc.collect()
        torch.cuda.empty_cache()

        # WebUIの元のモデルをVRAMに戻す
        if sdxl_components["unet"]:
            move_model_to_device_with_memory_preservation(sdxl_components["unet"], root.device)
        if sdxl_components["vae"]:
            move_model_to_device_with_memory_preservation(sdxl_components["vae"], root.device)
        if sdxl_components["text_encoders"]:
            move_model_to_device_with_memory_preservation(sdxl_components["text_encoders"], root.device)
            
        print("--- [FramePack F1] Process Finished. WebUI state restored. ---")
