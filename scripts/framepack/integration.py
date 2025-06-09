import os
import torch
import gc
import numpy as np
from PIL import Image
import json

from modules import shared
from modules.devices import cpu

from .memory import (
    offload_model_from_device_for_memory_preservation,
    move_model_to_device_with_memory_preservation,
    get_cuda_free_memory_gb,
    DynamicSwapInstaller,
)
from .transformer_manager import TransformerManager
from .text_encoder_manager import TextEncoderManager
from .k_diffusion_hunyuan import sample_hunyuan
from .hunyuan import vae_encode, vae_decode, encode_prompt_conds
# ★★★ 修正/追加箇所 ★★★
from .utils import resize_and_center_crop, save_bcthw_as_mp4, numpy2pytorch, crop_or_pad_yield_mask, soft_append_bcthw
from .bucket_tools import find_nearest_bucket
# ★★★★★★★★★★★★★★★★
from .discovery import FramepackDiscovery
from scripts.diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaTokenizerFast, CLIPTokenizer, CLIPVisionModelWithProjection, SiglipImageProcessor
from .clip_vision import hf_clip_vision_encode

# --- マネージャークラス定義 (変更なし) ---
class VaeManager:
    """Hunyuan VAEのロードとライフサイクルを管理するクラス"""
    def __init__(self, device, high_vram_mode: bool, model_path: str):
        self.model = None
        self.device = device
        self.high_vram_mode = high_vram_mode
        self.is_loaded = False
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"VaeManager received an invalid model_path: {model_path}")
        self.model_path = model_path

    def _load_model(self):
        print(f"Loading Hunyuan VAE from: {self.model_path}")
        self.model = AutoencoderKLHunyuanVideo.from_pretrained(
            self.model_path, subfolder='vae', torch_dtype=torch.bfloat16, local_files_only=True
        ).cpu()
        self.model.eval()
        self.model.requires_grad_(False)
        self.is_loaded = True
        print("Hunyuan VAE loaded.")

    def get_model(self):
        if not self.is_loaded: self._load_model()
        return self.model

    def dispose(self):
        if self.model is not None:
            print("Disposing Hunyuan VAE...")
            self.model.to(cpu)
            del self.model
            self.model = None
            self.is_loaded = False

class ImageEncoderManager:
    """Image Encoder (CLIP Vision)のロードとライフサイクルを管理するクラス"""
    def __init__(self, device, model_path: str):
        self.model = None
        self.device = device
        self.is_loaded = False
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"ImageEncoderManager received an invalid model_path: {model_path}")
        self.model_path = model_path

    def get_model(self):
        if not self.is_loaded:
            print(f"Loading Image Encoder from: {self.model_path}")
            # ★★★ 修正箇所：ImageEncoderはSiglipVisionModelではなくCLIPVisionModelWithProjectionが正しい ★★★
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                self.model_path, subfolder="image_encoder", torch_dtype=torch.bfloat16,
                local_files_only=True, ignore_mismatched_sizes=True
            ).cpu()
            self.model.eval()
            self.is_loaded = True
        return self.model

    def dispose(self):
        if self.model is not None:
            print("Disposing Image Encoder...")
            self.model.to(cpu)
            del self.model
            self.model = None
            self.is_loaded = False

class ImageProcessorManager:
    """Image Processorのロードとライフサイクルを管理するクラス"""
    def __init__(self, model_path: str):
        self.processor = None
        self.is_loaded = False
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"ImageProcessorManager received an invalid model_path: {model_path}")
        self.model_path = model_path

    def get_processor(self):
        if not self.is_loaded:
            print(f"Loading Image Processor from: {self.model_path}")
            self.processor = SiglipImageProcessor.from_pretrained(
                self.model_path, subfolder="feature_extractor", local_files_only=True
            )
            self.is_loaded = True
        return self.processor

    def dispose(self):
        if self.processor is not None:
            print("Disposing Image Processor...")
            del self.processor
            self.processor = None
            self.is_loaded = False

class TokenizerManager:
    """Tokenizerのロードとライフサイクルを管理するクラス"""
    def __init__(self, model_path: str):
        self.tokenizer = None
        self.tokenizer_2 = None
        self.is_loaded = False
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"TokenizerManager received an invalid model_path: {model_path}")
        self.model_path = model_path

    def get_tokenizers(self):
        if not self.is_loaded:
            print(f"Loading Tokenizers from: {self.model_path}")
            self.tokenizer = LlamaTokenizerFast.from_pretrained(self.model_path, subfolder="tokenizer", local_files_only=True)
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer_2", local_files_only=True)
            self.is_loaded = True
        return self.tokenizer, self.tokenizer_2

    def dispose(self):
        if self.tokenizer is not None:
            print("Disposing Tokenizers...")
            del self.tokenizer; del self.tokenizer_2
            self.tokenizer = None; self.tokenizer_2 = None
            self.is_loaded = False


class FramepackIntegration:
    """FramePack F1のモデル管理とビデオ生成を統合する司令塔クラス。"""
    def __init__(self, device):
        self.device = device
        self.managers = None
        self.sdxl_components = None
        self.discovery = FramepackDiscovery()

    def _initialize_managers(self, local_paths: dict[str, str]):
        global_managers = {}
        free_mem_gb = get_cuda_free_memory_gb(self.device)
        high_vram = free_mem_gb > 16

        print("Initializing managers with explicit model paths...")
        
        global_managers["transformer"] = TransformerManager(
            device=self.device, 
            high_vram_mode=high_vram, 
            use_f1_model=True,
            model_path=local_paths.get("transformer")
        )
        global_managers["text_encoder"] = TextEncoderManager(
            device=self.device, 
            high_vram_mode=high_vram,
            model_path=local_paths.get("text_encoder")
        )
        global_managers["image_encoder"] = ImageEncoderManager(
            device=self.device, 
            model_path=local_paths.get("flux_bfl")
        )
        global_managers["image_processor"] = ImageProcessorManager(
            model_path=local_paths.get("flux_bfl")
        )
        global_managers["vae"] = VaeManager(
            device=self.device, 
            high_vram_mode=high_vram,
            model_path=local_paths.get("vae")
        )
        global_managers["tokenizers"] = TokenizerManager(
            model_path=local_paths.get("text_encoder")
        )
        
        print("All managers initialized.")
        return global_managers

    def _get_sdxl_components(self, sd_model):
        components = {"unet": None, "vae": None, "text_encoders": None}
        if hasattr(sd_model, "model") and hasattr(sd_model.model, "diffusion_model"):
            components["unet"] = sd_model.model.diffusion_model
        elif hasattr(sd_model, "unet"):
            components["unet"] = sd_model.unet
        if hasattr(sd_model, "first_stage_model"):
            components["vae"] = sd_model.first_stage_model
        if hasattr(sd_model, "cond_stage_model"):
            components["text_encoders"] = sd_model.cond_stage_model
        return components

    def setup_environment(self):
        print("Step 1: Checking for required local models...")
        models_exist, missing_repos = self.discovery.check_models_exist()

        if not models_exist:
            error_message = (
                "One or more FramePack F1 models are missing. Please download them manually before running Deforum.\n"
                "Missing repositories:\n"
            )
            for repo in missing_repos: error_message += f" - {repo}\n"
            error_message += "You can use the standalone downloader script if needed: 'python extensions/sd-forge-deforum/scripts/framepack/model_downloader.py'"
            raise FileNotFoundError(error_message)
        
        print("All required models found locally.")

        print("Step 2: Resolving local paths...")
        local_paths = {
            "transformer": self.discovery.get_local_path("transformer"),
            "text_encoder": self.discovery.get_local_path("text_encoder"),
            "vae": self.discovery.get_local_path("vae"),
            "flux_bfl": self.discovery.get_local_path("flux_bfl"),
        }
        
        for name, path in local_paths.items():
            if path is None or not os.path.isdir(path):
                raise RuntimeError(f"Failed to resolve a valid directory path for component '{name}': {path}")
            print(f"  - Resolved {name}: {path}")

        print("Step 3: Initializing model managers with resolved paths...")
        self.managers = self._initialize_managers(local_paths)

        print("Step 4: Offloading base SD model from VRAM...")
        self.sdxl_components = self._get_sdxl_components(shared.sd_model)
        if self.sdxl_components["unet"]: offload_model_from_device_for_memory_preservation(self.sdxl_components["unet"], self.device)
        if self.sdxl_components["vae"]: offload_model_from_device_for_memory_preservation(self.sdxl_components["vae"], self.device)
        if self.sdxl_components["text_encoders"]: offload_model_from_device_for_memory_preservation(self.sdxl_components["text_encoders"], self.device)
        
        gc.collect()
        torch.cuda.empty_cache()
        print("Environment setup complete.")

    def generate_video(self, args, anim_args, video_args, framepack_f1_args, root):
        managers = self.managers
        
        f1_vae = managers["vae"].get_model()
        f1_tokenizer, f1_tokenizer_2 = managers["tokenizers"].get_tokenizers()
        f1_image_processor = managers["image_processor"].get_processor()
        f1_image_encoder = managers["image_encoder"].get_model()

        prompt_text = ""
        prompts_schedule = args.prompts
        if not isinstance(prompts_schedule, dict) or not prompts_schedule:
            raise ValueError("Prompts are not in the expected dictionary format or are empty. Please check your Deforum prompt settings.")
        try:
            first_frame_key = sorted(prompts_schedule.keys(), key=int)[0]
            prompt_text = prompts_schedule[first_frame_key]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Could not extract the first prompt from the schedule: {e}")
        if not prompt_text:
            raise ValueError("The first prompt in the schedule is empty. Please provide a prompt.")
        
        print(f"[FramePack F1] Using single prompt for entire generation: '{prompt_text}'")

        pil_init_image = Image.open(args.init_image).convert("RGB")
        
        # ★★★ 修正箇所：解像度の最適化 (Bucket Resolution) ★★★
        print(f"Original resolution: {pil_init_image.width}x{pil_init_image.height}")
        optimal_height, optimal_width = find_nearest_bucket(pil_init_image.height, pil_init_image.width, resolution=640)
        print(f"Optimized to bucket resolution: {optimal_width}x{optimal_height}")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        print(f"[DEBUG] Initial VRAM Free: {get_cuda_free_memory_gb(self.device):.2f} GB")

        try:
            print("[DEBUG] Moving Image Encoder to GPU...")
            move_model_to_device_with_memory_preservation(f1_image_encoder, self.device)
            init_image_np_for_clip = np.array(pil_init_image)
            image_encoder_output = hf_clip_vision_encode(
                image=init_image_np_for_clip,
                feature_extractor=f1_image_processor,
                image_encoder=f1_image_encoder
            )
            image_embeddings_for_transformer = image_encoder_output.last_hidden_state
            print(f"[DEBUG] image_embeddings_for_transformer created. Shape: {image_embeddings_for_transformer.shape}, Device: {image_embeddings_for_transformer.device}")
            print(f"[DEBUG] VRAM Free after Image Encoding: {get_cuda_free_memory_gb(self.device):.2f} GB")
        finally:
            offload_model_from_device_for_memory_preservation(f1_image_encoder, self.device)
            print("[DEBUG] Image Encoder offloaded from GPU.")

        try:
            print("[DEBUG] Moving VAE to GPU for encoding...")
            move_model_to_device_with_memory_preservation(f1_vae, self.device)
            init_image_np = np.array(pil_init_image)
            # ★★★ 修正箇所：最適化された解像度を使用 ★★★
            init_image_np = resize_and_center_crop(init_image_np, optimal_width, optimal_height)
            init_tensor = numpy2pytorch([init_image_np])
            init_tensor = init_tensor.unsqueeze(2)
            
            start_latent = vae_encode(init_tensor, f1_vae)
            print(f"[DEBUG] start_latent created. Shape: {start_latent.shape}, Device: {start_latent.device}")
            print(f"[DEBUG] VRAM Free after VAE encoding: {get_cuda_free_memory_gb(self.device):.2f} GB")
        finally:
            offload_model_from_device_for_memory_preservation(f1_vae, self.device)
            print("[DEBUG] VAE offloaded from GPU.")

        print("[FramePack F1] Encoding prompts and then forcefully clearing text encoders from VRAM...")
        managers["text_encoder"].ensure_text_encoder_state()
        f1_text_encoder, f1_text_encoder_2 = managers["text_encoder"].get_text_encoders()
        
        print(f"[DEBUG] Text Encoders loaded. VRAM Free: {get_cuda_free_memory_gb(self.device):.2f} GB")
        
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt_text, f1_text_encoder, f1_text_encoder_2,
            f1_tokenizer, f1_tokenizer_2,
        )

        # ★★★ 修正箇所：プロンプト埋め込みとマスクの厳密な処理 ★★★
        llama_vec, prompt_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        
        llama_vec = llama_vec.to(self.device)
        prompt_mask = prompt_mask.to(self.device) # マスクもGPUへ
        clip_l_pooler = clip_l_pooler.to(self.device)
        print(f"[DEBUG] Prompt conditioning created. llama_vec shape: {llama_vec.shape}, clip_l_pooler shape: {clip_l_pooler.shape}, prompt_mask shape: {prompt_mask.shape}")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        print("[FramePack F1] Disposing text encoders...")
        managers["text_encoder"].dispose_text_encoders()
        gc.collect(); torch.cuda.empty_cache()
        print(f"[FramePack F1] VRAM cleaned. Free space: {get_cuda_free_memory_gb(self.device):.2f} GB")

        if not managers["transformer"].ensure_transformer_state():
            raise RuntimeError("Failed to load or setup the Transformer model. Check logs for OOM errors.")

        f1_transformer = managers["transformer"].get_transformer()

        # ★★★ 修正箇所：履歴をCPUで管理し、高度なコンテキスト管理を導入 ★★★
        history_latents = start_latent.clone().cpu()
        total_sections = int(max(round((anim_args.max_frames) / (framepack_f1_args.f1_generation_latent_size * 4 - 3)), 1))
        
        seed = args.seed if args.seed != -1 else torch.seed()
        generator = torch.Generator(device="cpu").manual_seed(int(seed)) # generatorはCPUで初期化
        print(f"[FramePack F1] Using seed: {seed}")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        print("\n" + "="*40)
        print("[DEBUG] PRE-SAMPLING CHECK")
        print(f"  - Target Device: {self.device}")
        try:
            print(f"  - Transformer Device: {f1_transformer.device}")
            print(f"  - Transformer Sample Param Device: {next(f1_transformer.parameters()).device}")
        except Exception as e:
            print(f"  - Could not determine transformer device: {e}")
        print(f"  - history_latents (initial_latent on CPU): Shape={history_latents.shape}, Dtype={history_latents.dtype}, Device={history_latents.device}")
        print(f"  - prompt_embeds (llama_vec): Shape={llama_vec.shape}, Dtype={llama_vec.dtype}, Device={llama_vec.device}")
        print(f"  - prompt_poolers (clip_l_pooler): Shape={clip_l_pooler.shape}, Dtype={clip_l_pooler.dtype}, Device={clip_l_pooler.device}")
        print(f"  - image_embeddings: Shape={image_embeddings_for_transformer.shape}, Dtype={image_embeddings_for_transformer.dtype}, Device={image_embeddings_for_transformer.device}")
        print(f"  - VRAM Free before sampling loop: {get_cuda_free_memory_gb(self.device):.2f} GB")
        print("="*40 + "\n")

        # --- ループ内でのビデオ生成とデコード ---
        final_video_frames_list = []
        final_video_frames_list.append(vae_decode(start_latent, f1_vae).cpu())

        for i_section in range(total_sections):
            if shared.state.interrupted: break
            shared.state.job = f"FramePack F1: Section {i_section + 1}/{total_sections}"
            shared.state.job_no = i_section + 1

            # ★★★ 修正箇所：リファレンス実装からコンテキスト管理ロジックを移植 ★★★
            current_history_gpu = history_latents.to(self.device)
            latent_window_size = getattr(framepack_f1_args, 'f1_latent_window_size', 9) # デフォルト9
            
            # 各種インデックスを計算
            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            # 履歴の長さに基づき、複数解像度のコンテキストを準備
            if current_history_gpu.shape[2] > (16 + 2 + 1):
                clean_latents_4x, clean_latents_2x, clean_latents_1x = current_history_gpu[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            else: # 初回など履歴が短い場合
                clean_latents_4x = torch.zeros((1, 16, 16, current_history_gpu.shape[3], current_history_gpu.shape[4]), device=self.device, dtype=current_history_gpu.dtype)
                clean_latents_2x = torch.zeros((1, 16, 2, current_history_gpu.shape[3], current_history_gpu.shape[4]), device=self.device, dtype=current_history_gpu.dtype)
                clean_latents_1x = current_history_gpu[:,:,-1:,:,:]

            clean_latents = torch.cat([start_latent.to(self.device), clean_latents_1x], dim=2)
            
            generated_latents = sample_hunyuan(
                transformer=f1_transformer,
                strength=framepack_f1_args.f1_image_strength,
                num_inference_steps=framepack_f1_args.f1_generation_latent_size,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=prompt_mask,
                prompt_poolers=clip_l_pooler,
                generator=generator,
                width=optimal_width, height=optimal_height,
                image_embeddings=image_embeddings_for_transformer,
                device=self.device,
                # --- 新しく追加するコンテキスト引数 ---
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
            )
            
            history_latents = torch.cat([history_latents, generated_latents.cpu()], dim=2)
            
            # ★★★ 修正箇所：逐次デコードと結合 ★★★
            try:
                print(f"[DEBUG] Decoding section {i_section + 1}...")
                move_model_to_device_with_memory_preservation(f1_vae, self.device)
                
                # デコードは最後のセクションのみ行い、soft_appendで結合
                section_to_decode = generated_latents
                overlap_frames = 4 # 例: 4フレーム分のoverlapで結合
                decoded_section = vae_decode(section_to_decode, f1_vae).cpu()

                if i_section == 0:
                     final_video_frames_list[0] = soft_append_bcthw(final_video_frames_list[0], decoded_section, overlap=overlap_frames)
                else:
                    last_video = final_video_frames_list.pop()
                    final_video_frames_list.append(soft_append_bcthw(last_video, decoded_section, overlap=overlap_frames))

                print(f"[DEBUG] VRAM Free after section VAE decoding: {get_cuda_free_memory_gb(self.device):.2f} GB")
            finally:
                offload_model_from_device_for_memory_preservation(f1_vae, self.device)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★

        final_video_frames = torch.cat(final_video_frames_list, dim=2)
        output_path = os.path.join(args.outdir, f"{root.timestring}_framepack_f1.mp4")
        save_bcthw_as_mp4(final_video_frames, output_path, fps=video_args.fps)
        print(f"[FramePack F1] Video saved to {output_path}")


    def cleanup_environment(self):
        """
        全てのマネージャーの後片付けを行い、ベースのSDモデルをVRAMに戻す。
        """
        if self.managers is None:
            print("Cleanup skipped: managers were not initialized.")
            return

        print("Cleaning up FramePack F1 environment...")
        
        for manager_name, manager in self.managers.items():
            if manager and hasattr(manager, 'dispose'):
                print(f"Disposing manager: {manager_name}...")
                manager.dispose()
        
        self.managers = None

        gc.collect()
        torch.cuda.empty_cache()

        print("Restoring base SD model to VRAM...")
        if self.sdxl_components and self.sdxl_components["unet"]:
            move_model_to_device_with_memory_preservation(self.sdxl_components["unet"], self.device)
        if self.sdxl_components and self.sdxl_components["vae"]:
            move_model_to_device_with_memory_preservation(self.sdxl_components["vae"], self.device)
        if self.sdxl_components and self.sdxl_components["text_encoders"]:
            move_model_to_device_with_memory_preservation(self.sdxl_components["text_encoders"], self.device)
        
        print("Cleanup complete.")
