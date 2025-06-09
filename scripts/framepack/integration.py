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
from .utils import resize_and_center_crop, save_bcthw_as_mp4, numpy2pytorch
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
    # __init__ から generate_video まで変更なしのため省略
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
            image_embeds = image_encoder_output.image_embeds
            print(f"[DEBUG] image_embeds created. Shape: {image_embeds.shape}, Device: {image_embeds.device}")
            print(f"[DEBUG] VRAM Free after Image Encoding: {get_cuda_free_memory_gb(self.device):.2f} GB")
        finally:
            offload_model_from_device_for_memory_preservation(f1_image_encoder, self.device)
            print("[DEBUG] Image Encoder offloaded from GPU.")

        try:
            print("[DEBUG] Moving VAE to GPU for encoding...")
            move_model_to_device_with_memory_preservation(f1_vae, self.device)
            init_image_np = np.array(pil_init_image)
            init_image_np = resize_and_center_crop(init_image_np, args.W, args.H)
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
        llama_vec = llama_vec.to(self.device)
        clip_l_pooler = clip_l_pooler.to(self.device)
        
        print(f"[DEBUG] Prompt conditioning created. llama_vec shape: {llama_vec.shape}, clip_l_pooler shape: {clip_l_pooler.shape}")
        
        print("[FramePack F1] Disposing text encoders...")
        managers["text_encoder"].dispose_text_encoders()

        gc.collect(); torch.cuda.empty_cache()
        print(f"[FramePack F1] VRAM cleaned. Free space: {get_cuda_free_memory_gb(self.device):.2f} GB")

        if not managers["transformer"].ensure_transformer_state():
            raise RuntimeError("Failed to load or setup the Transformer model. Check logs for OOM errors.")

        f1_transformer = managers["transformer"].get_transformer()

        history_latents = start_latent.clone()
        total_sections = int(max(round((anim_args.max_frames) / (framepack_f1_args.f1_generation_latent_size * 4 - 3)), 1))
        history_latents = history_latents.to(self.device)
        seed = args.seed if args.seed != -1 else torch.seed()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        print(f"[FramePack F1] Using seed: {seed}")

        print("\n" + "="*40)
        print("[DEBUG] PRE-SAMPLING CHECK")
        print(f"  - Target Device: {self.device}")
        try:
            print(f"  - Transformer Device: {f1_transformer.device}")
            print(f"  - Transformer Sample Param Device: {next(f1_transformer.parameters()).device}")
        except Exception as e:
            print(f"  - Could not determine transformer device: {e}")
        print(f"  - history_latents (initial_latent): Shape={history_latents.shape}, Dtype={history_latents.dtype}, Device={history_latents.device}")
        print(f"  - prompt_embeds (llama_vec): Shape={llama_vec.shape}, Dtype={llama_vec.dtype}, Device={llama_vec.device}")
        print(f"  - prompt_poolers (clip_l_pooler): Shape={clip_l_pooler.shape}, Dtype={clip_l_pooler.dtype}, Device={clip_l_pooler.device}")
        print(f"  - image_embeds: Shape={image_embeds.shape}, Dtype={image_embeds.dtype}, Device={image_embeds.device}")
        print(f"  - VRAM Free before sampling loop: {get_cuda_free_memory_gb(self.device):.2f} GB")
        print("="*40 + "\n")

        for i_section in range(total_sections):
            if shared.state.interrupted: break
            shared.state.job = f"FramePack F1: Section {i_section + 1}/{total_sections}"
            shared.state.job_no = i_section + 1

            generated_latents = sample_hunyuan(
                transformer=f1_transformer,
                initial_latent=history_latents[:, :, -1:],
                strength=framepack_f1_args.f1_image_strength,
                num_inference_steps=framepack_f1_args.f1_generation_latent_size,
                prompt_embeds=llama_vec,
                prompt_poolers=clip_l_pooler,
                generator=generator,
                width=args.W, height=args.H,
                image_embeddings=image_embeds,
                latent_indices=None,
                device=self.device,
            )
            history_latents = torch.cat([history_latents, generated_latents], dim=2)

        try:
            print("[DEBUG] Moving VAE to GPU for decoding...")
            move_model_to_device_with_memory_preservation(f1_vae, self.device)
            final_video_frames = vae_decode(history_latents, f1_vae)
            print(f"[DEBUG] VRAM Free after VAE decoding: {get_cuda_free_memory_gb(self.device):.2f} GB")
        finally:
            offload_model_from_device_for_memory_preservation(f1_vae, self.device)
            print("[DEBUG] VAE offloaded from GPU.")

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

        # ▼▼▼ 修正箇所 ▼▼▼
        print("Cleaning up FramePack F1 environment...")
        
        # 全てのマネージャーのdisposeを呼び出すことで、責務を各マネージャーに委譲する
        for manager_name, manager in self.managers.items():
            if manager and hasattr(manager, 'dispose'):
                print(f"Disposing manager: {manager_name}...")
                manager.dispose()
        # ▲▲▲ ここまで ▲▲▲
        
        # 全マネージャーへの参照を削除
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


### 補足：transformer_manager.py に追加すべき dispose メソッド

# 以下のメソッドを TransformerManager クラスの末尾に追加してください。

def dispose(self):
    """
    Transformerモデルの後片付けを行う。
    DynamicSwapInstallerのアンインストール、CPUへの移動、インスタンス削除を含む。
    """
    if not self.transformer:
        return

    print("Disposing Transformer model...")
    
    # 低VRAMモードでDynamicSwapInstallerを使用した場合、パッチを解除する
    if not self.current_state['high_vram']:
        print("Uninstalling DynamicSwapInstaller patches from Transformer...")
        DynamicSwapInstaller.uninstall_model(self.transformer)
    
    try:
        # モデルをCPUに移動
        self.transformer.to(cpu)
    except Exception as e:
        print(f"Could not move transformer to cpu: {e}")

    # 参照を削除
    del self.transformer
    self.transformer = None
    
    # 状態をリセット
    self.current_state['is_loaded'] = False

    print("Transformer disposed.")
