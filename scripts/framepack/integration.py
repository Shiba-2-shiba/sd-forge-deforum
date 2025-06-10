# integration.py (最終確定版)

import os
import torch
import gc

from . import tensor_tool
from .memory import (
    offload_model_from_device_for_memory_preservation,
    move_model_to_device_with_memory_preservation,
    get_cuda_free_memory_gb,
)
from .transformer_manager import TransformerManager
from .text_encoder_manager import TextEncoderManager
from .vae_manager import VaeManager
from .discovery import FramepackDiscovery


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
            from transformers import CLIPVisionModelWithProjection
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
            from transformers import SiglipImageProcessor
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
            from transformers import LlamaTokenizerFast, CLIPTokenizer
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
        from modules.sd_models import FakeInitialModel
        if isinstance(sd_model, FakeInitialModel):
            return {"unet": None, "vae": None, "text_encoders": None}
            
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

    # ★★★ 修正点 1: メソッドの引数に`sd_model`を追加 ★★★
    def setup_environment(self, sd_model):
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
        # ★★★ 修正点 2: `shared.sd_model`の代わりに引数`sd_model`を使用 ★★★
        self.sdxl_components = self._get_sdxl_components(sd_model)
        if self.sdxl_components["unet"]: offload_model_from_device_for_memory_preservation(self.sdxl_components["unet"], self.device)
        if self.sdxl_components["vae"]: offload_model_from_device_for_memory_preservation(self.sdxl_components["vae"], self.device)
        if self.sdxl_components["text_encoders"]: offload_model_from_device_for_memory_preservation(self.sdxl_components["text_encoders"], self.device)
        
        gc.collect()
        torch.cuda.empty_cache()
        print("Environment setup complete.")

    def generate_video(self, args, anim_args, video_args, framepack_f1_args, root):
        """
        動画生成処理を外部モジュール `tensor_tool.py` に委譲する。
        """
        print("[FramePack Integration] Delegating video generation to tensor_tool module...")

        try:
            # tensor_toolから返されるのはPILイメージのリスト
            returned_images = tensor_tool.execute_generation(
                managers=self.managers,
                device=self.device,
                args=args,
                anim_args=anim_args,
                video_args=video_args,
                framepack_f1_args=framepack_f1_args,
                root=root
            )

            if returned_images and isinstance(returned_images, list) and len(returned_images) > 0:
                print(f"[FramePack Integration] Generation completed successfully. {len(returned_images)} images returned.")
            else:
                print("[FramePack Integration] Generation finished, but no images were returned from tensor_tool.")

            # Deforumのメイン処理はファイルシステム上の画像を直接扱うため、
            # ここで何かを返す必要はない。
            return None

        except Exception as e:
            print(f"[FramePack Integration] An error occurred during video generation delegated to tensor_tool.")
            raise e

    def cleanup_environment(self):
        if self.managers is None:
            print("Cleanup skipped: managers were not initialized.")
            return

        print("Cleaning up FramePack F1 environment...")
        
        for manager_name, manager in self.managers.items():
            if manager and hasattr(manager, 'dispose'):
                print(f"Disposing manager: {manager_name}...")
                manager.dispose()
        
        self.managers = None
        gc.collect(); torch.cuda.empty_cache()

        print("Restoring base SD model to VRAM...")
        if self.sdxl_components:
            if self.sdxl_components["unet"]: move_model_to_device_with_memory_preservation(self.sdxl_components["unet"], self.device)
            if self.sdxl_components["vae"]: move_model_to_device_with_memory_preservation(self.sdxl_components["vae"], self.device)
            if self.sdxl_components["text_encoders"]: move_model_to_device_with_memory_preservation(self.sdxl_components["text_encoders"], self.device)
        
        print("Cleanup complete.")
