import os
import torch
import gc
import numpy as np
from PIL import Image
import json # 念のためインポートは残します

from modules import shared
from modules.devices import cpu

from .memory import (
    offload_model_from_device_for_memory_preservation,
    move_model_to_device_with_memory_preservation,
    model_on_device,
    get_cuda_free_memory_gb,
)
from .transformer_manager import TransformerManager
from .text_encoder_manager import TextEncoderManager
from .k_diffusion_hunyuan import sample_hunyuan
from .hunyuan import vae_encode, vae_decode, encode_prompt_conds
from .utils import resize_and_center_crop, save_bcthw_as_mp4
from .discovery import FramepackDiscovery
from scripts.diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaTokenizerFast, CLIPTokenizer


class FramepackIntegration:
    """FramePack F1のモデル管理とビデオ生成を統合する司令塔クラス。"""

    def __init__(self, device):
        self.device = device
        self.managers = None
        self.sdxl_components = None
        
        self.discovery = FramepackDiscovery()

    def _initialize_managers(self, local_paths: dict[str, str]):
        """
        解決済みの絶対パスを受け取り、各モデルマネージャーを初期化する。
        """
        global_managers = {
            "transformer": None,
            "text_encoder": None,
            "vae": None,
            "tokenizers": None,
        }

        free_mem_gb = get_cuda_free_memory_gb(self.device)
        high_vram = free_mem_gb > 16

        print("Initializing managers with explicit model paths...")
        
        transformer_path = local_paths.get("transformer")
        global_managers["transformer"] = TransformerManager(
            device=self.device, 
            high_vram_mode=high_vram, 
            use_f1_model=True,
            model_path=transformer_path
        )

        text_encoder_path = local_paths.get("text_encoder")
        global_managers["text_encoder"] = TextEncoderManager(
            device=self.device, 
            high_vram_mode=high_vram,
            model_path=text_encoder_path
        )

        vae_path = local_paths.get("vae")

        class VaeManager:
            def __init__(self, device, model_path: str):
                self.model = None
                self.device = device
                self.model_path = model_path

            def get(self):
                if self.model is None:
                    print(f"Loading VAE from: {self.model_path}")
                    self.model = AutoencoderKLHunyuanVideo.from_pretrained(
                        self.model_path, subfolder="vae", torch_dtype=torch.float16, local_files_only=True
                    ).cpu()
                    self.model.eval()
                return self.model

        class TokenizerManager:
            def __init__(self, model_path: str):
                self.tokenizer = None
                self.tokenizer_2 = None
                self.model_path = model_path

            def get(self):
                if self.tokenizer is None:
                    print(f"Loading Tokenizers from: {self.model_path}")
                    self.tokenizer = LlamaTokenizerFast.from_pretrained(self.model_path, subfolder="tokenizer", local_files_only=True)
                    self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.model_path, subfolder="tokenizer_2", local_files_only=True)
                return self.tokenizer, self.tokenizer_2

        global_managers["vae"] = VaeManager(self.device, model_path=vae_path)
        global_managers["tokenizers"] = TokenizerManager(model_path=text_encoder_path)
        
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
        """
        モデルの存在チェックからマネージャーの初期化まで、実行環境をセットアップする。
        """
        print("Step 1: Checking for required local models...")
        models_exist, missing_repos = self.discovery.check_models_exist()

        if not models_exist:
            error_message = (
                "One or more FramePack F1 models are missing. Please download them manually before running Deforum.\n"
                "Missing repositories:\n"
            )
            for repo in missing_repos:
                error_message += f" - {repo}\n"
            error_message += "You can use the standalone downloader script if needed: 'python extensions/sd-forge-deforum/scripts/framepack/model_downloader.py'"
            raise FileNotFoundError(error_message)
        
        print("All required models found locally.")

        print("Step 2: Resolving local paths...")
        local_paths = {
            "transformer": self.discovery.get_local_path("transformer"),
            "text_encoder": self.discovery.get_local_path("text_encoder"),
            "vae": self.discovery.get_local_path("vae"),
        }
        
        for name, path in local_paths.items():
            if path is None or not os.path.isdir(path):
                raise RuntimeError(f"Failed to resolve a valid directory path for component '{name}': {path}")
            print(f"  - Resolved {name}: {path}")

        print("Step 3: Initializing model managers with resolved paths...")
        self.managers = self._initialize_managers(local_paths)

        print("Step 4: Offloading base SD model from VRAM...")
        self.sdxl_components = self._get_sdxl_components(shared.sd_model)
        if self.sdxl_components["unet"]:
            offload_model_from_device_for_memory_preservation(self.sdxl_components["unet"], self.device)
        if self.sdxl_components["vae"]:
            offload_model_from_device_for_memory_preservation(self.sdxl_components["vae"], self.device)
        if self.sdxl_components["text_encoders"]:
            offload_model_from_device_for_memory_preservation(self.sdxl_components["text_encoders"], self.device)
        
        gc.collect()
        torch.cuda.empty_cache()
        print("Environment setup complete.")

    def generate_video(self, args, anim_args, video_args, framepack_f1_args, root):
        managers = self.managers
        f1_vae = managers["vae"].get()
        f1_tokenizer, f1_tokenizer_2 = managers["tokenizers"].get()

        # --- ▼▼▼ 修正箇所 ▼▼▼ ---

        # 1. args.promptsが辞書(dict)であることを前提に、最初のプロンプトを安全に取得する
        prompt_text = ""
        prompts_schedule = args.prompts # args.promptsは既に辞書

        if not isinstance(prompts_schedule, dict) or not prompts_schedule:
            raise ValueError("Prompts are not in the expected dictionary format or are empty. Please check your Deforum prompt settings.")

        try:
            # キー(フレーム番号)を整数に変換してソートし、最初のキーを取得
            first_frame_key = sorted(prompts_schedule.keys(), key=int)[0]
            prompt_text = prompts_schedule[first_frame_key]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Could not extract the first prompt from the schedule: {e}")

        if not prompt_text:
            raise ValueError("The first prompt in the schedule is empty. Please provide a prompt.")

        print(f"[FramePack F1] Using single prompt for entire generation: '{prompt_text}'")
        
        # --- ▲▲▲ 修正完了 ▲▲▲ ---

        with model_on_device(f1_vae, self.device):
            init_image = np.array(Image.open(args.init_image).convert("RGB"))
            init_image = resize_and_center_crop(init_image, args.W, args.H)
            start_latent = vae_encode(init_image, f1_vae)

        managers["text_encoder"].ensure_text_encoder_state()
        f1_text_encoder, f1_text_encoder_2 = managers["text_encoder"].get_text_encoders()
        
        with model_on_device(f1_text_encoder, self.device), model_on_device(f1_text_encoder_2, self.device):
            llama_vec, clip_l_pooler = encode_prompt_conds(
                prompt_text,
                f1_text_encoder,
                f1_text_encoder_2,
                f1_tokenizer,
                f1_tokenizer_2,
            )
        f1_text_encoder.to(cpu)
        f1_text_encoder_2.to(cpu)

        managers["transformer"].ensure_transformer_state()
        f1_transformer = managers["transformer"].get_transformer()
        history_latents = start_latent.clone()
        total_sections = int(max(round((anim_args.max_frames) / (framepack_f1_args.f1_generation_latent_size * 4 - 3)), 1))

        with model_on_device(f1_transformer, self.device):
            history_latents = history_latents.to(self.device)
            llama_vec = llama_vec.to(self.device)
            clip_l_pooler = clip_l_pooler.to(self.device)
            
            for i_section in range(total_sections):
                shared.state.job = f"FramePack F1: Section {i_section + 1}/{total_sections}"
                shared.state.job_no = i_section + 1
                if shared.state.interrupted:
                    break
                    
                generated_latents = sample_hunyuan(
                    transformer=f1_transformer,
                    initial_latent=history_latents[:, :, -1:],
                    strength=framepack_f1_args.f1_image_strength,
                    steps=framepack_f1_args.f1_generation_latent_size,
                    llama_vec=llama_vec,
                    clip_l_pooler=clip_l_pooler,
                )
                history_latents = torch.cat([history_latents, generated_latents], dim=2)

        with model_on_device(f1_vae, self.device):
            final_video_frames = vae_decode(history_latents, f1_vae)

        output_path = os.path.join(args.outdir, f"{root.timestring}_framepack_f1.mp4")
        save_bcthw_as_mp4(final_video_frames, output_path, fps=video_args.fps)
        print(f"[FramePack F1] Video saved to {output_path}")

    def cleanup_environment(self):
        if self.managers is None:
            print("Cleanup skipped: managers were not initialized.")
            return

        f1_transformer = self.managers.get("transformer").get_transformer()
        if f1_transformer is not None and next(f1_transformer.parameters()).device.type != "meta":
            f1_transformer.to(cpu)
        
        f1_vae = self.managers.get("vae").get()
        if f1_vae is not None and next(f1_vae.parameters()).device.type != "meta":
            f1_vae.to(cpu)
        
        f1_text_encoder, f1_text_encoder_2 = self.managers.get("text_encoder").get_text_encoders()
        if f1_text_encoder is not None and next(f1_text_encoder.parameters()).device.type != "meta":
            f1_text_encoder.to(cpu)
        if f1_text_encoder_2 is not None and next(f1_text_encoder_2.parameters()).device.type != "meta":
            f1_text_encoder_2.to(cpu)

        gc.collect()
        torch.cuda.empty_cache()

        if self.sdxl_components["unet"]:
            move_model_to_device_with_memory_preservation(self.sdxl_components["unet"], self.device)
        if self.sdxl_components["vae"]:
            move_model_to_device_with_memory_preservation(self.sdxl_components["vae"], self.device)
        if self.sdxl_components["text_encoders"]:
            move_model_to_device_with_memory_preservation(self.sdxl_components["text_encoders"], self.device)
