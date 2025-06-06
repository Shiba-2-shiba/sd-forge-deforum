import torch
from ..diffusers import AutoencoderKLHunyuanVideo
from .memory import DynamicSwapInstaller, model_on_device

class VaeManager:
    def __init__(self, device, high_vram_mode=False):
        self.vae = None
        self.device = device
        self.high_vram_mode = high_vram_mode
        self.is_loaded = False

    def _load_model(self):
        print("Loading Hunyuan VAE...")
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder='vae',
            torch_dtype=torch.float16
        ).cpu()
        self.vae.eval()
        self.vae.requires_grad_(False)

        if self.high_vram_mode:
            self.vae.to(self.device)
        else:
            self.vae.enable_slicing()
            self.vae.enable_tiling()
            DynamicSwapInstaller.install_model(self.vae, device=self.device)

        self.is_loaded = True
        print("Hunyuan VAE loaded.")

    def get_model(self):
        if not self.is_loaded:
            self._load_model()
        return self.vae

    def dispose(self):
        if self.vae is not None:
            del self.vae
            self.vae = None
            self.is_loaded = False
            torch.cuda.empty_cache()
            print("Hunyuan VAE disposed.")
