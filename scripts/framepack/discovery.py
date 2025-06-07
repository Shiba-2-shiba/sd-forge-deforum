from pathlib import Path

class FramepackDiscovery:
    """Locate FramePack F1 models inside the HuggingFace cache"""

    REQUIRED = {
        "unet": "lllyasviel/FramePack_F1_I2V_HY_20250503/unet/config.json",
        "text_encoder": "hunyuanvideo-community/HunyuanVideo/text_encoder/config.json",
        "tokenizer": "hunyuanvideo-community/HunyuanVideo/tokenizer/tokenizer.model",
        "vae": "hunyuanvideo-community/HunyuanVideo/vae/config.json",
    }

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface" / "hub"

    def discover_models(self) -> dict[str, bool]:
        results: dict[str, bool] = {"all_found": True}
        for name, rel in self.REQUIRED.items():
            path = self.cache_dir / ("models--" + rel.replace("/", "--"))
            exists = path.exists()
            results[name] = exists
            if not exists:
                results["all_found"] = False
        return results
