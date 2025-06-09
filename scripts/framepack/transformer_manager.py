import os
import glob
import torch
import traceback
from accelerate import init_empty_weights
import gc

# --- ローカルのdiffusersからインポートすることを明示 ---
from scripts.diffusers import HunyuanVideoFramepackTransformer3DModel
from .memory import get_cuda_free_memory_gb # get_cuda_free_memory_gb を直接インポート

class TransformerManager:
    """
    transformerモデルの状態管理を行うクラス（修正版）
    責務：渡された単一のモデルパスに基づき、モデルのライフサイクルを管理する。
    """

    def __init__(self, device, high_vram_mode: bool, use_f1_model: bool, model_path: str):
        self.transformer = None
        self.device = device
        
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(f"TransformerManager received an invalid model_path: {model_path}")
        self.model_path = model_path
        print(f"TransformerManager initialized with model path: {self.model_path}")

        # 現在適用されている設定
        self.current_state = {
            'lora_paths': [],
            'lora_scales': [],
            'fp8_enabled': False,
            'is_loaded': False,
            'high_vram': high_vram_mode,
            'use_f1_model': use_f1_model
        }

        # 次回のロード時に適用する設定
        self.next_state = self.current_state.copy()

        # 仮想デバイスへのtransformerのロード
        self._load_virtual_transformer()
        print("Transformer model shell created on virtual device.")
        
    def set_next_settings(self, lora_paths=None, lora_scales=None, fp8_enabled=False, high_vram_mode=False, lora_path=None, lora_scale=None, force_dict_split=False):
        if lora_paths is None and lora_path is not None:
            lora_paths = [lora_path]
            lora_scales = [lora_scale]
        if lora_paths is not None and not isinstance(lora_paths, list):
            lora_paths = [lora_paths]
        if lora_scales is not None and not isinstance(lora_scales, list):
            lora_scales = [lora_scales]
        if lora_paths is not None and lora_paths and lora_scales is None:
            lora_scales = [0.8] * len(lora_paths)
        if lora_paths is not None and lora_scales is not None:
            if len(lora_scales) < len(lora_paths):
                lora_scales.extend([0.8] * (len(lora_paths) - len(lora_scales)))
            elif len(lora_scales) > len(lora_paths):
                lora_scales = lora_scales[:len(lora_paths)]
        
        self.next_state = {
            'lora_paths': lora_paths if lora_paths else [],
            'lora_scales': lora_scales if lora_scales else [],
            'fp8_enabled': fp8_enabled,
            'force_dict_split': force_dict_split,
            'high_vram': high_vram_mode,
            'is_loaded': self.current_state['is_loaded'],
            'use_f1_model': self.current_state['use_f1_model']
        }
    
    def _needs_reload(self):
        if not self._is_loaded():
            return True
        current_paths = self.current_state.get('lora_paths', []) or []
        next_paths = self.next_state.get('lora_paths', []) or []
        if len(current_paths) != len(next_paths) or set(current_paths) != set(next_paths):
            return True
        if next_paths:
            current_scales = self.current_state.get('lora_scales', [])
            next_scales = self.next_state.get('lora_scales', [])
            current_path_to_scale = {path: scale for path, scale in zip(current_paths, current_scales)}
            next_path_to_scale = {path: scale for path, scale in zip(next_paths, next_scales)}
            for path in next_paths:
                if current_path_to_scale.get(path) != next_path_to_scale.get(path):
                    return True
        if self.current_state.get('fp8_enabled') != self.next_state.get('fp8_enabled'):
            return True
        if self.current_state.get('force_dict_split', False) != self.next_state.get('force_dict_split', False):
            return True
        if self.current_state['high_vram'] != self.next_state['high_vram']:
            return True
        return False
    
    def _is_loaded(self):
        return self.transformer is not None and self.current_state['is_loaded']
    
    def get_transformer(self):
        return self.transformer

    def ensure_transformer_state(self):
        if self._needs_reload():
            return self._reload_transformer()        
        print("Using pre-loaded transformer model.")
        return True
    
    def _load_virtual_transformer(self):
        """仮想デバイスへ空のtransformerをロードする"""
        try:
            with init_empty_weights():
                config = HunyuanVideoFramepackTransformer3DModel.load_config(self.model_path, local_files_only=True)
                self.transformer = HunyuanVideoFramepackTransformer3DModel.from_config(config, torch_dtype=torch.bfloat16)
            self.transformer.to(torch.bfloat16)
        except Exception as e:
            print(f"Failed to create a virtual transformer from config at {self.model_path}")
            traceback.print_exc()
            raise e

    def _find_model_files(self):
        """self.model_path から状態辞書のファイルを取得する"""
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"The specified model directory does not exist: {self.model_path}")
        
        model_files = glob.glob(os.path.join(self.model_path, '**', '*.safetensors'), recursive=True)
        model_files = [f for f in model_files if os.path.basename(f).startswith('diffusion_pytorch_model')]
        
        model_files.sort()
        return model_files

    def _reload_transformer(self):
        try:
            if self.transformer is not None:
                ### ▼▼▼ IndentationError 修正箇所 ▼▼▼ ###
                # 既存のモデルを解放する処理が抜けていたため、インデントエラーが発生していました。
                # メモリを解放するためのコードをここに追加します。
                print("Disposing existing transformer model...")
                self.current_state['is_loaded'] = False
                del self.transformer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # ▼ デバッグログ追加 ▼
            print(f"[DEBUG] VRAM Free before Transformer reload: {get_cuda_free_memory_gb(self.device):.2f} GB")
            print("Reloading transformer...")
            print(f"Applying new settings from model path: {self.model_path}")

            # ▼ device_map="auto" を使ったロード処理 ▼
            print("Loading transformer with automatic device mapping (offloading)...")
            self.transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                device_map="auto",
                offload_folder="offload",
                offload_state_dict=True
            )
            
            # ▼ デバッグログ追加 ▼
            print("Transformer loaded via device_map.")
            try:
                # 実際にパラメータがどのデバイスにあるか確認
                print(f"  - Transformer execution device: {self.transformer.device}")
                print(f"  - Transformer sample parameter device: {next(self.transformer.parameters()).device}")
            except Exception as e:
                print(f"  - Could not check parameter device: {e}")
            print(f"[DEBUG] VRAM Free after Transformer reload: {get_cuda_free_memory_gb(self.device):.2f} GB")

            self.transformer.eval() 
            self.transformer.high_quality_fp32_output_for_inference = True 
            self.transformer.requires_grad_(False) 
            
            # ▼ Gradient Checkpointing 有効化 ▼
            if hasattr(self.transformer, 'enable_gradient_checkpointing'):
                print("Enabling gradient checkpointing to conserve VRAM during inference...")
                self.transformer.enable_gradient_checkpointing()
            
            self.next_state['is_loaded'] = True
            self.current_state = self.next_state.copy()

            print("Transformer reload complete.")
            return True
            
        except Exception as e:
            print(f"Transformer reload failed: {e}")
            traceback.print_exc()
            self.current_state['is_loaded'] = False
            return False
