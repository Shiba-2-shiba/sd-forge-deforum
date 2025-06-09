import os
import glob
import torch
import traceback
from accelerate import init_empty_weights
import gc
# from safetensors.torch import load_file  <- 不要になるためコメントアウトまたは削除

# DynamicSwapInstallerをインポート
from .hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from .memory import get_cuda_free_memory_gb, DynamicSwapInstaller, cpu

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

        # この段階では仮想モデルは作成しない。リロード時に実体を直接ロードする。
        print("TransformerManager initialized. Model will be loaded on demand.")
        
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
        # このメソッドは現在使用されていませんが、将来的な拡張のために残すこともできます。
        # 今回の修正では `_reload_transformer` が実体を直接ロードするため、呼び出されません。
        pass

    def _find_model_files(self):
        # `from_pretrained` を使うため、このメソッドは不要になります。
        pass

    # ★★★★★ ここからが主要な修正箇所 ★★★★★
    def _reload_transformer(self):
        """
        モデルを再ロードする。
        Hugging Face の from_pretrained を使用し、メモリ効率よくロードする。
        """
        try:
            if self.transformer is not None:
                self.dispose()

            print(f"[DEBUG] VRAM Free before Transformer reload: {get_cuda_free_memory_gb(self.device):.2f} GB")
            print("Reloading transformer using 'from_pretrained' for memory efficiency...")
            print(f"Applying new settings from model path: {self.model_path}")

            # 1. メモリ効率の良い from_pretrained でモデルを直接ロード
            #    bfloat16 を指定し、まずCPUにロードすることでピークRAM使用量を抑える
            self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                self.model_path, 
                torch_dtype=torch.bfloat16
            ).cpu()
            
            print("Transformer model loaded successfully to CPU.")

            self.transformer.eval() 
            self.transformer.high_quality_fp32_output_for_inference = True 
            self.transformer.requires_grad_(False)
            
            # 2. VRAMモードに応じてデバイス配置を決定する
            if self.next_state['high_vram']:
                print("High VRAM mode: Moving entire transformer to GPU...")
                self.transformer.to(self.device)
            else:
                print("Low VRAM mode: Applying DynamicSwapInstaller...")
                DynamicSwapInstaller.install_model(self.transformer, device=self.device)
            
            print(f"[DEBUG] VRAM Free after Transformer setup: {get_cuda_free_memory_gb(self.device):.2f} GB")

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
            # 失敗した場合、transformerをNoneにしておく
            if self.transformer is not None:
                del self.transformer
                self.transformer = None
                gc.collect()
            return False
    # ★★★★★ ここまでが主要な修正箇所 ★★★★★

    def dispose(self):
        """
        Transformerモデルの後片付けを行う。
        DynamicSwapInstallerのアンインストール、CPUへの移動、インスタンス削除を含む。
        """
        if self.transformer is None:
            return

        print("Disposing Transformer model...")
        
        # is_loaded状態はリロードの成否に依存するため、high_vramモードはcurrent_stateから直接確認
        if self.current_state.get('is_loaded', False) and not self.current_state.get('high_vram', False):
            print("Uninstalling DynamicSwapInstaller patches from Transformer...")
            DynamicSwapInstaller.uninstall_model(self.transformer)
        
        try:
            self.transformer.to(cpu)
        except Exception as e:
            print(f"Could not move transformer to cpu: {e}")

        del self.transformer
        self.transformer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.current_state['is_loaded'] = False
        
        print("Transformer disposed.")
