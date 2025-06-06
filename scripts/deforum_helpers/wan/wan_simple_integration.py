#!/usr/bin/env python3
"""
Wan Simple Integration with Styled Progress and Enhanced VRAM Management
Updated to use experimental render core styling for progress indicators
and added VRAM logging for debugging memory issues.
Restored previously omitted helper methods.
Applied fix for TypeError: got multiple values for keyword argument 'width'.
Applied fix for TypeError: WanVace.prepare_source() got an unexpected keyword argument. (v3 using positional args for paths)
Applied fix for TypeError: WanVace.generate() got an unexpected keyword argument 'steps'.
Applied fix for TypeError: SmartVACEWrapper got multiple values for keyword argument 'guidance_scale'.
Applied fix for TypeError: WanVace.generate() got an unexpected keyword argument 'anim_args'.
Applied fix for ValueError: high is out of bounds for int32 in np.random.randint.
Applied fix for tensor permutation in _process_and_save_frames.
Applied fix for qwen_manager import path.
Added detailed debugging for frame processing and ensured contiguous numpy array for PIL.
"""

from pathlib import Path
from typing import List, Dict, Optional
import torch
import os
import numpy as np
import time
import sys
import json
from decimal import Decimal


# Import our new styled progress utilities
# utilsディレクトリ内のwan_progress_utilsから必要な関数をインポート
from .utils.wan_progress_utils import (
    WanModelLoadingContext, WanGenerationContext,
    print_wan_info, print_wan_success, print_wan_warning, print_wan_error, print_wan_progress,
    create_wan_clip_progress, create_wan_frame_progress, create_wan_inference_progress
)

# Import Deforum utilities for settings and audio handling
# Deforumのルートディレクトリにあるモジュールをインポート
from ..video_audio_utilities import download_audio
from ..subtitle_handler import init_srt_file, write_frame_subtitle, calculate_frame_duration
from ..settings import save_settings_from_animation_run, get_deforum_version # Added get_deforum_version import

# Helper function to get VRAM stats
def get_vram_stats(context_message: str):
    # VRAM統計情報を取得するヘルパー関数
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2 # PyTorch 1.7+
        # VRAM情報を表示（デバッグ用）
        print_wan_info(f"DEBUG VRAM ({context_message}): Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB, Max Reserved: {max_reserved:.2f} MiB")
    else:
        print_wan_info(f"DEBUG VRAM ({context_message}): CUDA not available.")

class WanSimpleIntegration:
    """Simplified Wan integration with auto-discovery, proper progress styling, and VRAM management."""
    
    def __init__(self, device='cuda'):
        # 初期化メソッド
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models = []
        self.pipeline = None
        self.model_size = None
        self.optimal_width = 720 # Default, might be overridden by model
        self.optimal_height = 480 # Default, might be overridden by model
        self.flash_attention_mode = "auto"  # auto, enabled, disabled
        print_wan_info(f"Simple Integration initialized on {self.device}")
        get_vram_stats("WanSimpleIntegration.__init__")
    
    def discover_models(self) -> List[Dict]:
        """Discover available Wan models with styled progress"""
        # 利用可能なWanモデルを検出するメソッド
        models = []
        # モデル検索パスのリスト
        search_paths = [
            Path("models/wan"), # WebUIのmodels/wan
            Path("models/Wan"), # 大文字のWanも考慮
            Path("models"), # models直下も検索（直接配置されている場合）
            Path("../models/wan"), # スクリプトからの相対パス
            Path("../models/Wan"),
            Path("/tmp/stable-diffusion-webui-forge/models/wan"), # 特定環境の絶対パス例
        ]
        
        print_wan_progress("Discovering Wan models...")
        
        for search_path in search_paths:
            abs_search_path = search_path.resolve() # 絶対パスを取得
            print_wan_info(f"🔍 Searching in: {abs_search_path} (Exists: {abs_search_path.exists()})")
            if abs_search_path.exists() and abs_search_path.is_dir():
                for model_dir_item in abs_search_path.iterdir():
                    # ディレクトリ名が'wan'または'vace'で始まるか、'wan'を含む場合
                    if model_dir_item.is_dir() and (model_dir_item.name.lower().startswith(('wan', 'vace')) or "wan" in model_dir_item.name.lower()):
                        model_info = self._analyze_model_directory(model_dir_item)
                        if model_info:
                            models.append(model_info)
                            print_wan_success(f"Found: {model_info['name']} ({model_info['type']}, {model_info['size']}) at {model_info['path']}")
        
        if not models:
            print_wan_warning("No Wan models found in specified search paths. Please check paths and model names (e.g., Wan2.1-VACE-1.3B).")
        else:
            print_wan_success(f"Discovery complete - found {len(models)} model(s)")
            
        self.models = models # 発見したモデルをインスタンス変数に保存
        return models
    
    def _analyze_model_directory(self, model_dir: Path) -> Optional[Dict]:
        """Analyze a model directory and return model info if valid"""
        # モデルディレクトリを解析し、有効な場合はモデル情報を返す内部メソッド
        if not model_dir.is_dir():
            return None
            
        model_name_lower = model_dir.name.lower()
        
        # Wanモデルである可能性をより堅牢にチェック
        is_wan_model_candidate = 'wan' in model_name_lower or \
                                 'vace' in model_name_lower or \
                                 any(file.name.lower().startswith(('wan', 'vace')) for file in model_dir.rglob('*') if file.is_file())
        
        if not is_wan_model_candidate:
            # print_wan_info(f"Skipping non-Wan directory: {model_dir.name}")
            return None
        
        # 必要なファイル（config.jsonやモデルの重みファイルなど）が存在するか確認
        if not self._has_required_files(model_dir):
            print_wan_warning(f"Skipping {model_dir.name}: Missing required files (e.g., config.json, model weights).")
            return None
        
        model_type = "Unknown"
        model_size_str = "Unknown" # self.model_sizeとの衝突を避けるために改名
        
        if 'vace' in model_name_lower:
            model_type = "VACE"
        elif 't2v' in model_name_lower: # Text-to-Video
            model_type = "T2V"
        elif 'i2v' in model_name_lower: # Image-to-Video
            model_type = "I2V"
        
        if '1.3b' in model_name_lower:
            model_size_str = "1.3B"
        elif '14b' in model_name_lower: # 14Bの可能性を想定
            model_size_str = "14B"
        
        return {
            'name': model_dir.name,
            'path': str(model_dir.resolve()), # 解決された絶対パスを使用
            'type': model_type,
            'size': model_size_str,
            'directory': model_dir
        }
    
    def _has_required_files(self, model_dir: Path) -> bool:
        """Check if model directory has required files like config and weights."""
        # モデルディレクトリに必要なファイル（configや重みなど）があるか確認する内部メソッド
        has_config = (model_dir / "config.json").exists()
        
        # 様々なモデルの重みファイルの拡張子をチェック
        weight_extensions = ['.safetensors', '.bin', '.pt', '.pth']
        has_weights = any(
            file.suffix.lower() in weight_extensions
            for file in model_dir.rglob('*') # サブディレクトリも検索
            if file.is_file() and file.stat().st_size > 1024 * 1024 # 空でないファイルの基本的なチェック (例: > 1MB)
        )
        
        # model_index.jsonもHugging Faceモデルでは一般的
        has_model_index = (model_dir / "model_index.json").exists()

        # VACEの場合、特定のファイルが期待される
        if "vace" in model_dir.name.lower():
            # ユーザー提供のファイルリストに基づいたVACEのより具体的なチェック
            has_diffusion_pytorch_model = (model_dir / "diffusion_pytorch_model.safetensors").exists()
            has_t5_model = (model_dir / "models_t5_umt5-xxl-enc-bf16.pth").exists()
            has_vae_model = (model_dir / "Wan2.1_VAE.pth").exists()
            
            # print_wan_info(f"VACE Check for {model_dir.name}: config: {has_config}, diffusion: {has_diffusion_pytorch_model}, t5: {has_t5_model}, vae: {has_vae_model}")
            return has_config and has_diffusion_pytorch_model and has_t5_model and has_vae_model

        return (has_config or has_model_index) and has_weights

    # --- Restored Helper Methods Start ---
    # 以前省略されていたヘルパーメソッドを復元
    def _validate_vace_weights(self, model_path: Path) -> bool:
        """Validate that VACE model has required weights - compatibility method for UI"""
        # VACEモデルが必要な重みを持っているか検証するメソッド（UI互換性のため）
        try:
            # この特定の検証フィードバックについては、元のスクリプトに従って標準のprintを使用
            print(f"🔍 Validating VACE model: {model_path.name}")
            
            # メインの拡散モデルファイルのチェック（safetensorsを優先）
            diffusion_model_sf = model_path / "diffusion_pytorch_model.safetensors"
            # T5モデルのチェック
            t5_model_file = model_path / "models_t5_umt5-xxl-enc-bf16.pth"
            # VAEモデルのチェック
            vae_model_file = model_path / "Wan2.1_VAE.pth"
            # configファイルのチェック
            config_file = model_path / "config.json"

            if not config_file.exists():
                print(f"❌ VACE model {model_path.name} missing config.json")
                return False
            if not diffusion_model_sf.exists():
                print(f"❌ VACE model {model_path.name} missing diffusion_pytorch_model.safetensors")
                return False
            if not t5_model_file.exists():
                 print(f"❌ VACE model {model_path.name} missing models_t5_umt5-xxl-enc-bf16.pth")
                 return False
            if not vae_model_file.exists():
                print(f"❌ VACE model {model_path.name} missing Wan2.1_VAE.pth")
                return False

            # 主要コンポーネントの基本的なファイルサイズチェック
            if diffusion_model_sf.stat().st_size < 1_000_000_000:  # 約1GB、VACE拡散モデルは大きい（例: 6.7G）
                print(f"❌ VACE diffusion model file too small: {diffusion_model_sf.stat().st_size / (1024**2):.2f} MiB for {model_path.name}")
                return False
            if t5_model_file.stat().st_size < 1_000_000_000: # 約1GB、T5モデルは非常に大きい（例: 11G）
                print(f"❌ VACE T5 model file too small: {t5_model_file.stat().st_size / (1024**2):.2f} MiB for {model_path.name}")
                return False
            if vae_model_file.stat().st_size < 100_000_000: # 約100MB、VAEモデルは小さいが実質的（例: 485M）
                 print(f"❌ VACE VAE model file too small: {vae_model_file.stat().st_size / (1024**2):.2f} MiB for {model_path.name}")
                 return False
            
            # configの内容をロードして検証しようと試みる
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # VACE configに期待されるキーのチェック（例）
                # これらはWan2.1-VACEの実際のconfig構造によって異なる場合がある
                expected_class_name = config_data.get("_class_name", "").lower()
                if "vace" not in expected_class_name and "wanvacepipeline" not in expected_class_name : # クラス名の例
                    print(f"⚠️ Model config for {model_path.name} doesn't strongly indicate VACE type via _class_name: '{expected_class_name}'. Proceeding with caution.")
                
                print(f"✅ VACE model validation passed for: {model_path.name}")
                return True
                
            except json.JSONDecodeError as config_e_json:
                print(f"❌ VACE config.json for {model_path.name} is not valid JSON: {config_e_json}")
                return False
            except Exception as config_e:
                print(f"⚠️ VACE config content validation failed for {model_path.name}: {config_e}. Model might still work.")
                return True # 基本的なファイルが存在するがconfig内容のチェックに失敗した場合は続行を許可
            
        except Exception as e:
            print(f"❌ VACE validation error for {model_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _has_incomplete_models(self) -> bool:
        """Check if there are incomplete models - compatibility method for UI"""
        # 不完全なモデルがあるか確認するメソッド（UI互換性のため）
        try:
            incomplete_models = self._check_for_incomplete_models()
            return len(incomplete_models) > 0
        except Exception as e:
            print_wan_warning(f"⚠️ Error checking for incomplete models: {e}")
            return False
    
    def _check_for_incomplete_models(self) -> List[Path]:
        """Check for incomplete model downloads - compatibility method for UI"""
        # 不完全なモデルダウンロードを確認するメソッド（UI互換性のため）
        # これは基本的なチェックです。より高度なチェックには、チェックサムやメタデータが含まれる場合があります。
        incomplete = []
        if not self.models: # モデルが検出されていることを確認
            self.discover_models()

        for model_info in self.models:
            model_dir = Path(model_info['path'])
            if not self._has_required_files(model_dir): # 改良された_has_required_filesを使用
                print_wan_warning(f"Potentially incomplete model (missing required files): {model_dir.name}")
                incomplete.append(model_dir)
            elif model_info['type'] == "VACE": # VACEにはより具体的な要件がある
                 if not self._validate_vace_weights(model_dir): # 検証メソッドを使用
                    print_wan_warning(f"VACE model failed validation (potentially incomplete): {model_dir.name}")
                    incomplete.append(model_dir)
        return incomplete

    def _fix_incomplete_model(self, model_dir: Path, downloader=None) -> bool:
        """Fix incomplete model - compatibility method for UI (provides instructions)"""
        # 不完全なモデルを修正するメソッド（UI互換性のため、指示を提供する）
        # このメソッドは主にガイダンスを提供します。自動修正は危険な場合があるためです。
        try:
            print_wan_warning(f"🔧 Model '{model_dir.name}' appears incomplete or corrupted.")
            print_wan_info(f"💡 To fix incomplete model '{model_dir.name}':")
            print_wan_info(f"   1. Manually delete the directory: {model_dir}")
            print_wan_info(f"   2. Re-download the model using HuggingFace CLI or other means.")
            
            model_name_lower = model_dir.name.lower()
            download_cmd = "huggingface-cli download <repo_id> --local-dir <your_path>" # 一般的
            if 'wan2.1-vace-1.3b' in model_name_lower:
                download_cmd = f"huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir {model_dir.parent}/Wan2.1-VACE-1.3B --local-dir-use-symlinks False"
            elif 'wan2.1-vace-14b' in model_name_lower: # 14Bバリアントを想定
                download_cmd = f"huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir {model_dir.parent}/Wan2.1-VACE-14B --local-dir-use-symlinks False" 
            # 必要に応じて他の既知のモデルの具体的なダウンロードコマンドを追加

            print_wan_info(f"   Example download command: {download_cmd}")
            print_wan_info(f"   Ensure the download completes fully.")
            
            # 手動介入が必要であるか、またはガイドされたことを示すためにFalseを返す。
            return False
            
        except Exception as e:
            print_wan_error(f"❌ Error providing fix instructions for {model_dir.name}: {e}")
            return False
            
    def _process_and_save_frames(self, result, clip_idx: int, output_dir: str, timestring: str, start_frame_idx: int, frame_progress=None):
        """Process generation result and save frames with progress tracking."""
        # 生成結果を処理し、進捗追跡付きでフレームを保存するメソッド
        try:
            from PIL import Image

            frames_pil = [] # PILイメージを保存

            frames_data = None
            if isinstance(result, tuple) and len(result) > 0:
                frames_data = result[0] 
            elif hasattr(result, 'frames'):
                frames_data = result.frames
            else:
                frames_data = result

            if frames_data is None:
                print_wan_warning(f"No frame data found in result for clip {clip_idx + 1}.")
                return []

            if torch.is_tensor(frames_data):
                print_wan_info(f"Processing tensor output: {frames_data.shape}, dtype: {frames_data.dtype}, device: {frames_data.device}")
                frames_tensor = frames_data.detach().cpu()

                if frames_tensor.ndim == 5 and frames_tensor.shape[0] == 1: 
                    frames_tensor = frames_tensor.squeeze(0) 

                if frames_tensor.ndim == 4:
                    if frames_tensor.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                        if frames_tensor.min() < -0.5: 
                            frames_tensor = (frames_tensor + 1.0) / 2.0
                        frames_tensor = (frames_tensor.clamp(0, 1) * 255).byte()
                    
                    # テンソルの形状に応じて適切なpermuteを実行
                    # 想定される入力形状: (C, F, H, W) または (F, C, H, W)
                    # 出力目標形状: (F, H, W, C) for PIL
                    
                    # (C, F, H, W) の場合
                    if frames_tensor.shape[0] == 3 or frames_tensor.shape[0] == 1: 
                        print_wan_info(f"Tensor shape {frames_tensor.shape} detected as (Channels, Frames, Height, Width). Permuting to (Frames, Height, Width, Channels).")
                        frames_tensor_fhwc = frames_tensor.permute(1, 2, 3, 0)
                    # (F, C, H, W) の場合
                    elif frames_tensor.shape[1] == 3 or frames_tensor.shape[1] == 1: 
                        print_wan_info(f"Tensor shape {frames_tensor.shape} detected as (Frames, Channels, Height, Width). Permuting to (Frames, Height, Width, Channels).")
                        frames_tensor_fhwc = frames_tensor.permute(0, 2, 3, 1)
                    else:
                        print_wan_warning(f"Unexpected tensor shape for clip {clip_idx + 1}: {frames_tensor.shape}. Cannot reliably determine channel/frame order.")
                        return []
                    
                    print_wan_info(f"DEBUG: frames_tensor_fhwc shape after permute: {frames_tensor_fhwc.shape}, dtype: {frames_tensor_fhwc.dtype}")

                    for i in range(frames_tensor_fhwc.shape[0]): 
                        frame_np = frames_tensor_fhwc[i].numpy() 
                        print_wan_info(f"DEBUG: frame_np original shape for frame {i}: {frame_np.shape}, dtype: {frame_np.dtype}")
                        
                        if frame_np.shape[-1] == 1: 
                            frame_np = frame_np.squeeze(-1)
                            print_wan_info(f"DEBUG: frame_np after squeeze for frame {i}: {frame_np.shape}, dtype: {frame_np.dtype}")
                        
                        frame_np_contiguous = np.ascontiguousarray(frame_np) # PILのために連続配列を保証
                        print_wan_info(f"DEBUG: frame_np_contiguous shape for frame {i}: {frame_np_contiguous.shape}, dtype: {frame_np_contiguous.dtype}")

                        try:
                            pil_img = Image.fromarray(frame_np_contiguous)
                            frames_pil.append(pil_img)
                        except Exception as e_pil_fromarray:
                            print_wan_error(f"Error converting frame {i} to PIL Image: {e_pil_fromarray}. frame_np_contiguous shape: {frame_np_contiguous.shape}, dtype: {frame_np_contiguous.dtype}")
                            continue # このフレームをスキップして次に進む

                        if frame_progress: frame_progress.update(1)
                else:
                    print_wan_warning(f"Unexpected tensor ndim for clip {clip_idx + 1}: {frames_tensor.ndim}. Cannot process into frames.")
                    return []

            elif isinstance(frames_data, list): 
                if all(isinstance(f, Image.Image) for f in frames_data):
                    frames_pil = frames_data
                    if frame_progress: [frame_progress.update(1) for _ in frames_pil]
                elif all(isinstance(f, np.ndarray) for f in frames_data): 
                    for frame_np_item in frames_data:
                        if frame_np_item.dtype in [np.float32, np.float64]:
                            if frame_np_item.min() < -0.5: frame_np_item = (frame_np_item + 1.0) / 2.0
                            frame_np_item = (np.clip(frame_np_item, 0, 1) * 255).astype(np.uint8)
                        else:
                            frame_np_item = np.clip(frame_np_item, 0, 255).astype(np.uint8)
                        if frame_np_item.shape[-1] == 1: frame_np_item = frame_np_item.squeeze(-1)
                        frames_pil.append(Image.fromarray(frame_np_item))
                        if frame_progress: frame_progress.update(1)
                else:
                    print_wan_warning(f"Mixed or unsupported list content for frames_data in clip {clip_idx + 1}.")
                    return []
            else:
                print_wan_warning(f"Unsupported frames_data type for clip {clip_idx + 1}: {type(frames_data)}")
                return []

            saved_paths = []
            if not frames_pil:
                print_wan_warning(f"No PIL images to save for clip {clip_idx + 1}")
                return []

            for i, pil_image in enumerate(frames_pil):
                actual_frame_index = start_frame_idx + i
                frame_filename = f"{timestring}_{actual_frame_index:09d}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                
                try:
                    pil_image.save(frame_path, 'PNG')
                    saved_paths.append(frame_path)
                except Exception as save_e:
                    print_wan_warning(f"Failed to save frame {actual_frame_index} (clip {clip_idx+1}, item {i}): {save_e}")
                    continue
            
            if saved_paths:
                print_wan_success(f"Saved {len(saved_paths)} frames for clip {clip_idx + 1} (Indices {start_frame_idx} to {start_frame_idx + len(saved_paths) - 1})")
            else:
                print_wan_warning(f"No frames were successfully saved for clip {clip_idx + 1}")
            return saved_paths
            
        except Exception as e:
            print_wan_error(f"Frame processing and saving failed for clip {clip_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def test_wan_setup(self) -> bool:
        """Test if Wan setup is working with styled output"""
        # Wanセットアップがスタイル付き出力で動作するかテストするメソッド
        try:
            print_wan_progress("Testing Wan setup...")
            
            models = self.discover_models()
            if not models:
                print_wan_error("No Wan models found during setup test.")
                # 可能であれば、より具体的なアドバイスを提供
                if self._has_incomplete_models():
                    print_wan_warning("Found potentially incomplete models. Please check and re-download if necessary.")
                return False
            
            best_model = self.get_best_model()
            if not best_model:
                print_wan_error("No suitable Wan model could be selected for testing.")
                return False
            
            print_wan_success(f"Wan setup test passed: Found {len(models)} model(s), selected '{best_model['name']}' as best.")
            return True
            
        except Exception as e:
            print_wan_error(f"Wan setup test failed critically: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_wan_settings_and_metadata(self, output_dir: str, timestring: str, clips: List[Dict], 
                                       model_info: Dict, wan_args=None, **kwargs) -> Optional[str]:
        """Save Wan generation settings to match normal Deforum format"""
        # Wan生成設定を通常のDeforum形式に合わせて保存するメソッド
        try:
            settings_filename = os.path.join(output_dir, f"{timestring}_wan_settings.txt") # Deforum設定と区別
            
            settings = {
                "wan_model_name": model_info['name'],
                "wan_model_type": model_info['type'],
                "wan_model_size": model_info['size'],
                "wan_model_path": model_info['path'],
                "wan_flash_attention_mode": self.flash_attention_mode,
                
                "width": kwargs.get('width', self.optimal_width),
                "height": kwargs.get('height', self.optimal_height),
                "num_inference_steps": kwargs.get('num_inference_steps', 20), # DummyWanArgsからのデフォルト
                "guidance_scale": kwargs.get('guidance_scale', 7.5), # デフォルト
                "seed": kwargs.get('seed', -1), # デフォルト
                "fps": kwargs.get('fps', getattr(wan_args, 'fps', 10)), # wan_argsから取得、またはデフォルト
                
                "total_clips": len(clips),
                "total_frames_generated_this_run": sum(clip.get('num_frames',0) for clip in clips), # 概算
                "clips_data": clips, # 生成に使用されたクリップ構造を保存
                
                "generation_mode": "wan_video_generation", # 特定のモード
                "timestring": timestring,
                "output_directory": output_dir,
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())),
                "device": str(self.device),
                
                "wan_integration_version": "1.1.9_tensor_debug", # このスクリプトのバージョン
                "deforum_version_info": self._get_deforum_version(),
            }
            
            if wan_args:
                # 関連するwan_argsを追加。機密情報や冗長すぎるオブジェクトをダンプしないように注意
                relevant_wan_args = {
                    k: v for k, v in vars(wan_args).items() 
                    if isinstance(v, (str, int, float, bool, list, dict)) and not k.startswith('_')
                }
                settings["wan_args_snapshot"] = relevant_wan_args
            
            # その他の有用なkwargsを追加
            for k, v in kwargs.items():
                if k not in settings and isinstance(v, (str, int, float, bool, list, dict)):
                    settings[f"kwarg_{k}"] = v

            with open(settings_filename, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4, default=str) # シリアライズ不可能な場合はdefault=str
            
            print_wan_success(f"Wan settings and metadata saved to: {os.path.basename(settings_filename)}")
            return settings_filename
            
        except Exception as e:
            print_wan_error(f"Failed to save Wan settings: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_wan_srt_file(self, output_dir: str, timestring: str, clips: List[Dict], 
                           fps: float = 10.0) -> Optional[str]:
        """Create SRT subtitle file for Wan generation based on clips data"""
        # クリップデータに基づいてWan生成用のSRT字幕ファイルを作成するメソッド
        try:
            srt_filename = os.path.join(output_dir, f"{timestring}_wan.srt")
            
            if not clips:
                print_wan_warning("No clips data provided, cannot create SRT file.")
                return None

            if fps <= 0: 
                print_wan_error(f"Invalid FPS value for SRT: {fps}. Cannot create SRT.")
                return None
            frame_duration_sec = 1.0 / fps

            with open(srt_filename, "w", encoding="utf-8") as f:
                current_time_sec = 0.0
                for idx, clip_data in enumerate(clips):
                    prompt_text = clip_data.get('prompt', f"Clip {idx+1}")[:100] # 長さを制限
                    num_frames_in_clip = clip_data.get('num_frames', 0)
                    if num_frames_in_clip <= 0:
                        continue

                    start_time_srt = self._time_to_srt_format(Decimal(current_time_sec))
                    clip_duration_sec = num_frames_in_clip * frame_duration_sec
                    end_time_sec = current_time_sec + clip_duration_sec
                    end_time_srt = self._time_to_srt_format(Decimal(end_time_sec))

                    f.write(f"{idx + 1}\n")
                    f.write(f"{start_time_srt} --> {end_time_srt}\n")
                    f.write(f"{prompt_text}\n\n")
                    
                    current_time_sec = end_time_sec # 次のクリップのために時間を進める
            
            print_wan_success(f"Wan SRT file created: {os.path.basename(srt_filename)}")
            return srt_filename
            
        except Exception as e:
            print_wan_error(f"Failed to create Wan SRT file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def download_and_cache_audio(self, audio_url: str, output_dir: str, timestring: str) -> Optional[str]:
        """Download and cache audio file in output directory"""
        # 出力ディレクトリにオーディオファイルをダウンロードしてキャッシュするメソッド
        try:
            if not audio_url or not audio_url.strip().startswith(('http://', 'https://')):
                print_wan_info(f"Audio path is not a downloadable URL: {audio_url}. Assuming local path or no audio.")
                return audio_url # URLでないか空の場合はそのまま返す
            
            print_wan_progress(f"Downloading audio from: {audio_url}")
            
            # Deforumのdownload_audioユーティリティを使用
            temp_audio_path = download_audio(audio_url) # ダウンロード失敗時に例外が発生する可能性あり
            
            if not temp_audio_path or not os.path.exists(temp_audio_path):
                print_wan_error(f"Audio download failed or temp file not found for URL: {audio_url}")
                return None

            # ファイル拡張子を決定
            _, ext = os.path.splitext(audio_url.split('?')[0]) # 拡張子のためにクエリパラメータを削除
            if not ext or len(ext) > 5 : # 有効な拡張子の基本的なチェック
                 # 元のURLがおかしい場合は一時ファイルから拡張子を取得しようと試みる
                _, temp_ext = os.path.splitext(temp_audio_path)
                ext = temp_ext if temp_ext else '.mp3' # すべて失敗した場合はMP3をデフォルトとする

            cached_audio_filename = f"{timestring}_soundtrack{ext}"
            cached_audio_path = os.path.join(output_dir, cached_audio_filename)
            
            import shutil
            shutil.copy2(temp_audio_path, cached_audio_path)
            print_wan_success(f"Audio downloaded and cached as: {cached_audio_filename}")
            
            # download_audioからの一時ファイルをクリーンアップ
            try:
                os.unlink(temp_audio_path)
            except OSError as e_unlink:
                print_wan_warning(f"Could not delete temporary audio file {temp_audio_path}: {e_unlink}")
            
            return cached_audio_path
            
        except Exception as e:
            print_wan_error(f"Failed to download or cache audio from {audio_url}: {e}")
            import traceback
            traceback.print_exc()
            return None # 失敗を示す

    def _time_to_srt_format(self, seconds_decimal: Decimal) -> str:
        """Convert seconds (as Decimal) to SRT time format (HH:MM:SS,mmm)"""
        # 秒数（Decimal型）をSRT時間形式（HH:MM:SS,mmm）に変換する内部メソッド
        if not isinstance(seconds_decimal, Decimal):
            try:
                seconds_decimal = Decimal(str(seconds_decimal)) # Decimalでない場合は変換を試みる
            except:
                print_wan_warning(f"Invalid input for _time_to_srt_format: {seconds_decimal}. Using 0.")
                seconds_decimal = Decimal(0)

        total_seconds_int = int(seconds_decimal)
        milliseconds = int((seconds_decimal - total_seconds_int) * 1000)
        
        hours = total_seconds_int // 3600
        remainder = total_seconds_int % 3600
        minutes = remainder // 60
        seconds = remainder % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _get_deforum_version(self) -> str:
        """Get Deforum version/commit ID using the imported function."""
        # インポートされた関数を使用してDeforumのバージョン/コミットIDを取得する内部メソッド
        try:
            return get_deforum_version() 
        except NameError: 
            print_wan_warning("get_deforum_version not found directly, trying dynamic import for settings.")
            try:
                from .. import settings as deforum_settings 
                if hasattr(deforum_settings, 'get_deforum_version'):
                    return deforum_settings.get_deforum_version()
            except ImportError:
                pass 
            except AttributeError:
                pass 
            return "unknown (get_deforum_version not accessible)"
        except Exception as e:
            print_wan_warning(f"Error getting Deforum version: {e}")
            return "unknown (error)"
    # --- Restored Helper Methods End ---

    def load_simple_wan_pipeline(self, model_info: Dict, wan_args=None) -> bool:
        """Load Wan pipeline with VRAM management and styled progress indicators"""
        # VRAM管理とスタイル付き進捗インジケータでWanパイプラインをロードするメソッド
        model_name = model_info['name']
        print_wan_info(f"🎬 Attempting to load model: {model_name} ({model_info['type']}, {model_info['size']}) from {model_info['path']}")
        get_vram_stats(f"Start of load_simple_wan_pipeline for {model_name}")

        # --- STEP 1: Qwen Unload and Cache Clear ---
        # QwenモデルのアンロードとCUDAキャッシュのクリア
        try:
            print_wan_info("Attempting to unload Qwen model and clear CUDA cache...")
            
            qwen_manager_instance = None # qwen_managerインスタンスを保持する変数
            try:
                # 修正箇所: 相対インポートのパスを修正 ( ..utils -> .utils )
                from .utils.qwen_manager import qwen_manager as qm_instance 
                qwen_manager_instance = qm_instance 
                print_wan_info("Successfully imported qwen_manager.")
            except ImportError:
                print_wan_warning("qwen_manager could not be imported. Qwen model might not be unloaded automatically.")
            except Exception as e:
                print_wan_error(f"Error importing qwen_manager: {e}")

            if qwen_manager_instance and hasattr(qwen_manager_instance, 'is_model_loaded') and hasattr(qwen_manager_instance, 'ensure_model_unloaded'):
                if qwen_manager_instance.is_model_loaded():
                    print_wan_info("Qwen model is loaded. Unloading to free VRAM...")
                    qwen_manager_instance.ensure_model_unloaded() 
                    print_wan_success("Qwen model unloaded successfully.")
                else:
                    print_wan_info("Qwen model is not currently loaded or manager indicates so.")
            else:
                if not qwen_manager_instance: 
                    print_wan_warning("qwen_manager was not imported. Skipping Qwen unload.")
                else: 
                    print_wan_warning("qwen_manager does not have expected methods (is_model_loaded, ensure_model_unloaded). Skipping Qwen unload.")


            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print_wan_success("CUDA cache cleared.")
            get_vram_stats("After Qwen unload and cache clear")

        except Exception as e:
            print_wan_error(f"Error during Qwen unload or cache clear: {e}")
            import traceback
            traceback.print_exc()
        # --- End of STEP 1 ---

        try:
            with WanModelLoadingContext(model_name) as loading_progress: 
                loading_progress.update(10, "Initializing...")
                
                if model_info['type'] == 'VACE':
                    loading_progress.update(30, "Loading VACE model components...")
                    success = self._load_vace_model(model_info, wan_args)
                else: 
                    loading_progress.update(30, f"Loading {model_info['type']} model components...")
                    success = self._load_standard_wan_model(model_info, wan_args)
                
                if success:
                    loading_progress.update(80, "Configuring pipeline...")
                    if wan_args and hasattr(wan_args, 'wan_flash_attention_mode'):
                        self.flash_attention_mode = wan_args.wan_flash_attention_mode
                        print_wan_info(f"Flash Attention mode set to: {self.flash_attention_mode} from wan_args")
                    elif hasattr(self, 'pipeline') and self.pipeline and hasattr(self.pipeline, 'flash_attention_mode'):
                         pass 
                    else:
                         print_wan_info(f"Using default Flash Attention mode: {self.flash_attention_mode}")

                    loading_progress.update(100, "Pipeline ready!")
                    print_wan_success(f"✅ Model loaded successfully: {model_name}")
                    get_vram_stats(f"End of load_simple_wan_pipeline (Success) for {model_name}")
                    return True
                else:
                    print_wan_error(f"❌ Failed to load pipeline for {model_name}")
                    self.pipeline = None 
                    get_vram_stats(f"End of load_simple_wan_pipeline (Failure) for {model_name}")
                    return False
                    
        except Exception as e:
            print_wan_error(f"❌ Critical model loading failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None 
            get_vram_stats(f"End of load_simple_wan_pipeline (Exception) for {model_name}")
            return False

    def _ensure_wan_repo_in_path(self):
        """Ensures the Wan2.1 repository's 'wan' module is discoverable."""
        current_script_path = Path(__file__).resolve()
        deforum_ext_root = current_script_path.parent.parent.parent.parent 
        
        possible_wan_repo_paths = [
            deforum_ext_root / "Wan2.1",                            
            deforum_ext_root.parent / "Wan2.1",                     
            Path.cwd() / "Wan2.1",                                  
            Path("/tmp/stable-diffusion-webui-forge/extensions/Wan2.1") 
        ]
        
        wan_repo_path_found = None
        for potential_path in possible_wan_repo_paths:
            resolved_path = potential_path.resolve()
            wan_module_dir = resolved_path / "wan" 
            wan_module_init = wan_module_dir / "__init__.py" 
            print_wan_info(f"Checking for Wan repo at: {resolved_path} (Exists: {resolved_path.exists()}, 'wan' dir: {wan_module_dir.is_dir()}, 'wan/__init__.py': {wan_module_init.exists()})")
            
            if resolved_path.exists() and wan_module_dir.is_dir() and wan_module_init.exists():
                print_wan_success(f"Found valid Wan repository structure at: {resolved_path}")
                wan_repo_path_found = resolved_path
                break
        
        if wan_repo_path_found:
            wan_repo_path_str = str(wan_repo_path_found)
            if wan_repo_path_str not in sys.path:
                sys.path.insert(0, wan_repo_path_str) 
                print_wan_info(f"Added to sys.path: {wan_repo_path_str}")
            else:
                print_wan_info(f"Already in sys.path: {wan_repo_path_str}")
            return True
        else:
            print_wan_error("Could not find a valid Wan2.1 repository. Ensure it's correctly placed and contains the 'wan' module (e.g., extensions/Wan2.1/wan/__init__.py).")
            print_wan_info("Searched paths:")
            for p in possible_wan_repo_paths: print_wan_info(f"  - {p.resolve()}")
            return False

    def _load_vace_model(self, model_info: Dict, wan_args=None) -> bool:
        """Load VACE model with VRAM monitoring and proper handling"""
        model_path_str = model_info['path']
        print_wan_info(f"🔧 Attempting to load VACE model: {model_info['name']}")
        get_vram_stats(f"Start of _load_vace_model for {model_info['name']}")

        if not self._ensure_wan_repo_in_path(): 
            return False

        try:
            from wan.vace import WanVace 
            
            class VACEConfigPlaceholder:
                def __init__(self, model_path_str_config): 
                    self.num_train_timesteps = 1000 
                    self.param_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                    self.t5_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 
                    self.text_len = 512 
                    self.vae_stride = [4, 8, 8] 
                    self.patch_size = [1, 2, 2] 
                    self.sample_neg_prompt = "Low quality, blurry, distorted, artifacts, bad anatomy, worst quality, lowres"
                    self.sample_fps = wan_args.fps if wan_args and hasattr(wan_args, 'fps') else 8

                    self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                    self.vae_checkpoint = 'Wan2.1_VAE.pth'
                    self.t5_tokenizer = 'google/umt5-xxl' 

                    config_json_path = Path(model_path_str_config) / "config.json"
                    if config_json_path.exists():
                        print_wan_info(f"Loading VACE model configuration from: {config_json_path}")
                        try:
                            with open(config_json_path, 'r') as f:
                                model_config_data = json.load(f)
                            self.num_train_timesteps = model_config_data.get("num_train_timesteps", self.num_train_timesteps)
                            print_wan_success("Successfully loaded and applied VACE model config.json overrides.")
                        except Exception as e_cfg:
                            print_wan_warning(f"Could not load or apply VACE config.json: {e_cfg}. Using defaults.")
                    else:
                        print_wan_warning(f"VACE config.json not found at {config_json_path}. Using default VACEConfigPlaceholder values.")

            vace_config = VACEConfigPlaceholder(model_path_str)
            
            print_wan_info(f"WanVace will be initialized with checkpoint_dir: {model_path_str}")
            print_wan_info(f"Using T5 checkpoint: {vace_config.t5_checkpoint}, VAE checkpoint: {vace_config.vae_checkpoint}, Tokenizer: {vace_config.t5_tokenizer}")
            
            get_vram_stats(f"Before WanVace('{model_info['name']}') instantiation")
            
            if self.flash_attention_mode in ["auto", "enabled"]:
                try:
                    from .wan_flash_attention_patch import apply_flash_attention_patch, update_patched_flash_attention_mode
                    update_patched_flash_attention_mode(self.flash_attention_mode) 
                    patch_applied = apply_flash_attention_patch()
                    if patch_applied:
                        print_wan_success("Flash Attention monkey patch applied successfully for VACE.")
                    else:
                        print_wan_info("Flash Attention patch for VACE not applied (possibly already patched or not needed).")
                except ImportError:
                    print_wan_warning("wan_flash_attention_patch.py not found. Cannot apply Flash Attention patch.")
                except Exception as patch_e:
                    print_wan_error(f"Error applying Flash Attention patch for VACE: {patch_e}")

            vace_model_instance = WanVace(
                config=vace_config,
                checkpoint_dir=model_path_str, 
                device_id=torch.cuda.current_device() if self.device == 'cuda' else -1, 
                rank=0, 
                dit_fsdp=False, 
                t5_fsdp=False
            )
            get_vram_stats(f"After WanVace('{model_info['name']}') instantiation")

            class SmartVACEWrapper:
                def __init__(self, vace_model_instance_wrap, model_path_str_ref, device_ref): 
                    self.vace_model = vace_model_instance_wrap
                    self.device = device_ref 
                    
                    name_lower = Path(model_path_str_ref).name.lower()
                    if '14b' in name_lower: 
                        self.optimal_width = 1024 
                        self.optimal_height = 576
                        self.model_size_str = "14B"
                    elif '1.3b' in name_lower: 
                        self.optimal_width = 832 
                        self.optimal_height = 480
                        self.model_size_str = "1.3B"
                    else: 
                        self.optimal_width = 720
                        self.optimal_height = 480
                        self.model_size_str = "Unknown"
                    print_wan_info(f"VACE Wrapper: Model size '{self.model_size_str}', Optimal resolution set to {self.optimal_width}x{self.optimal_height}")

                def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, seed=-1, **kwargs):
                    aligned_width = width
                    aligned_height = height

                    if width != self.optimal_width or height != self.optimal_height:
                        print_wan_info(
                            f"VACE T2V: Using non-optimal resolution {width}x{height} (optimal {self.optimal_width}x{self.optimal_height})"
                        )

                    print_wan_info(f"VACE T2V generating with prompt: '{prompt[:70]}...' Res: {aligned_width}x{aligned_height}, Frames: {num_frames}, Steps: {num_inference_steps}")
                    get_vram_stats(f"Before VACE T2V generate call for '{prompt[:30]}...'")
                    try:
                        src_video_list_t2v = [None]
                        src_mask_list_t2v = [None]
                        src_ref_images_list_t2v = [None]

                        src_video, src_mask, src_ref_images = self.vace_model.prepare_source(
                            src_video_list_t2v,                
                            src_mask_list_t2v,                 
                            src_ref_images_list_t2v,           
                            num_frames=num_frames,             
                            image_size=(aligned_height, aligned_width), 
                            device=self.device                 
                        )
                        result = self.vace_model.generate(
                            input_prompt=prompt,
                            input_frames=src_video,
                            input_masks=src_mask,
                            input_ref_images=src_ref_images,
                            size=(aligned_height, aligned_width),
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps, 
                            guide_scale=guidance_scale,
                            shift=16, 
                            seed=seed if seed != -1 else np.random.randint(0, 2**31 - 1), 
                            **kwargs 
                        )
                        get_vram_stats(f"After VACE T2V generate call for '{prompt[:30]}...'")
                        return result
                    except Exception as e_gen:
                        print_wan_error(f"VACE T2V generation failed: {e_gen}")
                        import traceback
                        traceback.print_exc()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        get_vram_stats(f"After VACE T2V failure and cache clear for '{prompt[:30]}...'")
                        raise

                def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, seed=-1, **kwargs):
                    aligned_width = width
                    aligned_height = height
                    enhanced_prompt = f"Continuing from the provided image, {prompt}. Maintain visual style and continuity."

                    if width != self.optimal_width or height != self.optimal_height:
                         print_wan_info(
                             f"VACE I2V: Using non-optimal resolution {width}x{height} (optimal {self.optimal_width}x{self.optimal_height})"
                         )

                    print_wan_info(f"VACE I2V generating with prompt: '{enhanced_prompt[:70]}...' Res: {aligned_width}x{aligned_height}, Frames: {num_frames}, Steps: {num_inference_steps}")
                    get_vram_stats(f"Before VACE I2V generate call for '{prompt[:30]}...'")
                    
                    temp_image_path = None
                    try:
                        src_video_list_i2v = [None]
                        src_mask_list_i2v = [None]
                        src_ref_images_list_i2v = [None] 

                        if image is not None:
                            from PIL import Image as PILImage
                            if not isinstance(image, PILImage.Image): 
                                if hasattr(image, 'cpu') and hasattr(image, 'numpy'): 
                                    image_np = image.cpu().numpy()
                                else: 
                                    image_np = image
                                
                                if image_np.dtype in [np.float32, np.float64, torch.float32, torch.float16, torch.bfloat16]:
                                    if image_np.min() >= -1.001 and image_np.max() <= 1.001: 
                                        image_np = (image_np + 1.0) / 2.0
                                    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
                                else: 
                                    image_np = image_np.astype(np.uint8)
                                image = PILImage.fromarray(image_np)

                            if image.size != (aligned_width, aligned_height):
                                print_wan_info(f"Resizing input image for I2V from {image.size} to {aligned_width}x{aligned_height}")
                                image = image.resize((aligned_width, aligned_height), PILImage.Resampling.LANCZOS)
                            
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                image.save(tmp_file.name, 'PNG')
                                temp_image_path = tmp_file.name
                            
                            src_ref_images_list_i2v = [[temp_image_path]] 
                        else: 
                            print_wan_warning("VACE I2V: No image provided, falling back to T2V generation.")
                            return self.__call__(prompt, height, width, num_frames, num_inference_steps, guidance_scale, seed=seed, **kwargs)

                        src_video, src_mask, src_ref_images = self.vace_model.prepare_source(
                            src_video_list_i2v,                
                            src_mask_list_i2v,                 
                            src_ref_images_list_i2v,           
                            num_frames=num_frames,             
                            image_size=(aligned_height, aligned_width), 
                            device=self.device                 
                        )

                        result = self.vace_model.generate(
                            input_prompt=enhanced_prompt,
                            input_frames=src_video,
                            input_masks=src_mask,
                            input_ref_images=src_ref_images,
                            size=(aligned_height, aligned_width),
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps, 
                            guide_scale=guidance_scale,
                            shift=16, 
                            seed=seed if seed != -1 else np.random.randint(0, 2**31 - 1),
                            **kwargs
                        )
                        get_vram_stats(f"After VACE I2V generate call for '{prompt[:30]}...'")
                        return result
                    except Exception as e_gen_i2v:
                        print_wan_error(f"VACE I2V generation failed: {e_gen_i2v}")
                        import traceback
                        traceback.print_exc()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        get_vram_stats(f"After VACE I2V failure and cache clear for '{prompt[:30]}...'")
                        raise
                    finally:
                        if temp_image_path and os.path.exists(temp_image_path):
                            try: os.unlink(temp_image_path)
                            except Exception as e_del: print_wan_warning(f"Could not delete temp image {temp_image_path}: {e_del}")
            
            self.pipeline = SmartVACEWrapper(vace_model_instance, model_path_str, self.device)
            self.model_size = self.pipeline.model_size_str 
            self.optimal_width = self.pipeline.optimal_width
            self.optimal_height = self.pipeline.optimal_height
            print_wan_success(f"Smart VACE model '{model_info['name']}' loaded successfully.")
            get_vram_stats(f"End of _load_vace_model (Success) for {model_info['name']}")
            return True
            
        except ImportError as ie:
            print_wan_error(f"ImportError during VACE load: {ie}. Could not import 'wan.vace'. Ensure Wan2.1 repository is correctly placed and its 'wan' module is importable (check sys.path and __init__.py files).")
            return False
        except Exception as e_vace_load:
            print_wan_error(f"VACE model loading failed for {model_info['name']}: {e_vace_load}")
            import traceback
            traceback.print_exc()
            get_vram_stats(f"End of _load_vace_model (Exception) for {model_info['name']}")
            return False

    def _load_standard_wan_model(self, model_info: Dict, wan_args=None) -> bool:
        """Load standard (non-VACE) Wan T2V/I2V model with VRAM monitoring."""
        model_path_str = model_info['path']
        print_wan_info(f"🔧 Attempting to load standard Wan model: {model_info['name']} ({model_info['type']})")
        get_vram_stats(f"Start of _load_standard_wan_model for {model_info['name']}")

        if not self._ensure_wan_repo_in_path(): 
            print_wan_warning("Official Wan repo not found/configured. Standard model loading might rely on diffusers if compatible.")

        try:
            print_wan_info("Attempting to load standard Wan model using official 'wan' package...")
            WanModelClass = None
            if model_info['type'] == 'T2V':
                from wan.text2video import WanT2V 
                WanModelClass = WanT2V
                print_wan_info("Selected WanT2V for loading.")
            elif model_info['type'] == 'I2V':
                print_wan_warning("Official WanI2V class loading not explicitly implemented, attempting T2V-like load or diffusers.")
                raise NotImplementedError(f"Official WanI2V class loading for {model_info['name']} not yet fully supported in this script. Consider T2V or diffusers.")
            else:
                print_wan_error(f"Unknown standard Wan model type: {model_info['type']}")
                return False

            class StandardWanConfigPlaceholder:
                def __init__(self, model_p_str_cfg): 
                    self.model_path = model_p_str_cfg 
                    config_json_path = Path(model_p_str_cfg) / "config.json"
                    if config_json_path.exists():
                        try:
                            with open(config_json_path, 'r') as f:
                                cfg_data = json.load(f)
                            self.model = type('obj', (object,), cfg_data.get("model", {})) 
                            print_wan_success(f"Loaded config.json for standard model {Path(model_p_str_cfg).name}")
                        except Exception as e_cfg_std:
                            print_wan_warning(f"Could not load config.json for standard model: {e_cfg_std}. Using minimal defaults.")
                            self.model = type('obj', (object,), { 
                                'num_attention_heads': 32, 'attention_head_dim': 128,
                                'in_channels': 4, 'out_channels': 4, 'num_layers': 28,
                                'sample_size': 32, 'patch_size': 2,
                            }) 
                    else:
                         print_wan_warning(f"config.json not found for standard model at {config_json_path}. Using minimal defaults.")
                         self.model = type('obj', (object,), { 
                            'num_attention_heads': 32, 'attention_head_dim': 128,
                            'in_channels': 4, 'out_channels': 4, 'num_layers': 28,
                            'sample_size': 32, 'patch_size': 2,
                         })

            standard_config = StandardWanConfigPlaceholder(model_path_str)

            get_vram_stats(f"Before {WanModelClass.__name__}('{model_info['name']}') instantiation")
            if self.flash_attention_mode in ["auto", "enabled"]:
                try:
                    from .wan_flash_attention_patch import apply_flash_attention_patch, update_patched_flash_attention_mode
                    update_patched_flash_attention_mode(self.flash_attention_mode)
                    patch_applied_std = apply_flash_attention_patch() 
                    if patch_applied_std: print_wan_success("Flash Attention patch applied for standard Wan model.")
                    else: print_wan_info("Flash Attention patch for standard Wan model not applied.")
                except Exception as patch_e_std: print_wan_error(f"Error applying Flash Attention for std model: {patch_e_std}")

            official_wan_pipeline = WanModelClass(
                config=standard_config, 
                checkpoint_dir=model_path_str,
                device_id=torch.cuda.current_device() if self.device == 'cuda' else -1,
                rank=0,
                dit_fsdp=False, 
                t5_fsdp=False 
            )
            get_vram_stats(f"After {WanModelClass.__name__}('{model_info['name']}') instantiation")
            
            class StandardWanWrapper:
                def __init__(self, pipeline_instance_wrap, model_info_ref): 
                    self.pipeline = pipeline_instance_wrap
                    self.model_type = model_info_ref['type']
                    self.optimal_width = 720 
                    self.optimal_height = 480

                def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, seed=-1, **kwargs):
                    aligned_width = ((width + 15) // 16) * 16 
                    aligned_height = ((height + 15) // 16) * 16
                    if width != aligned_width or height != aligned_height:
                        print_wan_info(f"Standard {self.model_type}: Aligning resolution {width}x{height} to {aligned_width}x{aligned_height}")
                    
                    print_wan_info(f"Standard {self.model_type} generating: '{prompt[:70]}...' Res: {aligned_width}x{aligned_height}, Frames: {num_frames}")
                    get_vram_stats(f"Before standard {self.model_type} generate for '{prompt[:30]}...'")
                    try:
                        result = self.pipeline.generate(
                            input_prompt=prompt,
                            size=(aligned_width, aligned_height), 
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps, 
                            guide_scale=guidance_scale,
                            shift=kwargs.get('shift', 5.0), 
                            sample_solver=kwargs.get('sample_solver', 'unipc'), 
                            offload_model=kwargs.get('offload_model', True), 
                            seed=seed if seed != -1 else np.random.randint(0, 2**31 - 1),
                            **kwargs
                        )
                        get_vram_stats(f"After standard {self.model_type} generate for '{prompt[:30]}...'")
                        return result
                    except Exception as e_gen_std:
                        print_wan_error(f"Standard {self.model_type} generation failed: {e_gen_std}")
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        raise

                def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, seed=-1, **kwargs):
                    if self.model_type == 'I2V' and hasattr(self.pipeline, 'generate_i2v'): 
                        print_wan_info(f"Standard I2V generating (dedicated method)...")
                        raise NotImplementedError("Dedicated I2V method for standard Wan models not fully implemented here.")
                    else: 
                        print_wan_warning(f"Standard {self.model_type} used for I2V. True I2V continuity may vary. Enhancing prompt.")
                        enhanced_prompt = f"Continuing from a previous scene that includes an image similar to the input, {prompt}."
                        return self.__call__(enhanced_prompt, height, width, num_frames, num_inference_steps, guidance_scale, seed=seed, **kwargs)

            self.pipeline = StandardWanWrapper(official_wan_pipeline, model_info)
            self.model_size = model_info['size'] 
            self.optimal_width = self.pipeline.optimal_width 
            self.optimal_height = self.pipeline.optimal_height
            print_wan_success(f"Official Wan standard model '{model_info['name']}' loaded successfully.")
            get_vram_stats(f"End of _load_standard_wan_model (Official Success) for {model_info['name']}")
            return True

        except ImportError as ie_std:
            print_wan_warning(f"ImportError for official standard Wan model load: {ie_std}. Wan package might be missing or misconfigured for {model_info['type']}.")
        except NotImplementedError as nie_std:
            print_wan_warning(f"NotImplementedError for official standard Wan model: {nie_std}")
        except Exception as e_std_official:
            print_wan_warning(f"Official Wan standard model loading for {model_info['name']} failed: {e_std_official}. Trying diffusers fallback if applicable.")

        if model_info['type'] == 'T2V': 
            try:
                from ...diffusers import AutoPipelineForText2Video
                
                print_wan_info(f"Attempting to load {model_info['name']} using AutoPipelineForText2Video (diffusers)...")
                get_vram_stats(f"Before AutoPipelineForText2Video for {model_info['name']}")
                
                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                
                diffusers_pipeline_instance = AutoPipelineForText2Video.from_pretrained(
                    model_path_str,
                    torch_dtype=dtype,
                    use_safetensors=any((Path(model_path_str)/f).exists() for f in ["model.safetensors", "diffusion_pytorch_model.safetensors"]),
                )
                diffusers_pipeline_instance.to(self.device)

                get_vram_stats(f"After AutoPipelineForText2Video for {model_info['name']}")

                class DiffusersWrapper: 
                    def __init__(self, pipeline_instance_wrap, device_ref, model_info_ref): 
                        self.pipeline = pipeline_instance_wrap
                        self.device = device_ref
                        self.model_type = model_info_ref['type']
                        self.optimal_width = 720 
                        self.optimal_height = 480

                    def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, seed=-1, **kwargs):
                        print_wan_info(f"Diffusers {self.model_type} generating: '{prompt[:70]}...' Res: {width}x{height}, Frames: {num_frames}")
                        get_vram_stats(f"Before diffusers {self.model_type} generate for '{prompt[:30]}...'")
                        generator = torch.Generator(device=self.device).manual_seed(seed) if seed !=-1 else None
                        try:
                            output = self.pipeline(prompt=prompt,
                                                   num_frames=num_frames,
                                                   height=height,
                                                   width=width,
                                                   num_inference_steps=num_inference_steps, 
                                                   guidance_scale=guidance_scale,
                                                   generator=generator,
                                                   output_type="pt" 
                                                   )
                            video_frames = output.frames 
                            get_vram_stats(f"After diffusers {self.model_type} generate for '{prompt[:30]}...'")
                            return video_frames 
                        except Exception as e_gen_diff:
                            print_wan_error(f"Diffusers {self.model_type} generation failed: {e_gen_diff}")
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            raise
                    
                    def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, seed=-1, **kwargs):
                        print_wan_warning("Diffusers I2V called, but using T2V pipeline as fallback. True I2V continuity may not be present.")
                        enhanced_prompt = f"Inspired by an image, {prompt}"
                        return self.__call__(enhanced_prompt, height, width, num_frames, num_inference_steps, guidance_scale, seed=seed, **kwargs)

                self.pipeline = DiffusersWrapper(diffusers_pipeline_instance, self.device, model_info)
                self.model_size = model_info['size']
                self.optimal_width = self.pipeline.optimal_width
                self.optimal_height = self.pipeline.optimal_height
                print_wan_success(f"Diffusers pipeline for '{model_info['name']}' loaded successfully.")
                get_vram_stats(f"End of _load_standard_wan_model (Diffusers Success) for {model_info['name']}")
                return True

            except ImportError as ie_diff:
                print_wan_warning(f"Diffusers Import Error for {model_info['name']}: {ie_diff}. Diffusers might not be installed or pipeline type is incorrect.")
            except Exception as e_diffusers:
                print_wan_error(f"Diffusers fallback for {model_info['name']} failed: {e_diffusers}")

        print_wan_error(f"CRITICAL: Could not load standard Wan model '{model_info['name']}' with any available method.")
        get_vram_stats(f"End of _load_standard_wan_model (All Failed) for {model_info['name']}")
        return False

    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model based on discovery."""
        if not self.models: 
            print_wan_info("Models not discovered yet. Running discovery...")
            self.discover_models() 
        
        if not self.models:
            print_wan_warning("No Wan models found after discovery. Cannot select best model.")
            return None
        
        def model_priority(model):
            type_priority = {'VACE': 0, 'T2V': 1, 'I2V': 2, 'Unknown': 3}
            size_priority = {'1.3B': 0, '14B': 1, 'Unknown': 2} 
            return (type_priority.get(model['type'], 3), size_priority.get(model['size'], 2), model['name']) 
        
        try:
            sorted_models = sorted(self.models, key=model_priority)
            if not sorted_models:
                print_wan_warning("Model list is empty after sorting, cannot select best model.")
                return None
            best_model = sorted_models[0]
            print_wan_success(f"🎯 Best model selected: {best_model['name']} ({best_model['type']}, {best_model['size']})")
            return best_model
        except Exception as e_sort:
            print_wan_error(f"Error selecting best model: {e_sort}")
            if self.models: 
                print_wan_warning("Falling back to the first discovered model due to selection error.")
                return self.models[0]
            return None

    def unload_model(self):
        """Unload the model and clear CUDA cache to free VRAM."""
        print_wan_info("Attempting to unload model...")
        get_vram_stats("Start of unload_model")
        if self.pipeline:
            try:
                if hasattr(self.pipeline, 'vace_model') and hasattr(self.pipeline.vace_model, 'to'): 
                    print_wan_info("Moving VACE model components to CPU...")
                    if hasattr(self.pipeline.vace_model, 'text_encoder'): self.pipeline.vace_model.text_encoder.to('cpu')
                    if hasattr(self.pipeline.vace_model, 'model'): self.pipeline.vace_model.model.to('cpu') 
                    if hasattr(self.pipeline.vace_model, 'vae'): self.pipeline.vace_model.vae.to('cpu')
                    self.pipeline.vace_model.to('cpu') 
                    print_wan_info("VACE model components moved to CPU (attempted).")

                elif hasattr(self.pipeline, 'pipeline') and hasattr(self.pipeline.pipeline, 'to'): 
                    print_wan_info("Moving standard/diffusers pipeline to CPU...")
                    self.pipeline.pipeline.to('cpu')
                    print_wan_info("Standard/diffusers pipeline moved to CPU.")
                
                del self.pipeline 
                self.pipeline = None
                print_wan_success("Pipeline object deleted.")
                
            except Exception as e_unload:
                print_wan_error(f"Error during model offloading/deletion: {e_unload}")
                self.pipeline = None 

        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
            print_wan_success("CUDA cache cleared.")
        
        get_vram_stats("End of unload_model")
        print_wan_success("Model unloaded and memory freed (attempted).")
    
    def generate_video_with_i2v_chaining(self, clips: List[Dict], model_info: Dict, output_dir: str, wan_args=None, **kwargs):
        """Generate video with I2V chaining using styled progress indicators"""
        if not self.pipeline: 
            print_wan_error("Pipeline not loaded. Cannot generate video.")
            print_wan_info(f"Attempting to reload model: {model_info['name']}")
            if not self.load_simple_wan_pipeline(model_info, wan_args): 
                 raise RuntimeError("Pipeline not loaded and failed to reload.")
            print_wan_success("Model reloaded successfully.")

        get_vram_stats(f"Start of generate_video_with_i2v_chaining for {model_info['name']}")
        
        os.makedirs(output_dir, exist_ok=True) 
        timestring = kwargs.get('timestring', str(int(time.time()))) 
        
        fps_for_srt = getattr(wan_args, 'fps', kwargs.get('fps', 10.0)) 
        self.save_wan_settings_and_metadata(output_dir, timestring, clips, model_info, wan_args, **kwargs) 
        self.create_wan_srt_file(output_dir, timestring, clips, fps=fps_for_srt) 

        all_frame_paths = [] 
        last_frame_image_pil = None 
        total_frames_processed_count = 0 

        height = kwargs.get('height', getattr(wan_args, 'height', self.optimal_height))
        width = kwargs.get('width', getattr(wan_args, 'width', self.optimal_width))
        num_inference_steps = int(kwargs.get('num_inference_steps', getattr(wan_args, 'num_inference_steps', 20)))
        guidance_scale = float(kwargs.get('guidance_scale', getattr(wan_args, 'wan_guidance_scale', 7.5))) 
        seed = int(kwargs.get('seed', getattr(wan_args, 'seed', -1)))

        print_wan_info(f"Starting video generation: Clips: {len(clips)}, Res: {width}x{height}, Steps: {num_inference_steps}, Scale: {guidance_scale}")

        cleaned_kwargs = kwargs.copy()
        keys_to_remove_from_kwargs = [
            'prompt', 'height', 'width', 'num_frames',
            'num_inference_steps',    
            'guidance_scale',         
            'seed',                   
            'image',                  
            'anim_args',              
            'sampling_steps',         
            'steps',                  
            'size',                   
            'frame_num',              
            'input_prompt',           
            'shift',                  
        ]
        for key_to_remove in keys_to_remove_from_kwargs:
            cleaned_kwargs.pop(key_to_remove, None)


        with WanGenerationContext(len(clips)) as gen_context: 
            for clip_idx, clip_data in enumerate(clips):
                prompt = clip_data['prompt']
                num_frames = clip_data['num_frames']
                gen_context.update_clip(clip_idx, prompt[:30]) 

                print_wan_progress(f"Processing Clip {clip_idx + 1}/{len(clips)}: '{prompt[:50]}...' ({num_frames} frames)")
                get_vram_stats(f"Clip {clip_idx+1} - Before generation")

                video_output = None
                with create_wan_frame_progress(num_frames, clip_idx) as frame_progress_bar:
                    if clip_idx == 0 or last_frame_image_pil is None: 
                        print_wan_info("Generating T2V for current clip.")
                        video_output = self.pipeline(prompt=prompt, height=height, width=width, num_frames=num_frames,
                                                     num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, 
                                                     seed=seed, **cleaned_kwargs) 
                    else: 
                        print_wan_info("Generating I2V from previous clip's last frame.")
                        if not hasattr(self.pipeline, 'generate_image2video'): 
                            print_wan_error("Current pipeline does not support 'generate_image2video'. Falling back to T2V.")
                            video_output = self.pipeline(prompt=prompt, height=height, width=width, num_frames=num_frames,
                                                         num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, 
                                                         seed=seed, **cleaned_kwargs) 
                        else:
                            video_output = self.pipeline.generate_image2video(
                                image=last_frame_image_pil, prompt=prompt, height=height, width=width, num_frames=num_frames,
                                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, 
                                seed=seed, **cleaned_kwargs 
                            )
                
                get_vram_stats(f"Clip {clip_idx+1} - After generation")

                current_clip_frame_paths = self._process_and_save_frames(
                    video_output, clip_idx, output_dir, timestring,
                    total_frames_processed_count, 
                    frame_progress_bar 
                )
                
                if current_clip_frame_paths:
                    all_frame_paths.extend(current_clip_frame_paths)
                    try:
                        from PIL import Image as PILImage
                        last_frame_image_pil = PILImage.open(current_clip_frame_paths[-1])
                    except Exception as e_pil:
                        print_wan_warning(f"Could not load last frame for I2V chaining: {e_pil}")
                        last_frame_image_pil = None 
                    total_frames_processed_count += len(current_clip_frame_paths) 
                    print_wan_success(f"Clip {clip_idx+1} processed, {len(current_clip_frame_paths)} frames saved.")
                else:
                    print_wan_error(f"No frames processed or saved for clip {clip_idx+1}. I2V chaining might be affected.")
                    last_frame_image_pil = None 

        print_wan_success(f"Video generation complete. Total frames saved: {len(all_frame_paths)}")
        return {
            'output_dir': str(output_dir),
            'frame_paths': all_frame_paths,
            'timestring': timestring,
            'total_frames_generated': len(all_frame_paths)
        }

if __name__ == "__main__":
    print_wan_info("WanSimpleIntegration script running in test mode.")
    
    class DummyWanArgs: 
        def __init__(self):
            self.wan_flash_attention_mode = "auto"
            self.fps = 10
            self.num_inference_steps = 20 
            self.wan_guidance_scale = 7.0 
            self.seed = 12345
            self.height = 256 
            self.width = 256  
    dummy_args = DummyWanArgs()
    
    integration = WanSimpleIntegration(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    if not integration.test_wan_setup(): 
        print_wan_error("TEST: Wan setup test failed. Exiting.")
        sys.exit(1)

    selected_model_info = integration.get_best_model() 
    if not selected_model_info:
        print_wan_error("TEST: Could not select a best model after setup test. Exiting test.")
        sys.exit(1)
        
    print_wan_info(f"TEST: Selected model for loading: {selected_model_info['name']}")
    
    load_success = integration.load_simple_wan_pipeline(selected_model_info, wan_args=dummy_args) 
    if not load_success:
        print_wan_error(f"TEST: Failed to load model {selected_model_info['name']}. Exiting test.")
        sys.exit(1)
    
    print_wan_success(f"TEST: Model {selected_model_info['name']} loaded successfully.")
    
    if integration.pipeline: 
        print_wan_info("TEST: Attempting a short test generation...")
        test_clips = [ 
            {"prompt": "A beautiful landscape painting, trending on artstation", "num_frames": 8}, 
            {"prompt": "A futuristic city with flying cars, cinematic lighting", "num_frames": 8}  
        ]
        test_output_dir = Path("./wan_test_output") 
        test_output_dir.mkdir(exist_ok=True)
        
        try:
            gen_kwargs = { 
                'height': dummy_args.height, 
                'width': dummy_args.width,
                'num_inference_steps': 10, 
                'guidance_scale': 5.0,     
                'seed': 42,
                'timestring': f"test_{int(time.time())}",
                'fps': dummy_args.fps,
                'anim_args': None 
            }

            print_wan_info(f"TEST: Using generation parameters: H={gen_kwargs['height']}, W={gen_kwargs['width']}, Steps={gen_kwargs['num_inference_steps']}")

            results = integration.generate_video_with_i2v_chaining(
                clips=test_clips,
                model_info=selected_model_info,
                output_dir=str(test_output_dir),
                wan_args=dummy_args, 
                **gen_kwargs 
            )
            print_wan_success(f"TEST: Generation test completed. Output (if any) in: {results.get('output_dir')}")
            if results.get('frame_paths'):
                print_wan_info(f"TEST: Generated {len(results['frame_paths'])} frames.")
        except Exception as e_gen_test:
            print_wan_error(f"TEST: Generation test failed: {e_gen_test}")
            import traceback
            traceback.print_exc()

    integration.unload_model() 
    print_wan_success("TEST: Model unloaded (attempted).")
    
    print_wan_info("WanSimpleIntegration test script finished.")
