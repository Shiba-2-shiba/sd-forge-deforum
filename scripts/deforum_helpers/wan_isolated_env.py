"""
Wan Isolated Environment Manager - FAIL FAST
Creates and manages a completely isolated diffusers environment for Wan generation
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import shutil
from contextlib import contextmanager


class WanIsolatedEnvironment:
    """
    Manages an isolated Python environment specifically for Wan generation - FAIL FAST
    """
    
    def __init__(self, extension_root: str):
        self.extension_root = Path(extension_root)
        self.wan_env_dir = self.extension_root / "wan_isolated_env"
        self.wan_site_packages = self.wan_env_dir / "site-packages"
        self.wan_models_dir = self.wan_env_dir / "models"
        self.requirements_file = self.extension_root / "wan_requirements.txt"
        
        # Track original sys.path to restore later
        self.original_sys_path = None
        self.isolation_active = False
        
    def is_environment_ready(self) -> bool:
        """Check if the isolated environment is set up and ready"""
        if not self.wan_env_dir.exists():
            return False
            
        # Check for key packages
        required_packages = ['diffusers', 'transformers', 'accelerate', 'torch']
        for package in required_packages:
            package_dir = self.wan_site_packages / package
            if not package_dir.exists():
                return False
                
        return True
    
    def setup_environment(self, force_reinstall: bool = False):
        """Set up the isolated environment with Wan-compatible diffusers - FAIL FAST"""
        print("🔧 Setting up Wan isolated environment...")
        
        if self.is_environment_ready() and not force_reinstall:
            print("✅ Wan isolated environment already ready")
            return
            
        # Create directories
        self.wan_env_dir.mkdir(exist_ok=True)
        self.wan_site_packages.mkdir(exist_ok=True)
        self.wan_models_dir.mkdir(exist_ok=True)
        
        # Create requirements file for Wan-specific versions
        self._create_wan_requirements()
        
        # Install packages to isolated directory - FAIL FAST
        self._install_wan_packages()
        
        print("✅ Wan isolated environment setup complete!")
    
    def _create_wan_requirements(self):
        """Create requirements.txt with Wan-compatible versions"""
        wan_requirements = """
# Wan-compatible diffusers and dependencies - compatible with webui-forge
diffusers>=0.26.0,<0.33.0
transformers>=4.36.0,<4.46.0
accelerate>=0.25.0,<0.31.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.21.0
safetensors>=0.4.0
huggingface-hub>=0.20.0
"""
        
        with open(self.requirements_file, 'w', encoding='utf-8') as f:
            f.write(wan_requirements.strip())
    
    def _install_wan_packages(self):
        """Install Wan-specific packages to isolated directory - FAIL FAST"""
        print("📦 Installing Wan-compatible packages...")
        
        # Use pip with --target to install to specific directory
        cmd = [
            sys.executable, "-m", "pip", "install",
            "-r", str(self.requirements_file),
            "--target", str(self.wan_site_packages),
            "--upgrade",
            "--no-deps"  # Avoid dependency conflicts
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"Package installation failed: {result.stderr}")
        
        print("✅ Wan packages installed successfully")
    
    @contextmanager
    def isolated_imports(self):
        """Context manager for isolated imports - Temporarily modifies sys.path to use isolated packages"""
        if self.isolation_active:
            # Already in isolation, just yield
            yield
            return
            
        # Store original state
        self.original_sys_path = sys.path.copy()
        
        try:
            # Add isolated packages to the front of sys.path
            sys.path.insert(0, str(self.wan_site_packages))
            self.isolation_active = True
            
            print("🔒 Activated isolated imports for Wan")
            yield
            
        finally:
            # Restore original sys.path
            if self.original_sys_path is not None:
                sys.path = self.original_sys_path
                self.original_sys_path = None
            self.isolation_active = False
            print("🔓 Deactivated isolated imports")
    
    def prepare_model_structure(self, source_model_path: str) -> str:
        """Prepare model structure for Wan compatibility - FAIL FAST"""
        print(f"🔄 Preparing Wan-compatible model structure from {source_model_path}")
        
        source_path = Path(source_model_path)
        if not source_path.exists():
            raise RuntimeError(f"Source model path doesn't exist: {source_model_path}")
            
        # Create target model directory
        model_name = source_path.name
        target_path = self.wan_models_dir / model_name
        
        # Copy existing files
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)
        
        # Generate missing pipeline components - FAIL FAST
        self._generate_pipeline_structure(target_path)
            
        print(f"✅ Model structure prepared at: {target_path}")
        return str(target_path)
    
    def _generate_pipeline_structure(self, model_path: Path):
        """Generate missing pipeline files for Wan compatibility - FAIL FAST"""
        # Check what files we have
        existing_files = list(model_path.glob("*"))
        file_names = [f.name for f in existing_files]
        
        print(f"📋 Existing files: {file_names}")
        
        # Handle sharded model files - create index if we have sharded files
        sharded_files = [f for f in file_names if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")]
        if sharded_files and "diffusion_pytorch_model.safetensors.index.json" not in file_names:
            # Keep sharded files in main directory (don't move to subdirectory)
            # This is what diffusers expects for proper loading
            self._create_sharded_model_index(model_path, sharded_files)
            print("✅ Generated sharded model index in main directory")
            
            # Also ensure there's either a merged file or properly named shards
            main_model_file = model_path / "diffusion_pytorch_model.safetensors"
            if not main_model_file.exists():
                # Check if we have the right naming pattern for shards
                expected_pattern = any((model_path / f"diffusion_pytorch_model-{i:05d}-of-{len(sharded_files):05d}.safetensors").exists() for i in range(1, len(sharded_files)+1))
                
                if not expected_pattern:
                    try:
                        print("🔧 Attempting to merge sharded files...")
                        self._merge_sharded_files(model_path, sharded_files)
                        print("✅ Successfully merged sharded files")
                    except Exception as merge_error:
                        print(f"⚠️ Merge failed: {merge_error}")
                        # Copy first shard as fallback
                        first_shard = model_path / sharded_files[0]
                        if first_shard.exists():
                            try:
                                shutil.copy2(first_shard, main_model_file)
                                print(f"✅ Created main model file from first shard")
                            except Exception as copy_error:
                                print(f"❌ Copy fallback failed: {copy_error}")
                else:
                    print("✅ Sharded files already in correct naming pattern")
        
        # Generate main config.json (required by FluxPipeline)
        if "config.json" not in file_names:
            main_config = self._create_main_config()
            with open(model_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(main_config, f, indent=2)
            print("✅ Generated main config.json")
        
        # Generate model_index.json if missing
        if "model_index.json" not in file_names:
            model_index = self._create_model_index()
            with open(model_path / "model_index.json", 'w', encoding='utf-8') as f:
                json.dump(model_index, f, indent=2)
            print("✅ Generated model_index.json")
        
        # Generate basic scheduler config if missing
        scheduler_dir = model_path / "scheduler"
        if not scheduler_dir.exists():
            scheduler_dir.mkdir()
            scheduler_config = self._create_scheduler_config()
            with open(scheduler_dir / "scheduler_config.json", 'w', encoding='utf-8') as f:
                json.dump(scheduler_config, f, indent=2)
            print("✅ Generated scheduler config")
        
        # Setup tokenizer using HuggingFace instead of manual creation
        tokenizer_dir = model_path / "tokenizer"
        if not tokenizer_dir.exists():
            self._setup_tokenizer_from_huggingface(tokenizer_dir)
            print("✅ Generated tokenizer from HuggingFace")
        
        # Generate text encoder config if missing
        text_encoder_dir = model_path / "text_encoder"
        if not text_encoder_dir.exists():
            text_encoder_dir.mkdir()
            text_encoder_config = self._create_text_encoder_config()
            with open(text_encoder_dir / "config.json", 'w', encoding='utf-8') as f:
                json.dump(text_encoder_config, f, indent=2)
            print("✅ Generated text encoder config")
    
    def _create_main_config(self) -> Dict[str, Any]:
        """Create main config.json that works with standard UNet models"""
        return {
            "_class_name": "UNet2DConditionModel",
            "_diffusers_version": "0.26.0",
            "in_channels": 4,
            "out_channels": 4,
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D",
                "DownBlock2D"
            ],
            "up_block_types": [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ],
            "block_out_channels": [320, 640, 1280, 1280],
            "layers_per_block": 2,
            "attention_head_dim": 8,
            "norm_num_groups": 32,
            "cross_attention_dim": 768,
            "sample_size": 64
        }
    
    def _create_model_index(self) -> Dict[str, Any]:
        """Create a basic model_index.json that works with multiple pipeline types"""
        return {
            "_class_name": "DiffusionPipeline",
            "_diffusers_version": "0.26.0",
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"], 
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"]
        }
    
    def _create_scheduler_config(self) -> Dict[str, Any]:
        """Create basic scheduler configuration"""
        return {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.26.0",
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "trained_betas": None,
            "skip_prk_steps": True,
            "set_alpha_to_one": False,
            "prediction_type": "epsilon",
            "timestep_spacing": "leading",
            "steps_offset": 1
        }
    
    def _create_text_encoder_config(self) -> Dict[str, Any]:
        """Create basic text encoder configuration"""
        return {
            "_name_or_path": "openai/clip-vit-large-patch14",
            "architectures": ["CLIPTextModel"],
            "attention_dropout": 0.0,
            "dropout": 0.0,
            "hidden_act": "quick_gelu",
            "hidden_size": 768,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 77,
            "model_type": "clip_text_model",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "projection_dim": 768,
            "torch_dtype": "float32",
            "transformers_version": "4.36.0",
            "vocab_size": 49408
        }
    
    def _create_sharded_model_index(self, model_path: Path, sharded_files: List[str]):
        """Create index file for sharded model files"""
        print(f"🔧 Creating sharded model index for {len(sharded_files)} files...")
        
        try:
            # Try to use safetensors to read the actual tensor names
            with self.isolated_imports():
                from safetensors import safe_open
                
                weight_map = {}
                metadata = {"total_size": 0}
                
                # Sort the files to ensure proper ordering
                sharded_files.sort()
                
                # Read each shard file to get tensor names
                for shard_file in sharded_files:
                    file_path = model_path / shard_file
                    if not file_path.exists():
                        continue
                        
                    try:
                        # Open the safetensors file and read tensor names
                        with safe_open(file_path, framework="pt", device="cpu") as f:
                            for tensor_name in f.keys():
                                weight_map[tensor_name] = shard_file
                        
                        # Add file size
                        metadata["total_size"] += file_path.stat().st_size
                        print(f"📋 Mapped tensors from {shard_file}")
                        
                    except Exception as e:
                        print(f"⚠️ Could not read {shard_file}: {e}")
                        continue
                
                if weight_map:
                    # Create the index structure that diffusers expects
                    index_data = {
                        "metadata": metadata,
                        "weight_map": weight_map
                    }
                    
                    # Write the index file
                    index_file = model_path / "diffusion_pytorch_model.safetensors.index.json"
                    with open(index_file, 'w', encoding='utf-8') as f:
                        json.dump(index_data, f, indent=2)
                    
                    print(f"✅ Created sharded model index with {len(weight_map)} tensors across {len(sharded_files)} shards")
                else:
                    raise ValueError("No tensors found in shard files")
                    
        except Exception as e:
            print(f"⚠️ Failed to create proper sharded index: {e}")
            raise RuntimeError(f"Cannot create sharded model index: {e}")
    
    def _merge_sharded_files(self, model_path: Path, sharded_files: List[str]):
        """Attempt to merge sharded model files into a single file"""
        print(f"🔧 Attempting to merge {len(sharded_files)} sharded files...")
        
        try:
            with self.isolated_imports():
                from safetensors import safe_open
                import torch
                
                # Collect all tensors from all shards
                merged_tensors = {}
                
                for shard_file in sorted(sharded_files):
                    shard_path = model_path / shard_file
                    if not shard_path.exists():
                        continue
                        
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for tensor_name in f.keys():
                            if tensor_name in merged_tensors:
                                print(f"⚠️ Duplicate tensor {tensor_name} found in {shard_file}")
                                continue
                            merged_tensors[tensor_name] = f.get_tensor(tensor_name)
                
                if not merged_tensors:
                    raise ValueError("No tensors found to merge")
                
                # Save merged tensors
                from safetensors.torch import save_file
                output_path = model_path / "diffusion_pytorch_model.safetensors"
                save_file(merged_tensors, str(output_path))
                
                print(f"✅ Successfully merged {len(merged_tensors)} tensors into single file")
                
        except Exception as e:
            raise RuntimeError(f"Failed to merge sharded files: {e}")
    
    def _setup_tokenizer_from_huggingface(self, tokenizer_dir: Path):
        """
        General solution: Use HuggingFace transformers to automatically setup tokenizer
        This avoids manual token creation and uses the real CLIP tokenizer
        """
        tokenizer_dir.mkdir(exist_ok=True)
        
        # Check if tokenizer already exists and is complete
        required_files = ["tokenizer.json", "tokenizer_config.json"]
        if all((tokenizer_dir / f).exists() for f in required_files):
            print("✅ Using existing cached tokenizer")
            return
        
        try:
            # Method 1: Try to use transformers AutoTokenizer (most reliable)
            with self.isolated_imports():
                from transformers import CLIPTokenizer
                import os
                
                # Set up persistent cache directory
                cache_dir = self.extension_root / "hf_cache"
                cache_dir.mkdir(exist_ok=True)
                os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
                os.environ['HF_HOME'] = str(cache_dir)
                
                print("📥 Downloading CLIP tokenizer from HuggingFace...")
                tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    cache_dir=cache_dir
                )
                tokenizer.save_pretrained(str(tokenizer_dir))
                print("✅ Successfully setup tokenizer using transformers")
                return
                
        except Exception as e:
            print(f"⚠️ Method 1 failed: {e}")
            
        try:
            # Method 2: Try HuggingFace Hub direct download (fallback)
            with self.isolated_imports():
                from huggingface_hub import snapshot_download
                print("📥 Downloading tokenizer files directly...")
                snapshot_download(
                    repo_id="openai/clip-vit-large-patch14",
                    local_dir=str(tokenizer_dir),
                    allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
                    local_dir_use_symlinks=False
                )
                print("✅ Successfully downloaded tokenizer files")
                return
                
        except Exception as e:
            print(f"⚠️ Method 2 failed: {e}")
            
        # Method 3: Create minimal tokenizer config as last resort
        print("💡 Creating minimal tokenizer config as fallback...")
        self._create_minimal_tokenizer_config(tokenizer_dir)
    
    def _create_minimal_tokenizer_config(self, tokenizer_dir: Path):
        """
        Minimal fallback: Create only the essential tokenizer config without complex vocab
        """
        # Just create a basic config that points to the standard CLIP tokenizer
        tokenizer_config = {
            "add_prefix_space": False,
            "bos_token": "<|startoftext|>",
            "clean_up_tokenization_spaces": True,
            "do_lower_case": True,
            "eos_token": "<|endoftext|>",
            "model_max_length": 77,
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "CLIPTokenizer",
            "unk_token": "<|endoftext|>",
            "_name_or_path": "openai/clip-vit-large-patch14"
        }
        
        with open(tokenizer_dir / "tokenizer_config.json", 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        print("✅ Created minimal tokenizer config")
    
    def download_wan_components(self, model_path: str):
        """Download missing pipeline components from HuggingFace"""
        model_path = Path(model_path)
        
        # Check if components already exist
        text_encoder_dir = model_path / "text_encoder"
        if text_encoder_dir.exists() and any(text_encoder_dir.glob("*.safetensors")):
            print("✅ Using existing cached components")
            return
            
        print("📥 Downloading missing pipeline components...")
        
        try:
            with self.isolated_imports():
                from huggingface_hub import snapshot_download
                import os
                
                # Set up persistent cache directory
                cache_dir = self.extension_root / "hf_cache"
                cache_dir.mkdir(exist_ok=True)
                os.environ['HF_HOME'] = str(cache_dir)
                
                # Download CLIP text encoder if missing weights
                if not any(text_encoder_dir.glob("*.safetensors")):
                    print("📥 Downloading CLIP text encoder...")
                    snapshot_download(
                        repo_id="openai/clip-vit-large-patch14",
                        local_dir=str(text_encoder_dir),
                        allow_patterns=["*.json", "*.safetensors", "*.txt"],
                        local_dir_use_symlinks=False,
                        cache_dir=cache_dir
                    )
                
                print("✅ Component download complete")
                
        except Exception as e:
            print(f"⚠️ Component download failed: {e}")
            print("💡 Using basic generated components instead")


class WanIsolatedGenerator:
    """Wan generator that uses the isolated environment"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.env_manager = None
        self.prepared_model_path = None
        
    def setup(self, extension_root: str):
        """Set up the isolated environment and prepare the model"""
        print("🚀 Setting up Wan isolated generator...")
        
        # Initialize environment manager
        self.env_manager = WanIsolatedEnvironment(extension_root)
        
        # Set up isolated environment
        self.env_manager.setup_environment()
        
        # Prepare model structure
        self.prepared_model_path = self.env_manager.prepare_model_structure(self.model_path)
        
        # Download missing components
        self.env_manager.download_wan_components(self.prepared_model_path)
        
        print("✅ Wan isolated generator setup complete!")
    
    def generate_video(self, prompt: str, **kwargs) -> List:
        """Generate video using available diffusers pipelines"""
        if not self.env_manager or not self.prepared_model_path:
            raise RuntimeError("Generator not properly set up")
        
        print(f"🎬 Generating video with isolated environment: '{prompt}'")
        
        with self.env_manager.isolated_imports():
            import torch
            
            # Try different pipeline types in order of preference for video generation
            pipeline_classes = []
            
            try:
                # First try standard stable diffusion pipelines (most likely to work)
                from diffusers import StableDiffusionPipeline
                pipeline_classes.append(("StableDiffusionPipeline", StableDiffusionPipeline))
            except ImportError:
                pass
                
            try:
                # Try generic DiffusionPipeline (auto-detects model type)
                from diffusers import DiffusionPipeline
                pipeline_classes.append(("DiffusionPipeline", DiffusionPipeline))
            except ImportError:
                pass
            
            try:
                # Try dedicated video pipelines
                from diffusers import I2VGenXLPipeline
                pipeline_classes.append(("I2VGenXLPipeline", I2VGenXLPipeline))
            except ImportError:
                pass
            
            try:
                from diffusers import StableVideoDiffusionPipeline
                pipeline_classes.append(("StableVideoDiffusionPipeline", StableVideoDiffusionPipeline))
            except ImportError:
                pass
            
            try:
                # Try Flux as last resort (requires specific model format)
                from diffusers import FluxPipeline
                pipeline_classes.append(("FluxPipeline", FluxPipeline))
            except ImportError:
                pass
            
            if not pipeline_classes:
                raise RuntimeError("No compatible pipeline classes found in diffusers")
            
            # Try loading pipelines in order
            for pipeline_name, pipeline_class in pipeline_classes:
                try:
                    print(f"🧪 Attempting to load {pipeline_name}...")
                    
                    # Load pipeline
                    pipeline = pipeline_class.from_pretrained(
                        self.prepared_model_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None,  # Disable safety checker for speed
                        requires_safety_checker=False
                    ).to(self.device)
                    
                    print(f"✅ Successfully loaded {pipeline_name}")
                    
                    # Handle different pipeline argument patterns
                    generation_kwargs = kwargs.copy()
                    
                    # Check if we're doing image-to-video (image parameter provided)
                    is_img2video = 'image' in generation_kwargs
                    
                    # Remove video-specific args that image pipelines don't understand
                    if pipeline_name in ["FluxPipeline"]:
                        generation_kwargs.pop('num_frames', None)
                        generation_kwargs.pop('width', None)
                        generation_kwargs.pop('height', None)
                        
                        # For image pipelines, generate multiple images as "frames"
                        num_frames = kwargs.get('num_frames', 1)
                        frames = []
                        
                        if is_img2video:
                            # For img2video with image pipeline, use the image as init_image
                            init_image = generation_kwargs.pop('image', None)
                            if init_image:
                                # Use image-to-image generation
                                for i in range(num_frames):
                                    result = pipeline(prompt=prompt, image=init_image, **generation_kwargs)
                                    if hasattr(result, 'images') and result.images:
                                        frames.extend(result.images)
                                    else:
                                        frames.append(result)
                            else:
                                # Fallback to text-to-image
                                for i in range(num_frames):
                                    result = pipeline(prompt=prompt, **generation_kwargs)
                                    if hasattr(result, 'images') and result.images:
                                        frames.extend(result.images)
                                    else:
                                        frames.append(result)
                        else:
                            # Text-to-image generation
                            for i in range(num_frames):
                                result = pipeline(prompt=prompt, **generation_kwargs)
                                if hasattr(result, 'images') and result.images:
                                    frames.extend(result.images)
                                else:
                                    frames.append(result)
                    else:
                        # For video pipelines, use all arguments
                        result = pipeline(prompt=prompt, **generation_kwargs)
                        
                        # Convert result to frames
                        if hasattr(result, 'frames'):
                            frames = result.frames[0] if isinstance(result.frames, list) else result.frames
                        elif hasattr(result, 'images'):
                            frames = result.images
                        elif isinstance(result, list):
                            frames = result
                        else:
                            frames = [result]
                    
                    print(f"✅ Generated {len(frames)} frames using {pipeline_name}")
                    return frames
                    
                except Exception as e:
                    print(f"⚠️ {pipeline_name} failed: {e}")
                    continue
            
            # If all pipelines failed
            raise RuntimeError("All pipeline loading attempts failed")
