# FramePack-eichi LoRA Utilities
# 
# LoRAの適用、FP8最適化、LoRAフォーマット検出と変換のための機能を提供します。

from .lora_utils import (
    merge_lora_to_state_dict,
    load_safetensors_with_lora_and_fp8,
    load_safetensors_with_fp8_optimization,
    convert_hunyuan_to_framepack,
    convert_from_diffusion_pipe_or_something
)

from .fp8_optimization_utils import (
    calculate_fp8_maxval,
    quantize_tensor_to_fp8,
    optimize_state_dict_with_fp8_on_the_fly,
    fp8_linear_forward_patch,
    apply_fp8_monkey_patch,
    check_fp8_support
)

from .lora_loader import (
    load_and_apply_lora
)

from .safetensors_utils import (
    MemoryEfficientSafeOpen
)

