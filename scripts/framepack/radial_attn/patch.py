# scripts/framepack/radial_attn/patch.py (デバッグ機能強化版)

import torch
from torch.nn import functional as F
from einops import rearrange
from .attn_mask import RadialAttention, MaskMap
from ...diffusers.models.attention_processor import Attention

def print_tensor_stats(tensor, name):
    """テンソルの統計情報を表示するヘルパー関数"""
    if tensor is None:
        print(f"[RA-STATS] {name}: is None")
        return
    # .float()に変換してから統計情報を計算することで、bfloat16等でのエラーを防ぐ
    tensor_float = tensor.detach().float()
    print(
        f"[RA-STATS] {name}: "
        f"shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, "
        f"min={tensor_float.min():.4f}, max={tensor_float.max():.4f}, "
        f"mean={tensor_float.mean():.4f}, std={tensor_float.std():.4f}"
    )

def apply_rotary_emb_transposed(x, freqs_cis):
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = x.float() * cos + x_rotated.float() * sin
    out = out.to(x)
    return out

def radial_attention_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kwargs):
    ra_params = getattr(self, "_ra_params", {})
    layer_idx = ra_params.get("layer_idx", 999)
    transformer_ref = getattr(self, "_transformer_ref", None)
    timestep_tensor = getattr(transformer_ref, 'current_timestep_for_ra', None)
    
    is_dense = True
    current_timestep_val = -1
    if timestep_tensor is not None:
        current_timestep_val = int(timestep_tensor.flatten()[0].item())
        SPARSE_TIMESTEP_THRESHOLD = 500
        is_dense = (
            current_timestep_val > SPARSE_TIMESTEP_THRESHOLD
            or layer_idx < ra_params.get("dense_layers", 0)
        )

    # ネガティブプロンプト計算かどうかを判定 (CFGのトリック)
    # ポジティブ計算時のpooled_projectionは値を持つが、ネガティブでは持たないことが多い、という経験則に基づく
    is_negative_pass = kwargs.get('pooled_projections') is None or torch.all(kwargs['pooled_projections'] == 0)

    # ★★ 新しい解決策 ★★
    # スパースモードであっても、ネガティブプロンプトの計算時は常にDENSE（オリジナル）の計算を行う
    if not is_dense and is_negative_pass:
        if layer_idx == 0:
            print(f"[RA-PATCH] Timestep: {current_timestep_val}, Layer: {layer_idx} -> Mode: FORCING DENSE for negative prompt")
        is_dense = True # 強制的にデンスモードへ

    if is_dense:
        return self.original_forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs
        )

    # --- 以下、スパースアテンションの処理 ---
    if layer_idx == 0:
        print(f"[RA-PATCH] Timestep: {current_timestep_val}, Layer: {layer_idx} -> Mode: SPARSE (Radial) for positive prompt")
        print("="*40)
        print_tensor_stats(hidden_states, "Sparse Input HS")
        print_tensor_stats(encoder_hidden_states, "Sparse Input EHS")
        print_tensor_stats(image_rotary_emb, "Sparse Input RoPE")
        print("="*40)
        
    if not hasattr(self, '_corrected_mask_map'):
        actual_video_tokens = hidden_states.shape[1]
        num_frames = ra_params.get("num_frames")
        packed_num_frames = num_frames * 2
        self._corrected_mask_map = MaskMap(video_token_num=actual_video_tokens, num_frame=packed_num_frames)

    mask_map = self._corrected_mask_map
    
    context = encoder_hidden_states if encoder_hidden_states is not None else torch.tensor([], device=hidden_states.device, dtype=hidden_states.dtype)
    video_len = hidden_states.shape[1]
    
    if context.numel() > 0:
        x = torch.cat([hidden_states, context], dim=1)
    else:
        x = hidden_states

    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)
    
    if image_rotary_emb is not None:
        q_video, q_text = q[:, :video_len], q[:, video_len:]
        k_video, k_text = k[:, :video_len], k[:, video_len:]
        
        q_video = q_video.unflatten(2, (self.heads, -1))
        k_video = k_video.unflatten(2, (self.heads, -1))
        
        q_video = apply_rotary_emb_transposed(q_video, image_rotary_emb)
        k_video = apply_rotary_emb_transposed(k_video, image_rotary_emb)

        q_video = q_video.flatten(2)
        k_video = k_video.flatten(2)
        
        q = torch.cat([q_video, q_text], dim=1)
        k = torch.cat([k_video, k_text], dim=1)
    
    q = q.unflatten(2, (self.heads, -1)).transpose(1, 2)
    k = k.unflatten(2, (self.heads, -1)).transpose(1, 2)
    v = v.unflatten(2, (self.heads, -1)).transpose(1, 2)
    
    if hasattr(self, 'norm_q') and self.norm_q is not None: q = self.norm_q(q)
    if hasattr(self, 'norm_k') and self.norm_k is not None: k = self.norm_k(k)
    
    batch_size, _, seq_len, _ = q.shape
    query_ra = rearrange(q, "b h s d -> (b s) h d")
    key_ra = rearrange(k, "b h s d -> (b s) h d")
    value_ra = rearrange(v, "b h s d -> (b s) h d")

    query_ra_f32 = query_ra.to(torch.float32)
    key_ra_f32 = key_ra.to(torch.float32)
    value_ra_f32 = value_ra.to(torch.float32)

    out_ra = RadialAttention(
        query=query_ra_f32, key=key_ra_f32, value=value_ra_f32,
        mask_map=mask_map,
        sparsity_type="radial",
        block_size=128,
        decay_factor=ra_params.get("decay_factor", 1.0),
        model_type="hunyuan"
    )

    if hasattr(self, 'scale'):
        out_ra = out_ra * self.scale
    
    out = rearrange(out_ra, "(b s) h d -> b h s d", b=batch_size, s=seq_len)
    out = out.to(hidden_states.dtype)
    out = out.transpose(1, 2).flatten(2, 3)

    context_len = context.shape[1]
    if context_len > 0:
        hidden_states_out, context_out = out[:, :-context_len], out[:, -context_len:]
    else:
        hidden_states_out, context_out = out, None

    if isinstance(self.to_out, (torch.nn.ModuleList, torch.nn.Sequential)):
        hidden_states_out = self.to_out[0](hidden_states_out)
        hidden_states_out = self.to_out[1](hidden_states_out)
        if hasattr(self, 'to_add_out') and context_out is not None:
            context_out = self.to_add_out(context_out)
    else:
        pass

    if layer_idx == 0:
        print_tensor_stats(hidden_states_out, "Sparse Output HS")
        print("="*40 + "\n")

    return hidden_states_out, context_out

def apply_radial_attention_patch(model, dense_layers, dense_timesteps, decay_factor, num_frames, width, height):
    print("Applying Radial Attention patch...")
    ra_params = {
        "dense_layers": dense_layers,
        "dense_timesteps": dense_timesteps,
        "decay_factor": decay_factor,
        "num_frames": num_frames,
        "width": width,
        "height": height,
    }
    for block_list_name in ['transformer_blocks', 'single_transformer_blocks']:
        if hasattr(model, block_list_name):
            block_list = getattr(model, block_list_name)
            for layer_idx, block in enumerate(block_list):
                if hasattr(block, 'attn'):
                    block.attn._ra_params = {**ra_params, "layer_idx": layer_idx}
                    block.attn._transformer_ref = model
                    if not hasattr(block.attn, 'original_forward'):
                        block.attn.original_forward = block.attn.forward
                    block.attn.forward = radial_attention_forward.__get__(block.attn, type(block.attn))
