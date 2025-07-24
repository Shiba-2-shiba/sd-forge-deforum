# scripts/framepack/radial_attn/patch.py

import torch
from torch.nn import functional as F
from einops import rearrange
from .attn_mask import RadialAttention, MaskMap
from ...diffusers.models.attention_processor import Attention

ORIGINAL_ATTENTION_FORWARD = Attention.forward

def radial_attention_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kwargs):
    """Radial Attentionを適用する新しいforwardメソッド"""
    # このメソッドがアタッチされたattnインスタンスから、関連パラメータを取得
    ra_params = getattr(self, "_ra_params", {})
    
    # transformerオブジェクトにセットされた現在のタイムステップ数を取得
    # self._transformer_refはapply_patch時にセットされる
    transformer_ref = getattr(self, "_transformer_ref", None)
    numeral_timestep = getattr(transformer_ref, "numeral_timestep", 0) if transformer_ref else 0

    layer_idx = ra_params.get("layer_idx", 999)
    is_dense = (
        numeral_timestep < ra_params.get("dense_timestep", 0)
        or layer_idx < ra_params.get("dense_block", 0)
    )

    if is_dense:
        # オリジナルのforwardを呼び出す
        return ORIGINAL_ATTENTION_FORWARD(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

    # --- Radial Attentionの処理 ---
    mask_map = ra_params.get("mask_map")
    if mask_map is None:
        return ORIGINAL_ATTENTION_FORWARD(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

    # この部分はHunyuanVideoSingleTransformerBlockの実装を参考に адаптация
    # hunyuan_video_packed.py の HunyuanAttnProcessorFlashAttnSingle に相当するロジック
    # 注意：この実装はSingle Stream Blockを想定しています。Dual Streamの場合はロジックが異なります。
    
    context = encoder_hidden_states
    x = torch.cat([hidden_states, context], dim=1)

    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)
    
    q = self.norm_q(q)
    k = self.norm_k(k)

    q = q.unflatten(2, (self.heads, -1)).transpose(1, 2)
    k = k.unflatten(2, (self.heads, -1)).transpose(1, 2)
    v = v.unflatten(2, (self.heads, -1)).transpose(1, 2)
    
    # 以下、Radial Attentionの呼び出し
    batch_size, _, seq_len, _ = q.shape
    query_ra = rearrange(q, "b h s d -> (b s) h d")
    key_ra = rearrange(k, "b h s d -> (b s) h d")
    value_ra = rearrange(v, "b h s d -> (b s) h d")

    out_ra = RadialAttention(
        query=query_ra, key=key_ra, value=value_ra,
        mask_map=mask_map,
        sparsity_type="radial",
        block_size=128,
        decay_factor=ra_params.get("decay_factor", 1.0),
        model_type="hunyuan"
    )
    
    out = rearrange(out_ra, "(b s) h d -> b h s d", b=batch_size, s=seq_len)
    out = out.transpose(1, 2).flatten(2, 3)

    # 出力形式を整える
    hidden_states_out, context_out = out[:, :-context.shape[1]], out[:, -context.shape[1]:]

    hidden_states_out = self.to_out[0](hidden_states_out)
    hidden_states_out = self.to_out[1](hidden_states_out)
    context_out = self.to_add_out(context_out) if hasattr(self.to_out, 'to_add_out') else context_out # to_add_outの有無を確認

    return hidden_states_out, context_out

def apply_radial_attention_patch(model, dense_layers, dense_timesteps, decay_factor, num_frames, width, height):
    """モデルのAttentionモジュールにパッチを適用"""
    print("Applying Radial Attention patch with default settings...")
    
    # Framepackを考慮したシーケンス長を計算 (文脈フレーム + 生成フレーム)
    packed_num_frames = num_frames * 2 
    frame_size = (height // 8) * (width // 8)
    video_token_num = frame_size * packed_num_frames

    mask_map = MaskMap(video_token_num=video_token_num, num_frame=packed_num_frames)
    
    ra_params = {
        "dense_block": dense_layers,
        "dense_timestep": dense_timesteps,
        "decay_factor": decay_factor,
        "mask_map": mask_map,
    }

    # 各Attentionブロックにパッチを適用
    for block_list_name in ['transformer_blocks', 'single_transformer_blocks']:
        if hasattr(model.transformer, block_list_name):
            block_list = getattr(model.transformer, block_list_name)
            for layer_idx, block in enumerate(block_list):
                if hasattr(block, 'attn'):
                    block.attn._ra_params = {**ra_params, "layer_idx": layer_idx}
                    block.attn._transformer_ref = model.transformer # transformerへの参照を保存
                    block.attn.forward = radial_attention_forward.__get__(block.attn, Attention)
