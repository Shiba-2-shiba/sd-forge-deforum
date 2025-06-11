import torch
import math

from .uni_pc_fm import sample_unipc
from .wrapper import fm_wrapper
from .utils import repeat_to_batch_size


def flux_time_shift(t, mu=1.15, sigma=1.0):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def calculate_flux_mu(context_length, x1=256, y1=0.5, x2=4096, y2=1.15, exp_max=7.0):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    mu = k * context_length + b
    mu = min(mu, math.log(exp_max))
    return mu


def get_flux_sigmas_from_mu(n, mu):
    sigmas = torch.linspace(1, 0, steps=n + 1)
    sigmas = flux_time_shift(sigmas, mu=mu)
    return sigmas


@torch.inference_mode()
def sample_hunyuan(
        transformer,
        sampler='unipc',
        initial_latent=None,
        concat_latent=None,
        strength=1.0,
        initial_keyframe_strength=0.5,
        width=512,
        height=512,
        frames=16,
        real_guidance_scale=1.0,
        distilled_guidance_scale=10.0,
        guidance_rescale=0.0,
        shift=None,
        num_inference_steps=25,
        batch_size=None,
        generator=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        prompt_poolers=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_poolers=None,
        dtype=torch.bfloat16,
        device=None,
        negative_kwargs=None,
        callback=None,
        **kwargs,
):
    device = device or transformer.device

    if batch_size is None:
        batch_size = int(prompt_embeds.shape[0])

    latents = torch.randn((batch_size, 16, (frames + 3) // 4, height // 8, width // 8), generator=generator, device=generator.device).to(device=device, dtype=torch.float32)

    B, C, T, H, W = latents.shape
    seq_length = T * H * W // 4

    if shift is None:
        mu = calculate_flux_mu(seq_length, exp_max=7.0)
    else:
        mu = math.log(shift)

    main_sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)
    main_sigmas = main_sigmas * strength
    # サンプラーには、この一貫したスケジュールを渡す
    sigmas = main_sigmas
    # 初期latentがある場合のノイズ付加処理
    if initial_latent is not None:
        # 最初のキーフレーム専用のノイズスケジュールを別途計算
        initial_sigmas = get_flux_sigmas_from_mu(num_inference_steps, mu).to(device)
        initial_sigmas = initial_sigmas * initial_keyframe_strength

        # ノイズ付加に使うシグマ値（ノイズの強さ）を取得
        first_sigma_for_initial = initial_sigmas[0].to(device=device, dtype=torch.float32)
        first_sigma_for_others = main_sigmas[0].to(device=device, dtype=torch.float32)

        # フレーム（T次元）ごとに異なるシグマ値を持つテンソルを作成
        sigma_per_frame = torch.full((1, 1, T, 1, 1), first_sigma_for_others, device=latents.device, dtype=torch.float32)
        # 最初のキーフレーム(T=0)だけ、弱いノイズ用のシグマで上書き
        sigma_per_frame[:, :, 0, :, :] = first_sigma_for_initial

        # initial_latentを適切なデバイスと型に変換
        initial_latent_ready = initial_latent.to(device=latents.device, dtype=torch.float32)
            
        # フレームごとに異なる強度でノイズを付加
        # initial_latentは(B,C,1,H,W)なので、T次元に沿ってブロードキャストされる
        latents = initial_latent_ready * (1.0 - sigma_per_frame) + latents * sigma_per_frame

    k_model = fm_wrapper(transformer)


    if concat_latent is not None:
        concat_latent = concat_latent.to(latents)
        
    distilled_guidance = torch.tensor([distilled_guidance_scale * 1000.0] * batch_size).to(device=device, dtype=dtype)

    prompt_embeds = repeat_to_batch_size(prompt_embeds, batch_size)
    prompt_embeds_mask = repeat_to_batch_size(prompt_embeds_mask, batch_size)
    prompt_poolers = repeat_to_batch_size(prompt_poolers, batch_size)
    negative_prompt_embeds = repeat_to_batch_size(negative_prompt_embeds, batch_size)
    negative_prompt_embeds_mask = repeat_to_batch_size(negative_prompt_embeds_mask, batch_size)
    negative_prompt_poolers = repeat_to_batch_size(negative_prompt_poolers, batch_size)
    concat_latent = repeat_to_batch_size(concat_latent, batch_size)

    sampler_kwargs = dict(
        dtype=dtype,
        cfg_scale=real_guidance_scale,
        cfg_rescale=guidance_rescale,
        concat_latent=concat_latent,
        positive=dict(
            pooled_projections=prompt_poolers,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=distilled_guidance,
            **kwargs,
        ),
        negative=dict(
            pooled_projections=negative_prompt_poolers,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_prompt_embeds_mask,
            guidance=distilled_guidance,
            **(kwargs if negative_kwargs is None else {**kwargs, **negative_kwargs}),
        )
    )

    if sampler == 'unipc':
        results = sample_unipc(k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False, callback=callback)
    else:
        raise NotImplementedError(f'Sampler {sampler} is not supported.')

    return results
