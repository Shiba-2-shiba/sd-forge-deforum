import torch


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=1.0):
    if guidance_rescale == 0:
        return noise_cfg

    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg
    return noise_cfg


def fm_wrapper(transformer, device, t_scale=1000.0):
    def k_model(x, sigma, **extra_args):

        dtype = extra_args['dtype']
        cfg_scale = extra_args['cfg_scale']
        cfg_rescale = extra_args['cfg_rescale']
        concat_latent = extra_args['concat_latent']

        original_dtype = x.dtype
        
        # --- ▼▼▼【修正箇所】▼▼▼ ---
        # 1. 推論に使用するターゲットデバイスを明示的に使用
        model_device = device

        # 2. 【デバッグログ】各デバイスの状態を出力
        actual_device = next(transformer.parameters()).device
        print(f"[DEBUG wrapper.py] Devices: input_x={x.device}, sampler_arg={device}, actual_model_params={actual_device}")

        # 3. 入力テンソルをターゲットデバイスに移動
        x = x.to(device=model_device, dtype=dtype)
        sigma = sigma.to(device=model_device).float()
        # --- ▲▲▲【修正箇所】▲▲▲ ---

        timestep = (sigma * t_scale).to(dtype)

        if concat_latent is None:
            hidden_states = x
        else:
            # concat_latentも同じデバイスに転送
            concat_latent = concat_latent.to(device=model_device)
            hidden_states = torch.cat([x, concat_latent], dim=1)

        # 4. 【デバッグログ】transformer呼び出し直前の状態を出力
        print(f"[DEBUG wrapper.py] Before transformer call: hidden_states.device={hidden_states.device}")

        pred_positive = transformer(hidden_states=hidden_states, timestep=timestep, return_dict=False, **extra_args['positive'])[0].float()
        

        if cfg_scale == 1.0:
            pred_negative = torch.zeros_like(pred_positive)
        else:
            pred_negative = transformer(hidden_states=hidden_states, timestep=timestep, return_dict=False, **extra_args['negative'])[0].float()

        pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
        pred = rescale_noise_cfg(pred_cfg, pred_positive, guidance_rescale=cfg_rescale)

        x0 = x.float() - pred.float() * append_dims(sigma, x.ndim)

        return x0.to(dtype=original_dtype)

    return k_model
