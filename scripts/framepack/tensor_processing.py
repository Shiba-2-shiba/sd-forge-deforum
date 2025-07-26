import torch
import os
from .progress_bar import make_progress_bar_html
from PIL import Image
import numpy as np
from .hunyuan import vae_decode
# VAEの因果的デコードに必要なフック関数をvae_cacheからインポートします
from .vae_cache import hook_vae, restore_vae
from .utils import save_bcthw_as_mp4


def print_tensor_info(tensor: torch.Tensor, name: str = "テンソル") -> None:
    """テンソルの詳細情報を出力する

    Args:
        tensor (torch.Tensor): 分析対象のテンソル
        name (str, optional): テンソルの名前. デフォルトは"テンソル"

    Returns:
        None: 標準出力に情報を表示
    """
    try:
        print(("[DEBUG] {0}の詳細分析:").format(name))
        print(("  - 形状: {0}").format(tensor.shape))
        print(("  - 型: {0}").format(tensor.dtype))
        print(("  - デバイス: {0}").format(tensor.device))
        print(
            ("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(
                tensor.min().item(),
                tensor.max().item(),
                tensor.mean().item(),
            )
        )
        if torch.cuda.is_available():
            print(
                ("  - 使用GPUメモリ: {0:.2f}GB/{1:.2f}GB").format(
                    torch.cuda.memory_allocated() / 1024**3,
                    torch.cuda.get_device_properties(0).total_memory / 1024**3,
                )
            )
    except Exception as e:
        print(("[警告] テンソル情報の出力に失敗: {0}").format(str(e)))


def ensure_tensor_properties(
    tensor: torch.Tensor,
    target_device: torch.device,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """テンソルのデバイスと型を確認・調整する

    Args:
        tensor (torch.Tensor): 調整対象のテンソル
        target_device (torch.device): 目標のデバイス
        target_dtype (torch.dtype, optional): 目標のデータ型. デフォルトはfloat16

    Returns:
        torch.Tensor: デバイスと型が調整されたテンソル

    Raises:
        RuntimeError: テンソルの調整に失敗した場合
    """
    try:
        if tensor.device != target_device:
            tensor = tensor.to(target_device)
        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        return tensor
    except Exception as e:
        raise RuntimeError(
            ("テンソルプロパティの調整に失敗: {0}").format(str(e))
        )

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ エラーを修正し、ループを最適化したVAEデコード関数 ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def process_latents(
    latents: torch.Tensor,
    vae: torch.nn.Module,
    use_vae_cache: bool = True, # このモデルでは因果的デコードが必須のため、フラグに関わらず常に実行
    debug_str: str = "",
) -> torch.Tensor:
    """
    VAEデコード：latentをpixels化する (因果的デコード ＋ ループ最適化版)
    Hunyuan VAEの因果的（Causal）な性質を維持しつつ、ループ内のtorch.catを排除して高速化します。
    """
    if latents.dim() != 5:
        raise ValueError(f"Latent tensor must be 5D, but got shape {latents.shape}")

    print(f"--- VAE Causal Decode (Optimized Loop) 開始 {debug_str} ---")
    print_tensor_info(latents, "入力Latent")

    # 1. デコードの準備
    latents = latents / vae.config.scaling_factor
    latents = latents.to(device=vae.device, dtype=vae.dtype)
    frames = latents.shape[2]
    
    # 2. VAEに因果的デコード用のフックを適用
    hook_vae(vae)
    
    decoded_slices = [] # デコード結果を格納するリスト
    try:
        # 3. 1フレームずつループ処理 (このモデルのアーキテクチャでは必須)
        for i in range(frames):
            latents_slice = latents[:, :, i:i+1, :, :]
            
            # VAEのdecodeメソッドはフレーム間のキャッシュ（状態）をフックを通じて管理する
            image_slice = vae.decode(latents_slice).sample
            
            # ★最適化ポイント: テンソルを直接連結せず、リストに追加する
            decoded_slices.append(image_slice)
            
    except Exception as e:
        print(f"[エラー] VAEのフレームごとのデコード中にエラーが発生しました: {e}")
        # エラー発生時も必ずフックを元に戻す
        restore_vae(vae)
        raise e
    finally:
        # 4. VAEのフックを解除し、元の状態に戻す
        restore_vae(vae)

    # 5. ★最適化ポイント: ループ終了後、リスト内の全スライスを一度に結合
    pixels = torch.cat(decoded_slices, dim=2) # 時間（フレーム）次元で結合
    
    print(f"[DEBUG] 最終的なピクセル形状に復元: {pixels.shape}")
    print("--- VAE Causal Decode (Optimized Loop) 完了 ---")
    
    # 後続の処理のため、結果をCPUメモリに移動
    return pixels.cpu()


def process_tensor_chunk(
    chunk_idx: int,
    current_chunk: torch.Tensor,
    num_chunks: int,
    chunk_start: int,
    chunk_end: int,
    frames: int,
    use_vae_cache: bool,
    vae: torch.nn.Module,
    stream: any,
    reverse: bool,
) -> torch.Tensor:
    """個別のテンソルチャンクを処理する（この関数は変更なし）"""
    try:
        chunk_frames = chunk_end - chunk_start

        if chunk_frames <= 0:
            raise ValueError(f"不正なチャンクサイズ: {chunk_frames} (start={chunk_start}, end={chunk_end})")
        if current_chunk.shape[2] <= 0:
            raise ValueError(f"不正なテンソル形状: {current_chunk.shape}")

        chunk_progress = (chunk_idx + 1) / num_chunks * 100
        progress_message = ("テンソルデータ結合中: チャンク {0}/{1} (フレーム {2}-{3}/{4})").format(chunk_idx + 1, num_chunks, chunk_start, chunk_end, frames)
        stream.output_queue.push(("progress", (None, progress_message, make_progress_bar_html(int(80 + chunk_progress * 0.1), ("テンソルデータ処理中")))))

        print(("チャンク{0}/{1}処理中: フレーム {2}-{3}/{4}").format(chunk_idx + 1, num_chunks, chunk_start, chunk_end, frames))

        if torch.cuda.is_available():
            print(("[MEMORY] チャンク{0}処理前のGPUメモリ: {1:.2f}GB/{2:.2f}GB").format(chunk_idx + 1, torch.cuda.memory_allocated() / 1024**3, torch.cuda.get_device_properties(0).total_memory / 1024**3))
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.empty_cache(); import gc; gc.collect()

        print(("[INFO] VAEデコード開始: チャンク{0}").format(chunk_idx + 1))
        stream.output_queue.push(("progress", (None, ("チャンク{0}/{1}のVAEデコード中...").format(chunk_idx + 1, num_chunks), make_progress_bar_html(int(80 + chunk_progress * 0.1), ("デコード処理")))))
        print_tensor_info(current_chunk, "チャンク{0}".format(chunk_idx + 1))

        current_chunk = ensure_tensor_properties(current_chunk, vae.device)

        # 修正された process_latents を呼び出す
        chunk_pixels = process_latents(current_chunk, vae, use_vae_cache, ("チャンク"))
        
        print(("チャンク{0}のVAEデコード完了 (入力フレーム数: {1}, 出力形状: {2})").format(chunk_idx + 1, chunk_frames, chunk_pixels.shape))

        if reverse:
            chunk_pixels = reorder_tensor(chunk_pixels)
        return chunk_pixels

    except Exception as e:
        error_msg = ("チャンク{0}の処理中にエラーが発生: {1}").format(chunk_idx + 1, str(e))
        print(f"[エラー] {error_msg}")
        raise RuntimeError(error_msg)


def process_tensor_chunks(
    tensor: torch.Tensor,
    frames: int,
    use_vae_cache: bool,
    job_id: str,
    outputs_folder: str,
    mp4_crf: int,
    stream: any,
    vae: torch.nn.Module,
    reverse: bool = False,
    skip_save: bool = True,
) -> tuple[torch.Tensor, int]:
    """テンソルデータをチャンクに分割して処理する（この関数は変更なし）"""
    try:
        if frames <= 0:
            raise ValueError(f"不正なフレーム数: {frames}")

        combined_pixels = None
        chunk_size = min(5, frames)
        num_chunks = (frames + chunk_size - 1) // chunk_size

        print(f"[DEBUG] フレーム総数: {frames}"); print(f"[DEBUG] チャンクサイズ: {chunk_size}"); print(f"[DEBUG] チャンク数: {num_chunks}"); print(f"[DEBUG] 入力テンソル形状: {tensor.shape}"); print(f"[DEBUG] VAEキャッシュ(旧フラグ): {use_vae_cache}"); print(f"[DEBUG] ジョブID: {job_id}"); print_tensor_info(tensor, "入力テンソル")

        if tensor.shape[2] != frames:
            raise ValueError(f"テンソル形状不一致: テンソルのフレーム数 {tensor.shape[2]} != 指定フレーム数 {frames}")

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, frames)

            if chunk_end <= chunk_start:
                print(f"[警告] 不正なチャンク範囲をスキップ: start={chunk_start}, end={chunk_end}"); continue

            try:
                current_chunk = tensor[:, :, chunk_start:chunk_end, :, :]
                chunk_pixels = process_tensor_chunk(chunk_idx, current_chunk, num_chunks, chunk_start, chunk_end, frames, use_vae_cache, vae, stream, reverse)

                if combined_pixels is None:
                    combined_pixels = chunk_pixels
                else:
                    combined_pixels = ensure_tensor_properties(combined_pixels, torch.device("cpu"))
                    combined_pixels = torch.cat([combined_pixels, chunk_pixels], dim=2)

                current_total_frames = combined_pixels.shape[2]
                print(("チャンク{0}の結合完了: 現在の組み込みフレーム数 = {1}").format(chunk_idx + 1, current_total_frames))

                if not skip_save and (chunk_idx == num_chunks - 1 or (chunk_idx > 0 and (chunk_idx + 1) % 5 == 0)):
                    interim_output_filename = os.path.join(outputs_folder, f"{job_id}_combined_interim_{chunk_idx + 1}.mp4")
                    print(("中間結果を保存中: チャンク{0}/{1}").format(chunk_idx + 1, num_chunks))
                    
                    chunk_progress = (chunk_idx + 1) / num_chunks * 100
                    stream.output_queue.push(("progress", (None, ("中間結果のMP4変換中... (チャンク{0}/{1})").format(chunk_idx + 1, num_chunks), make_progress_bar_html(int(85 + chunk_progress * 0.1), ("MP4保存中")))))
                    
                    save_bcthw_as_mp4(combined_pixels, interim_output_filename, fps=30, crf=mp4_crf)
                    print(("中間結果を保存しました: {0}").format(interim_output_filename))
                    stream.output_queue.push(("file", interim_output_filename))

                del current_chunk, chunk_pixels
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                error_msg = ("チャンク{0}の処理中にエラー: {1}").format(chunk_idx + 1, str(e))
                print(f"[エラー] {error_msg}"); raise

        return combined_pixels, num_chunks

    except Exception as e:
        error_msg = ("テンソル処理中に重大なエラーが発生: {0}").format(str(e))
        print(f"[重大エラー] {error_msg}"); raise RuntimeError(error_msg)


def output_latent_to_image(
    latent: torch.Tensor, file_path: str, vae: torch.nn.Module, use_vae_cache: bool = False
) -> None:
    """VAEを使用してlatentを画像化してファイル出力する（この関数は変更なし）"""
    if latent.dim() != 5:
        raise Exception(f"Error: latent dimension must be 5, got {latent.dim()}")

    image_pixels = process_latents(latent, vae, use_vae_cache)
    image_pixels = ((image_pixels[0, :, 0] * 127.5 + 127.5).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8))
    Image.fromarray(image_pixels).save(file_path)


def fix_tensor_size(
    tensor: torch.Tensor, target_size: int = 1 + 2 + 16, fix_edge=True
) -> torch.Tensor:
    """tensorのフレーム情報が小さい場合に補間する（この関数は変更なし）"""
    if tensor.dim() != 5:
        raise Exception(f"Programing Error: latent dim != 5. {tensor.shape}")
    if tensor.shape[2] < target_size:
        print(f"[WARN] latentサイズが足りないので補間します。{tensor.shape[2]}=>{target_size}")
        mode = "nearest"
        if fix_edge:
            first_frame, last_frame, middle_frames = tensor[:, :, :1, :, :], tensor[:, :, -1:, :, :], tensor[:, :, 1:-1, :, :]
            fixed_tensor = torch.nn.functional.interpolate(middle_frames, size=(target_size - 2, tensor.shape[3], tensor.shape[4]), mode=mode)
            fixed_tensor = torch.cat([first_frame, fixed_tensor, last_frame], dim=2)
        else:
            fixed_tensor = torch.nn.functional.interpolate(tensor, size=(target_size, tensor.shape[3], tensor.shape[4]), mode=mode)
    else:
        fixed_tensor = tensor
    return fixed_tensor


def reorder_tensor(tensor: torch.Tensor, reverse: bool = False) -> torch.Tensor:
    """テンソルのフレーム順序を操作する（この関数は変更なし）"""
    if reverse:
        return tensor.flip(dims=[2])
    return tensor
