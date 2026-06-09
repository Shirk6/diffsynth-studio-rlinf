import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    torch.serialization.add_safe_globals(["set", "OrderedDict", "builtins.set"])
except AttributeError:
    pass

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline


DEFAULT_DATA_ROOT = "/project/peilab/srk/rss_2026_ws/Challenge-phase1-dataset-rlinf/tower-of-hanoi-game/val-data"
DEFAULT_MODEL_ROOT = "/project/peilab/srk/rss_2026_ws/models/Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_CKPT = "/project/peilab/srk/rss_2026_ws/diffsynth-studio/outputs/Wan2.2-TI2V-5B_challenge_rlinf/epoch-3599.safetensors"
DEFAULT_OUTPUT = "/project/peilab/srk/rss_2026_ws/diffsynth-studio/outputs/challenge_rlinf_autoregressive"


def parse_args():
    parser = argparse.ArgumentParser(description="Autoregressive inference for Wan2.2-TI2V-5B challenge RLinf checkpoints.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model_root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--episode", default=None, help="Substring matched against a val-data segment path. Defaults to the first valid segment.")
    parser.add_argument("--env_id", type=int, default=0)
    parser.add_argument("--max_sampled_frames", type=int, default=121, help="Maximum generated timeline length after 6-frame downsampling.")
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--save_compare", action="store_true", help="Also save a side-by-side GT/prediction video.")
    parser.add_argument("--no_latent_condition_cache", action="store_true", help="Use decoded frames as conditions instead of reusing condition latents.")
    return parser.parse_args()


def find_episode(data_root, episode_filter=None):
    root = Path(data_root)
    candidates = []
    for step_dir in sorted(root.iterdir()):
        if not step_dir.is_dir():
            continue
        for seg_dir in sorted(step_dir.iterdir()):
            if not seg_dir.is_dir():
                continue
            real_dir = seg_dir.resolve()
            if (real_dir / "rgb.npy").exists() and (real_dir / "actions.npy").exists():
                display_path = str(seg_dir)
                if episode_filter is None or episode_filter in display_path or episode_filter in str(real_dir):
                    candidates.append((seg_dir, real_dir))
    if not candidates:
        raise FileNotFoundError(f"No rgb.npy/actions.npy segment found under {data_root} with episode filter {episode_filter!r}.")
    return candidates[0]


def frame_to_image(frame):
    if frame.ndim == 3 and frame.shape[0] in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    if frame.max() <= 1.0:
        frame = (frame * 255).clip(0, 255)
    return Image.fromarray(frame.astype(np.uint8))


def load_downsampled_episode(segment_dir, env_id, max_sampled_frames):
    rgb = np.load(segment_dir / "rgb.npy", mmap_mode="r")
    actions = np.load(segment_dir / "actions.npy", mmap_mode="r")
    if env_id >= rgb.shape[1]:
        raise ValueError(f"env_id={env_id} is out of range for rgb shape {rgb.shape}")

    raw_ids = np.concatenate([[0], np.arange(5, rgb.shape[0], 6)])
    if max_sampled_frames is not None:
        raw_ids = raw_ids[:max_sampled_frames]
    if len(raw_ids) < 13:
        raise ValueError(f"Need at least 13 downsampled frames, got {len(raw_ids)} from {segment_dir}")

    frames = [frame_to_image(rgb[i, env_id]) for i in raw_ids]
    action = torch.from_numpy(np.asarray(actions[raw_ids, env_id])).float()
    first_action = torch.zeros(action.shape[-1], dtype=action.dtype)
    first_action[-1] = -1
    action[0] = first_action
    return frames, action, raw_ids


def load_pipeline(args):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=args.ckpt, offload_device="cpu"),
            ModelConfig(path=os.path.join(args.model_root, "Wan2.2_VAE.pth"), offload_device="cpu"),
        ],
    )
    pipe.dit.to(args.device)
    pipe.vae.to(args.device)
    return pipe


def encode_condition_latents(pipe, frames, height, width, tiled=False):
    pipe.load_models_to_device(["vae"])
    cond_video = pipe.preprocess_video([[frame.resize((width, height)) for frame in frames]])
    return pipe.vae.encode(cond_video, device=pipe.device, tiled=tiled).to(dtype=pipe.torch_dtype, device=pipe.device)


def action_slice(action_full, idx, device):
    idx = np.clip(np.asarray(idx), 0, len(action_full) - 1)
    action = action_full[idx].clone()
    action[0] = 0
    action[0, -1] = -1
    return action.to(dtype=torch.bfloat16, device=device)


def add_label(img, text):
    w, h = img.size
    bar_h = 70
    out = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    for i, line in enumerate(text.splitlines()):
        draw.text((8, 5 + i * 20), line, fill=(0, 0, 0))
    return out


def concat_compare(gt_frames, pred_frames, raw_ids):
    frames = []
    for i, (gt, pred) in enumerate(zip(gt_frames, pred_frames)):
        gt_labeled = add_label(gt, f"GT sampled={i}\nraw={int(raw_ids[i])}")
        pred_labeled = add_label(pred, f"AR pred sampled={i}\nraw={int(raw_ids[i])}")
        gh, ph = np.array(gt_labeled), np.array(pred_labeled)
        h = min(gh.shape[0], ph.shape[0])
        frames.append(Image.fromarray(np.concatenate([gh[:h], ph[:h]], axis=1)))
    return frames


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    display_dir, segment_dir = find_episode(args.data_root, args.episode)
    print(f"Using segment: {display_dir} -> {segment_dir}")

    gt_frames, action_full, raw_ids = load_downsampled_episode(segment_dir, args.env_id, args.max_sampled_frames)
    total_frames = len(gt_frames)
    print(f"Loaded {total_frames} sampled frames from raw ids {raw_ids[0]}..{raw_ids[-1]}")

    pipe = load_pipeline(args)

    condition_frames = 5
    predict_frames = 8
    window = condition_frames + predict_frames
    num_iters = max(0, (total_frames - condition_frames + predict_frames - 1) // predict_frames)
    print(f"Autoregressive chunks: {num_iters}, window={window}, condition={condition_frames}, predict={predict_frames}")

    generated = gt_frames[:condition_frames]
    input_image = gt_frames[0]
    input_image4 = gt_frames[1:condition_frames]
    condition_latents = None
    if not args.no_latent_condition_cache:
        condition_latents = encode_condition_latents(pipe, [input_image] + input_image4, args.height, args.width)

    last_latents = None
    for chunk_id in range(num_iters):
        predict_start = condition_frames + chunk_id * predict_frames
        predict_end = predict_start + predict_frames
        context_start = predict_start - (condition_frames - 1)
        idx = [0] + list(range(context_start, predict_end))
        print(f"Chunk {chunk_id + 1}/{num_iters}: context_idx={idx[:condition_frames]}, predict_idx={idx[condition_frames:]}")

        out_video, last_latents = pipe(
            seed=args.seed + chunk_id,
            tiled=False,
            input_image=input_image,
            input_image4=input_image4,
            condition_latents=None if args.no_latent_condition_cache else condition_latents,
            action=action_slice(action_full, idx, args.device),
            height=args.height,
            width=args.width,
            num_frames=window,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
            sigma_shift=args.sigma_shift,
            bs_1=True,
            return_latents=True,
        )
        out_video = out_video[0]

        new_frames = out_video[-predict_frames:]
        generated.extend(new_frames)
        generated = generated[:total_frames]

        if len(generated) >= total_frames:
            break

        input_image4 = generated[-4:]
        if args.no_latent_condition_cache:
            condition_latents = None
        else:
            first_latent = condition_latents[:, :, 0:1]
            last_generated_latent = last_latents[:, :, -1:]
            condition_latents = torch.cat([first_latent, last_generated_latent], dim=2)

        gc.collect()
        torch.cuda.empty_cache()

    pred_path = os.path.join(args.output_dir, f"{segment_dir.name}_ar_pred.mp4")
    save_video(generated, pred_path, fps=args.fps, quality=5)
    print(f"Saved prediction: {pred_path}")

    if args.save_compare:
        compare = concat_compare(gt_frames[:len(generated)], generated, raw_ids[:len(generated)])
        compare_path = os.path.join(args.output_dir, f"{segment_dir.name}_ar_compare.mp4")
        save_video(compare, compare_path, fps=args.fps, quality=5)
        print(f"Saved comparison: {compare_path}")


if __name__ == "__main__":
    main()
