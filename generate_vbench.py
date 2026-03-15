import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime

# Disable torch compile to avoid inductor import errors
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

warnings.filterwarnings('ignore')

import random
import traceback
import psutil

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import merge_video_audio, save_video, str2bool


_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
_VBENCH_ROOT       = os.path.join(_SCRIPT_DIR, "..", "VBench", "vbench2_beta_i2v")
_DEFAULT_INFO_JSON = os.path.join(_VBENCH_ROOT, "vbench2_i2v_full_info.json")
_DEFAULT_CROP_DIR  = os.path.join(_VBENCH_ROOT, "vbench2_beta_i2v", "data", "crop")
def _safe(s):
    return re.sub(r'[<>:"/\\|?*]', "_", s)[:150]


EXAMPLE_PROMPT = {
    "i2v-A14B": {
        "prompt":
            "The video presents a cinematic, first-person wandering experience through a hyper-realistic urban environment rendered in a video game engine. It begins with a static, sun-drenched alley framed by graffiti-laden industrial walls and overhead power lines, immediately establishing a gritty, lived-in atmosphere. As the camera pans right and tilts upward, it reveals a sprawling cityscape dominated by towering skyscrapers and industrial infrastructure, all bathed in warm, late-afternoon light that casts long shadows and produces dramatic lens flares. The perspective then transitions into a smooth forward tracking shot along a cracked sidewalk, passing weathered fences, palm trees, and distant pedestrians, creating a sense of immersion and exploration. Midway, the camera briefly follows a walking figure before refocusing on the broader streetscape, culminating in a stabilized view of a small blue van parked at an intersection surrounded by urban elements like parking garages and traffic lights. The entire sequence is characterized by its photorealistic detail, dynamic lighting, and deliberate pacing, evoking the feel of a quiet, sunlit afternoon in a futuristic metropolis.",
        "image":
            "examples/02/image.jpg",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=161,
        help="How many frames of video are generated. The number should be 4n+1 (default: 161)"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=False,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=True,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=5, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    # ---- VBench batch args ----
    parser.add_argument("--vbench", action="store_true", default=True,
        help="Run VBench batch generation instead of single-video mode.")
    parser.add_argument("--image_types", type=str, default="indoor,scenery",
        help="Comma-separated image_type values to include (default: scenery,indoor).")
    parser.add_argument("--vbench_output_dir", type=str, default="results_vbench/videos",
        help="Output directory for vbench videos.")
    parser.add_argument("--num_samples", type=int, default=5,
        help="Number of samples per prompt.")
    parser.add_argument("--vbench_info_json", type=str, default=None,
        help="Path to vbench2_i2v_full_info.json.")
    parser.add_argument("--crop_dir", type=str, default=None,
        help="Path to VBench crop directory.")
    parser.add_argument("--resolution", type=str, default="1-1",
        help="Crop resolution subfolder.")
    parser.add_argument("--stats_file", type=str, default=None,
        help="CSV stats output path (default: <vbench_output_dir>/../vbench_stats.csv).")
    parser.add_argument("--max_prompts", type=int, default=None,
        help="Limit number of prompts processed (default: all).")
    parser.add_argument("--start_prompt", type=int, default=0,
        help="Start from this prompt index (0-based), skipping earlier ones.")
    parser.add_argument("--skip_existing", action="store_true", default=True,
        help="Skip videos that already exist on disk (default: True).")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--nf4", action="store_true", default=True,
        help="Load DiT models in NF4 (4-bit) quantization via bitsandbytes. Reduces each model from ~35 GB to ~9 GB VRAM.")
    parser.add_argument("--no_nf4", action="store_false", dest="nf4",
        help="Disable NF4 quantization (requires ~35 GB VRAM per model).")

    args = parser.parse_args()
    if args.vbench:
        assert args.ckpt_dir is not None, "Please specify --ckpt_dir (path to Wan checkpoint directory)."
    else:
        _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    logging.info("Starting the generation process...")
    
    if args.offload_model is None:
        args.offload_model = True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        logging.info("Initializing distributed environment...")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
        logging.info("Distributed environment initialized.")
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    logging.info("Loading model configuration...")
    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        logging.info(f"Loading input image from {args.image}...")
        img = Image.open(args.image).convert("RGB")
        logging.info("Input image loaded.")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt...")
        if rank == 0:
            input_prompt = args.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")
    
    logging.info("Creating WanI2V pipeline...")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    logging.info("WanI2V pipeline created.")

    logging.info("Generating video...")
    video = wan_i2v.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)
    logging.info("Video generation completed.")

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}...")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        if "s2v" in args.task:
            if args.enable_tts is False:
                merge_video_audio(video_path=args.save_file, audio_path=args.audio)
            else:
                merge_video_audio(video_path=args.save_file, audio_path="tts.wav")
        logging.info("Video saved successfully.")
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Generation process finished.")


def vbench_batch(args):
    info_json = os.path.abspath(args.vbench_info_json or _DEFAULT_INFO_JSON)
    crop_base = os.path.abspath(args.crop_dir or _DEFAULT_CROP_DIR)
    image_dir = os.path.join(crop_base, args.resolution)
    out_dir   = os.path.abspath(args.vbench_output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats_path = os.path.abspath(args.stats_file) if args.stats_file else os.path.join(os.path.dirname(out_dir), 'vbench_stats.csv')
    stats_f    = open(stats_path, 'w', newline='', encoding='utf-8')
    stats_w    = csv.writer(stats_f)
    stats_w.writerow(['task_idx', 'prompt', 'sample_idx', 'duration_s', 'gen_fps', 'ram_gb', 'vram_gb', 'out_path', 'status'])

    if not os.path.isfile(info_json):
        print(f'[vbench] ERROR: info JSON not found: {info_json}'); return
    if not os.path.isdir(image_dir):
        print(f'[vbench] ERROR: crop dir not found: {image_dir}'); return

    with open(info_json, encoding='utf-8') as f:
        entries = json.load(f)

    allowed  = {t.strip() for t in args.image_types.split(',') if t.strip()} if args.image_types else None
    populate = None

    seen, prompts = set(), []
    for e in entries:
        name = e['image_name']
        if name in seen: continue
        if allowed and e.get('image_type') not in allowed: continue
        if populate is not None and (e.get('image_type') in _POPULATED_TYPES) != populate: continue
        seen.add(name)
        prompts.append((name, e['prompt_en']))
    if args.start_prompt:
        prompts = prompts[args.start_prompt:]
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]

    print(f'[vbench] {len(prompts)} prompts × {args.num_samples} samples = {len(prompts) * args.num_samples} total')

    cfg = WAN_CONFIGS[args.task]

    if torch.cuda.is_available():
        free_gb, total_gb = [x / 1024**3 for x in torch.cuda.mem_get_info()]
        print(f'[vbench] VRAM before model load: {free_gb:.1f} GB free / {total_gb:.1f} GB total')
    quantization_config = None
    if args.nf4:
        from diffusers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print(f'[vbench] NF4 quantization enabled — models will load in ~9 GB VRAM each')

    print(f'[vbench] loading WanI2V from {args.ckpt_dir} ...')
    try:
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            quantization_config=quantization_config,
        )
    except Exception as exc:
        print(f'[vbench] FATAL: model load failed — {type(exc).__name__}: {exc}')
        traceback.print_exc()
        stats_f.close()
        return
    if torch.cuda.is_available():
        free_gb, total_gb = [x / 1024**3 for x in torch.cuda.mem_get_info()]
        print(f'[vbench] VRAM after  model load: {free_gb:.1f} GB free / {total_gb:.1f} GB total')

    skipped = generated = errors = 0
    total      = len(prompts) * args.num_samples
    done       = 0
    ok_total_s = 0.0
    t_start    = time.time()
    print(f'[vbench] {len(prompts)} prompts × {args.num_samples} samples = {total} total')

    def _fmt(secs):
        h, m, s = int(secs//3600), int(secs%3600//60), int(secs%60)
        return f'{h:02d}h{m:02d}m{s:02d}s'

    for task_idx, (image_name, prompt) in enumerate(prompts):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            print(f'[vbench] skip {task_idx}: image not found — {image_path}')
            continue

        img = Image.open(image_path).convert("RGB")
        prompt_cache = {}

        for sample_idx in range(args.num_samples):
            seed = args.base_seed + sample_idx
            out_path = os.path.join(out_dir, f'{_safe(prompt)}-{sample_idx}-{seed}.mp4')
            if args.skip_existing and os.path.exists(out_path):
                skipped += 1
                done += 1
                stats_w.writerow([task_idx, prompt, sample_idx, '', '', '', '', out_path, 'skipped'])
                stats_f.flush()
                continue

            pct     = 100 * done / total if total else 0
            elapsed = time.time() - t_start
            eta = avg = ''
            if generated > 0:
                avg_s     = ok_total_s / generated
                secs_left = avg_s * (total - done)
                eta = f'  ETA {_fmt(secs_left)}'
                avg = f'  avg {avg_s/60:.1f}min/video'
            vram_free = torch.cuda.mem_get_info()[0] / 1024**3 if torch.cuda.is_available() else 0.0
            print(f'\n[vbench] [{done+1}/{total}  {pct:.0f}%{eta}{avg}]  elapsed {_fmt(elapsed)}')
            print(f'[vbench] prompt {task_idx+1}/{len(prompts)}  sample {sample_idx+1}/{args.num_samples}  seed {seed}  VRAM free {vram_free:.1f} GB')
            print(f'[vbench]   image : {image_name}')
            print(f'[vbench]   prompt: {prompt[:120]}')
            print(f'[vbench]   seed  : {seed}  out: {os.path.basename(out_path)}')
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                with torch.inference_mode():
                    t0 = time.time()
                    video = wan_i2v.generate(
                        prompt, img,
                        max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift or cfg.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps or cfg.sample_steps,
                        guide_scale=args.sample_guide_scale or cfg.sample_guide_scale,
                        seed=seed,
                        offload_model=args.offload_model,
                        _cache=prompt_cache,
                    )
                    elapsed = time.time() - t0
                gen_fps    = args.frame_num / elapsed if elapsed > 0 else 0.0
                ram_gb     = psutil.virtual_memory().used / 1024**3
                vram_gb    = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                vram_peak  = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                from wan.utils.utils import save_video as _save_video
                _save_video(tensor=video[None], save_file=out_path, fps=cfg.sample_fps,
                            nrow=1, normalize=True, value_range=(-1, 1))
                print(f'[vbench]   OK  {elapsed:.1f}s  {gen_fps:.2f} gen-fps  VRAM {vram_gb:.1f} GB (peak {vram_peak:.1f} GB)  RAM {ram_gb:.1f} GB')
                stats_w.writerow([task_idx, prompt, sample_idx, f'{elapsed:.2f}', f'{gen_fps:.2f}',
                                  f'{ram_gb:.2f}', f'{vram_gb:.2f}', out_path, 'ok'])
                stats_f.flush()
                ok_total_s += elapsed
                generated  += 1
            except Exception as exc:
                ram_gb  = psutil.virtual_memory().used / 1024**3
                vram_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                print(f'[vbench]   ERROR task {task_idx} sample {sample_idx}: {type(exc).__name__}: {exc}')
                print(f'[vbench]   image_path: {image_path}')
                print(f'[vbench]   out_path  : {out_path}')
                print(f'[vbench]   RAM {ram_gb:.1f} GB  VRAM {vram_gb:.1f} GB')
                traceback.print_exc()
                stats_w.writerow([task_idx, prompt, sample_idx, '', '', f'{ram_gb:.2f}', f'{vram_gb:.2f}', out_path, f'error:{type(exc).__name__}'])
                stats_f.flush()
                errors += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc; gc.collect()
                try:
                    for attr in ('low_noise_model', 'high_noise_model'):
                        m = getattr(wan_i2v, attr, None)
                        if m is not None:
                            m.to('cpu')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            done += 1

    elapsed_total = time.time() - t_start
    stats_f.close()
    print(f'\n[vbench] done — generated={generated}  skipped={skipped}  errors={errors}  elapsed={_fmt(elapsed_total)}')
    if generated:
        print(f'[vbench] avg per video: {ok_total_s/generated/60:.1f} min')
    print(f'[vbench] stats → {stats_path}')


if __name__ == "__main__":
    args = _parse_args()
    if args.vbench:
        vbench_batch(args)
    else:
        generate(args)
