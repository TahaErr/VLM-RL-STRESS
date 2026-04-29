"""
VLM-RL Training — Single Seed Worker.

train.py'nin --seed, --output-dir, --resume destekli versiyonu.
run_training.py orchestrator tarafından subprocess olarak çağrılır.

KULLANIM:
    # Yeni training
    python train_one_seed.py --seed 42 --output-dir tensorboard/seed_42

    # Resume (output-dir'de checkpoint varsa otomatik kaldığı yerden)
    python train_one_seed.py --seed 42 --output-dir tensorboard/seed_42 --resume

    # Custom step count (test için)
    python train_one_seed.py --seed 42 --output-dir tensorboard/seed_42_smoke --total-timesteps 50000

ÇIKTI:
    output-dir/
        config.json                      # CONFIG snapshot
        progress.csv                     # SB3 logger CSV
        events.out.tfevents.*           # tensorboard
        model_50000_steps.zip            # her 50K step
        model_100000_steps.zip
        ...
        replay_buffer.pkl                # her 100K step (resume için)
"""

import warnings
import os
import sys
import argparse
import time

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Train VLM-RL for one seed")
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", default=2000, type=int)
parser.add_argument("--seed", type=int, required=True,
                    help="Random seed (overrides CONFIG.seed)")
parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                    help="Total training steps (default: 1M, paper-faithful)")
parser.add_argument("--output-dir", type=str, required=True,
                    help="Where to save checkpoints + logs")
parser.add_argument("--config", type=str, default="vlm_rl",
                    help="Config name (default: vlm_rl, CLIP-SAC + ViT-bigG)")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--town", type=str, default="Town02",
                    help="Training town (paper: Town02)")
parser.add_argument("--start-carla", action="store_true",
                    help="Let env start CARLA (default: external)")
parser.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in output-dir")
parser.add_argument("--checkpoint-freq", type=int, default=50_000,
                    help="Save model every N steps (default: 50K)")
parser.add_argument("--replay-buffer-freq", type=int, default=100_000,
                    help="Save replay buffer every N steps (default: 100K)")
args = parser.parse_args()

print("=" * 72)
print(f"VLM-RL TRAINING — seed={args.seed}, town={args.town}")
print("=" * 72)
print(f"  Total timesteps:     {args.total_timesteps:,}")
print(f"  Output dir:          {args.output_dir}")
print(f"  Config:              {args.config}")
print(f"  Device:              {args.device}")
print(f"  Resume:              {args.resume}")
print(f"  Checkpoint freq:     {args.checkpoint_freq:,}")
print(f"  Replay buffer freq:  {args.replay_buffer_freq:,}")

# ============================================================
# CONFIG
# ============================================================
import config
CONFIG = config.set_config(args.config)
CONFIG.seed = args.seed                        # OVERRIDE
CONFIG.algorithm_params.device = args.device

print(f"\n[Config] algorithm={CONFIG.algorithm}, "
      f"reward_fn={CONFIG.reward_fn}, "
      f"vlm_reward_type={CONFIG.vlm_reward_type}")
print(f"[Config] CLIP model: {CONFIG.clip_reward_params.pretrained_model}")
print(f"[Config] CLG prompts: {CONFIG.clip_reward_params.target_prompts}")
print(f"[Config] seed: {CONFIG.seed} (overridden from CLI)")

# ============================================================
# CARLA PRE-FLIGHT
# ============================================================
if not args.start_carla:
    print("\n[Pre-flight] CARLA bağlantı kontrolü...")
    try:
        import carla as _carla_test
        _client = _carla_test.Client(args.host, args.port)
        _client.set_timeout(10.0)
        _ver = _client.get_server_version()
        _map = _client.get_world().get_map().name
        print(f"  [✓] CARLA: version={_ver}, map={_map}")
        del _client
    except Exception as e:
        print(f"  [✗] CARLA yanıt vermiyor: {e}")
        sys.exit(1)

# ============================================================
# IMPORTS (CONFIG sonrası)
# ============================================================
print("\n[Setup] Importing modules...")
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from clip.clip_rewarded_sac import CLIPRewardedSAC
from clip.clip_rewarded_ppo import CLIPRewardedPPO
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class

algorithm_dict = {
    "PPO": PPO, "DDPG": DDPG, "SAC": SAC,
    "CLIP-SAC": CLIPRewardedSAC, "CLIP-PPO": CLIPRewardedPPO,
}
AlgorithmRL = algorithm_dict[CONFIG.algorithm]

observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
action_space_type = 'continuous' if CONFIG.action_space_type != 'discrete' else 'discrete'

# ============================================================
# OUTPUT DIR + RESUME DETECTION
# ============================================================
os.makedirs(args.output_dir, exist_ok=True)

resume_checkpoint = None
resume_replay = None
if args.resume:
    ckpts = []
    for f in os.listdir(args.output_dir):
        if f.startswith("model_") and f.endswith("_steps.zip"):
            try:
                step = int(f.replace("model_", "").replace("_steps.zip", ""))
                ckpts.append((step, f))
            except ValueError:
                continue
    ckpts.sort()
    if ckpts:
        resume_step, resume_file = ckpts[-1]
        resume_checkpoint = os.path.join(args.output_dir, resume_file)
        replay_path = os.path.join(args.output_dir, "replay_buffer.pkl")
        if os.path.exists(replay_path):
            resume_replay = replay_path
        print(f"\n[Resume] Latest checkpoint: {resume_file} ({resume_step:,} steps)")
        print(f"[Resume] Replay buffer: {'found' if resume_replay else 'NOT FOUND (will start with empty buffer)'}")

        if resume_step >= args.total_timesteps:
            print(f"\n[SKIP] Training already complete ({resume_step:,} >= {args.total_timesteps:,})")
            sys.exit(0)
    else:
        print(f"\n[Resume] No checkpoint found in {args.output_dir}, starting fresh")

# ============================================================
# ENV
# ============================================================
print(f"\n[Env] Creating CarlaRouteEnv (town={args.town}, eval=False, traffic_flow=True)...")
t0 = time.time()
try:
    env = CarlaRouteEnv(
        obs_res=CONFIG.obs_res,
        host=args.host,
        port=args.port,
        reward_fn=reward_functions[CONFIG.reward_fn],
        observation_space=observation_space,
        encode_state_fn=encode_state_fn,
        fps=15,
        action_smoothing=CONFIG.action_smoothing,
        action_space_type=action_space_type,
        activate_spectator=False,
        activate_render=False,
        activate_bev=CONFIG.use_rgb_bev,
        activate_seg_bev=CONFIG.use_seg_bev,
        activate_traffic_flow=True,         # paper protokolü
        start_carla=args.start_carla,
        eval=False,                         # TRAINING mode
        town=args.town,
    )
    print(f"  [✓] Env created in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  [✗] Env creation FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Wrappers (config'de tanımlıysa)
for wrapper_class_str in CONFIG.wrappers:
    wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
    env = wrap_class(env, *wrap_params)

# ============================================================
# MODEL (FRESH or RESUME)
# ============================================================
if resume_checkpoint:
    print(f"\n[Model] Loading from {resume_checkpoint}...")
    t0 = time.time()
    try:
        model = CLIPRewardedSAC.load(
            resume_checkpoint, env=env, config=CONFIG,
            device=args.device, load_clip=True,
        )
        print(f"  [✓] Model loaded in {time.time()-t0:.1f}s")
        print(f"  num_timesteps: {model.num_timesteps:,}")
    except Exception as e:
        print(f"  [✗] Load failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    if resume_replay:
        print(f"[Model] Loading replay buffer from {resume_replay}...")
        t0 = time.time()
        try:
            model.load_replay_buffer(resume_replay)
            print(f"  [✓] Replay buffer loaded in {time.time()-t0:.1f}s "
                  f"(size: {model.replay_buffer.size():,})")
        except Exception as e:
            print(f"  [!] Replay buffer load failed: {e}, continuing with empty buffer")

    remaining = args.total_timesteps - model.num_timesteps
    print(f"\n[Model] Will train for {remaining:,} more steps "
          f"({model.num_timesteps:,} → {args.total_timesteps:,})")

else:
    print(f"\n[Model] Creating fresh {CONFIG.algorithm} model...")
    t0 = time.time()
    if AlgorithmRL.__name__ == "CLIPRewardedSAC":
        model = CLIPRewardedSAC(env=env, config=CONFIG)
    elif AlgorithmRL.__name__ == "CLIPRewardedPPO":
        model = CLIPRewardedPPO(env=env, config=CONFIG)
    else:
        model = AlgorithmRL('MultiInputPolicy', env, verbose=1, seed=CONFIG.seed,
                            tensorboard_log=args.output_dir, **CONFIG.algorithm_params)
    print(f"  [✓] Model created in {time.time()-t0:.1f}s")
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy params: {n_params:,}")
    remaining = args.total_timesteps

# ============================================================
# LOGGER
# ============================================================
new_logger = configure(args.output_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(args.output_dir, 'config.json'))


# ============================================================
# CALLBACKS
# ============================================================
class ReplayBufferCallback(BaseCallback):
    """Replay buffer'ı her N step'te bir kaydet (resume için kritik)."""
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_save >= self.save_freq:
            t0 = time.time()
            try:
                self.model.save_replay_buffer(self.save_path)
                if self.verbose > 0:
                    print(f"  [ReplayBuffer] Saved at step {self.num_timesteps:,} "
                          f"({time.time()-t0:.1f}s)")
                self.last_save = self.num_timesteps
            except Exception as e:
                print(f"  [ReplayBuffer] Save failed: {e}")
        return True


callbacks = [
    HParamCallback(CONFIG),
    TensorboardCallback(1),
    CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.output_dir,
        name_prefix="model",
    ),
    ReplayBufferCallback(
        save_freq=args.replay_buffer_freq,
        save_path=os.path.join(args.output_dir, "replay_buffer.pkl"),
        verbose=1,
    ),
]

# ============================================================
# TRAIN
# ============================================================
print(f"\n{'='*72}")
print(f"TRAINING START — seed={args.seed}")
print(f"{'='*72}")
print(f"  Goal:      {args.total_timesteps:,} steps")
print(f"  Remaining: {remaining:,} steps")
print(f"  Wall-time estimate: {remaining/15/3600:.1f}h @15fps simulator")

t_train_start = time.time()
try:
    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=False,           # resume continuity
    )
except KeyboardInterrupt:
    print("\n[!] Training interrupted by user (Ctrl+C)")
    print("    Saving current state...")
    model.save(os.path.join(args.output_dir, f"model_{model.num_timesteps}_steps.zip"))
    model.save_replay_buffer(os.path.join(args.output_dir, "replay_buffer.pkl"))
    print(f"    Saved at step {model.num_timesteps:,}")
    sys.exit(130)
except Exception as e:
    print(f"\n[✗] Training crashed: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    print("    Attempting emergency save...")
    try:
        model.save(os.path.join(args.output_dir, f"model_{model.num_timesteps}_emergency.zip"))
        model.save_replay_buffer(os.path.join(args.output_dir, "replay_buffer.pkl"))
        print(f"    Saved at step {model.num_timesteps:,}")
    except Exception as e2:
        print(f"    Emergency save also failed: {e2}")
    sys.exit(2)

train_dur = time.time() - t_train_start

# ============================================================
# FINAL SAVE
# ============================================================
final_model = os.path.join(args.output_dir, f"model_{args.total_timesteps}_steps.zip")
final_replay = os.path.join(args.output_dir, "replay_buffer.pkl")
model.save(final_model)
model.save_replay_buffer(final_replay)

print(f"\n{'='*72}")
print(f"TRAINING COMPLETE — seed={args.seed}")
print(f"{'='*72}")
print(f"  Wall time:       {train_dur/3600:.2f}h ({train_dur:.0f}s)")
print(f"  Steps trained:   {model.num_timesteps:,}")
print(f"  Final model:     {final_model}")
print(f"  Replay buffer:   {final_replay}")
print(f"  Effective FPS:   {remaining/train_dur:.1f} steps/s")

try:
    env.close()
except Exception:
    pass

sys.exit(0)
