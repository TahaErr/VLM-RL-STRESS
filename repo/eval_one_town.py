"""
VLM-RL Single Town Eval — Worker Script.

run_full_eval.py wrapper'ı tarafından subprocess olarak çağrılır. Tek bir
town için 10 episode koşturur, mevcut CSV'ye episode satırlarını ekler.

CARLA server'ın bu script çağrılmadan ÖNCE çalışıyor olması beklenir
(wrapper bu garantiyi verir).

KULLANIM:
    python eval_one_town.py Town03
    python eval_one_town.py Town04 --episodes 10 --max-steps 5000

ÇIKTI:
    - results/eval_full_episodes.csv (append mode)
    - Konsola: per-episode progress
    - Exit code: 0 = success, 1 = setup failed
"""

import sys
import argparse
import os
import zipfile
import io
import time
import warnings
import itertools
import traceback
import random

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="VLM-RL eval for one town")
parser.add_argument("town", choices=['Town01', 'Town02', 'Town03', 'Town04', 'Town05'])
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max-steps", type=int, default=5000)
parser.add_argument("--seed", type=int, default=101,
                    help="Inference seed (random/numpy/torch + CARLA traffic_manager)")
parser.add_argument("--results-dir",
                    default=r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\results")
parser.add_argument("--checkpoint",
                    default=r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\checkpoints\hf_vlmrl\sac_model.zip")
args = parser.parse_args()

TOWN = args.town
EPISODES_PER_TOWN = args.episodes
MAX_STEPS_PER_EPISODE = args.max_steps
SEED = args.seed
RESULTS_DIR = args.results_dir
EPISODES_CSV = os.path.join(RESULTS_DIR, "eval_full_episodes.csv")
CHECKPOINT_PATH = args.checkpoint

ROUTE_END_DETECT_AFTER = 150

# ============================================================
# SEED SETUP — global random sources
# ============================================================
# Bu repo'nun env kodu Python random.choice ve np.random.choice kullanıyor
# (NPC spawn, vehicle blueprint seçimi). Bu seed'ler global state'i etkiler.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 72)
print(f"EVAL ONE TOWN — {TOWN}, seed={SEED}")
print("=" * 72)
print(f"  Episodes:  {EPISODES_PER_TOWN}")
print(f"  Max steps: {MAX_STEPS_PER_EPISODE}")
print(f"  Seed:      {SEED}")

# ============================================================
# CARLA HEALTH CHECK
# ============================================================
print("\n[Pre-flight] CARLA bağlantı kontrolü...")
try:
    import carla as _carla_test
    _client = _carla_test.Client('localhost', 2000)
    _client.set_timeout(10.0)
    _ver = _client.get_server_version()
    _map = _client.get_world().get_map().name
    print(f"  [✓] version={_ver}, map={_map}")
    del _client
except Exception as e:
    print(f"  [✗] CARLA yanıt vermiyor: {e}")
    sys.exit(1)

# ============================================================
# IMPORTS
# ============================================================
print("\n[Setup] Loading config and modules...")
import config
CONFIG = config.set_config("vlm_rl")
CONFIG.algorithm_params.device = "cuda:0"

from clip.clip_rewarded_sac import CLIPRewardedSAC
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions
import carla_env.envs.carla_route_env as cre

EVAL_ROUTES_TUPLES = [
    (48, 21), (0, 72), (28, 83), (61, 39), (27, 14),
    (6, 67), (61, 49), (37, 64), (33, 80), (12, 30),
]

# Reset eval_routes — fresh subprocess'te zaten 0'dan başlıyor ama garanti olsun
cre.eval_routes = itertools.cycle(EVAL_ROUTES_TUPLES)

observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
print(f"  [✓] CONFIG loaded")


# ============================================================
# HELPERS
# ============================================================
def create_env(town):
    return CarlaRouteEnv(
        obs_res=CONFIG.obs_res,
        host="localhost",
        port=2000,
        reward_fn=reward_functions[CONFIG.reward_fn],
        observation_space=observation_space,
        encode_state_fn=encode_state_fn,
        fps=15,
        action_smoothing=CONFIG.action_smoothing,
        action_space_type='continuous',
        activate_spectator=False,
        activate_render=False,
        activate_bev=CONFIG.use_rgb_bev,
        activate_seg_bev=CONFIG.use_seg_bev,
        activate_traffic_flow=True,
        start_carla=False,
        eval=True,
        town=town,
    )


def load_model(env):
    model = CLIPRewardedSAC(env=env, config=CONFIG, inference_only=True)
    with zipfile.ZipFile(CHECKPOINT_PATH) as z:
        with z.open('policy.pth') as f:
            sd = torch.load(io.BytesIO(f.read()),
                            map_location='cuda:0',
                            weights_only=False)
    model.policy.load_state_dict(sd, strict=True)
    return model


def run_episode(env, model, town, ep_idx):
    t_reset = time.time()
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    reset_dur = time.time() - t_reset

    speeds, devs, rewards = [], [], []
    last_info = {}
    truncated = False
    terminated = False
    natural_route_end = False

    t_ep = time.time()
    step = 0
    for step in range(MAX_STEPS_PER_EPISODE):
        action, _ = model.predict(state, deterministic=True)
        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
            terminated, truncated = done, False
        else:
            next_state, reward, terminated, truncated, info = result

        state = next_state
        last_info = info if isinstance(info, dict) else {}
        rewards.append(float(reward))

        if hasattr(env, 'vehicle') and env.vehicle is not None:
            try:
                speeds.append(env.vehicle.get_speed())
            except Exception:
                pass
        if hasattr(env, 'distance_from_center'):
            devs.append(env.distance_from_center)

        wp_idx = getattr(env, 'current_waypoint_index', -1)
        if step >= ROUTE_END_DETECT_AFTER and wp_idx == 0:
            natural_route_end = True
            break

        if terminated or truncated:
            break

    ep_dur = time.time() - t_ep
    n_steps = step + 1

    rc = float(last_info.get('routes_completed', 0.0))
    td = float(last_info.get('total_distance', 0.0))
    collision_state = bool(last_info.get('collision_state', False))

    if collision_state:
        terminal_reason = 'collision'
    elif natural_route_end:
        terminal_reason = 'route_loop'
    elif n_steps >= MAX_STEPS_PER_EPISODE:
        terminal_reason = 'max_steps'
    elif terminated:
        terminal_reason = 'terminated_other'
    elif truncated:
        terminal_reason = 'truncated'
    else:
        terminal_reason = 'unknown'

    return {
        'town': town,
        'seed': SEED,
        'ep_idx': ep_idx,
        'route_idx': ep_idx % 10,  # paper'ın 10 predefined route'undan hangisi (0-9)
        'route_repetition': ep_idx // 10,  # bu route'un kaçıncı tekrarı (0-2 for N=30)
        'steps': n_steps,
        'wall_time_s': round(ep_dur, 2),
        'reset_time_s': round(reset_dur, 2),
        'fps_eff': round(n_steps / ep_dur, 1) if ep_dur > 0 else 0.0,
        'mean_speed_kmh': round(float(np.mean(speeds)), 3) if speeds else 0.0,
        'mean_center_dev_m': round(float(np.mean(devs)), 4) if devs else 0.0,
        'mean_reward': round(float(np.mean(rewards)), 4) if rewards else 0.0,
        'total_reward': round(float(np.sum(rewards)), 2) if rewards else 0.0,
        'routes_completed': round(rc, 4),
        'total_distance_m': round(td, 2),
        'collision_state': collision_state,
        'collision_speed_kmh': round(float(last_info.get('collision_speed', 0.0)), 3) if collision_state else 0.0,
        'collision_interval': int(last_info.get('collision_interval', 0)) if collision_state else 0,
        'CPS': round(float(last_info.get('CPS', 0.0)), 6) if collision_state else 0.0,
        'CPM': round(float(last_info.get('CPM', 0.0)), 4) if collision_state else 0.0,
        'route_complete_flag': bool(rc >= 1.0),
        'success': bool((not collision_state) and (rc >= 1.0)),
        'terminal_reason': terminal_reason,
    }


def append_episode_to_csv(metrics, csv_path):
    """Tek episode'u CSV'ye ekle (resume-safe). Mevcut town+seed+ep_idx satırını
    silip yenisini yazar — re-run durumunda eski veriyi üzerine yazsın."""
    new_row = pd.DataFrame([metrics])

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        # Eski CSV'lerde 'seed' kolonu olmayabilir — backward compat
        if 'seed' not in existing.columns:
            existing['seed'] = -1  # sentinel for legacy rows
        # Bu town+seed+ep_idx için mevcut satır varsa sil
        mask = ((existing['town'] == metrics['town'])
                & (existing['seed'] == metrics['seed'])
                & (existing['ep_idx'] == metrics['ep_idx']))
        existing = existing[~mask]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    # Sıralama: önce town, sonra seed, sonra ep_idx
    combined = combined.sort_values(['town', 'seed', 'ep_idx']).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)


# ============================================================
# RESUME LOGIC
# ============================================================
existing_count = 0
if os.path.exists(EPISODES_CSV):
    try:
        existing_df = pd.read_csv(EPISODES_CSV)
        if 'seed' not in existing_df.columns:
            existing_df['seed'] = -1
        # Sadece bu (town, seed) kombinasyonu için mevcut episode'ları say
        existing_count = len(existing_df[(existing_df['town'] == TOWN)
                                         & (existing_df['seed'] == SEED)])
    except Exception as e:
        print(f"  [!] CSV okuma hatası, baştan başlıyor: {e}")
        existing_count = 0

if existing_count >= EPISODES_PER_TOWN:
    print(f"\n[SKIP] {TOWN} seed={SEED} zaten {existing_count} episode'a sahip (>= {EPISODES_PER_TOWN}).")
    sys.exit(0)

start_ep = existing_count
print(f"\n[Resume] {TOWN} seed={SEED}: {existing_count} mevcut, ep {start_ep}-{EPISODES_PER_TOWN-1} koşturulacak")

# ============================================================
# SETUP ENV + MODEL
# ============================================================
print(f"\n[Setup] Creating env (town={TOWN}, eval=True, traffic_flow=True)...")
try:
    t0 = time.time()
    env = create_env(TOWN)
    print(f"  [✓] Env created in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  [✗] Env creation FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# CARLA traffic_manager'a seed ver — NPC autopilot davranışları deterministic olsun
if hasattr(env, 'traffic_manager') and env.traffic_manager is not None:
    try:
        env.traffic_manager.set_random_device_seed(SEED)
        print(f"  [✓] traffic_manager.set_random_device_seed({SEED})")
    except Exception as e:
        print(f"  [!] traffic_manager seed set failed: {e}")

try:
    model = load_model(env)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  [✓] Model loaded ({n_params:,} params)")
except Exception as e:
    print(f"  [✗] Model load FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    try:
        env.close()
    except Exception:
        pass
    sys.exit(1)

# ============================================================
# EPISODES
# ============================================================
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"\n[Episodes] Running {EPISODES_PER_TOWN - start_ep} episodes for {TOWN}...")
print(f"  {'ep':>3} {'steps':>5} {'speed':>7} {'RC':>6} {'TD':>8} {'CS':>3} {'SR':>3} {'time':>6} {'reason':<15}")

failures = 0
for ep_idx in range(start_ep, EPISODES_PER_TOWN):
    try:
        metrics = run_episode(env, model, TOWN, ep_idx)

        print(f"  {ep_idx+1:>3} {metrics['steps']:>5} "
              f"{metrics['mean_speed_kmh']:>5.1f}km "
              f"{metrics['routes_completed']:>6.2f} "
              f"{metrics['total_distance_m']:>7.1f}m "
              f"{int(metrics['collision_state']):>3} "
              f"{int(metrics['success']):>3} "
              f"{metrics['wall_time_s']:>5.1f}s "
              f"{metrics['terminal_reason']:<15}")

        # Per-episode partial save (mid-run crash safety)
        append_episode_to_csv(metrics, EPISODES_CSV)
    except Exception as e:
        print(f"  {ep_idx+1:>3} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        failures += 1
        # CARLA çökmüşse devam etmenin anlamı yok
        if "time-out" in str(e) or "Connection" in str(e):
            print(f"  [!] CARLA bağlantı sorunu, kalan episode'lar atlanıyor")
            break

# Cleanup
try:
    env.close()
except Exception as e:
    print(f"\n[Cleanup] env.close() warning: {e}")

print(f"\n[Done] {TOWN}: {EPISODES_PER_TOWN - start_ep - failures}/{EPISODES_PER_TOWN - start_ep} episode başarılı")

if failures == EPISODES_PER_TOWN - start_ep:
    print(f"  [✗] Tüm episode'lar fail oldu, exit 1")
    sys.exit(1)

sys.exit(0)
