"""
VLM-RL Minimal Eval — Tek Episode Reprodüksiyon Testi.

Hedef: HF checkpoint'i Town02'de tek bir episode boyunca koşturup
paper'daki metriklerle karşılaştırılabilir bir sonuç almak.

Bu, run_eval.py + eval.py'in basitleştirilmiş halidir:
- Tek town (Town02), tek episode
- Manual checkpoint load (pickle bypass)
- Per-step CSV log
- Episode sonu özet metrikleri

Eğer bu test başarılıysa, sonraki adımda 5 town × 10 route'a ölçeklendireceğiz.

ÖN KOŞUL:
    CARLA server çalışıyor olmalı.

KULLANIM:
    cd C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16\\PythonAPI\\VLM-RL-Stress\\repo
    conda activate carla_env
    python my_eval_minimal.py

ÇIKTI:
    - Konsola: per-step özet, episode sonu metrikleri
    - Dosyaya: ../results/eval_minimal_town02.csv
"""

import warnings
import os
import sys
import zipfile
import io
import time
import traceback
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CHECKPOINT_PATH = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\checkpoints\hf_vlmrl\sac_model.zip"
RESULTS_DIR = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\results"
CSV_PATH = os.path.join(RESULTS_DIR, "eval_minimal_town02.csv")

MAX_STEPS_PER_EPISODE = 1000  # paper limit, genelde route bittiğinde erken termine
TARGET_TOWN = "Town02"

print("=" * 60)
print("VLM-RL MINIMAL EVAL — TOWN02 SINGLE EPISODE")
print("=" * 60)

# ============================================================
# SETUP
# ============================================================
print("\n[Setup] Loading config, env, model...")

import config
CONFIG = config.set_config("vlm_rl")
CONFIG.algorithm_params.device = "cuda:0"

from clip.clip_rewarded_sac import CLIPRewardedSAC
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions

observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)

env = CarlaRouteEnv(
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
    activate_traffic_flow=True,   # eval'da trafik açık (paper protokolü)
    start_carla=False,
)

# Manual checkpoint load
model = CLIPRewardedSAC(env=env, config=CONFIG, inference_only=True)
with zipfile.ZipFile(CHECKPOINT_PATH) as z:
    with z.open('policy.pth') as f:
        state_dict = torch.load(io.BytesIO(f.read()), map_location='cuda:0', weights_only=False)
model.policy.load_state_dict(state_dict, strict=True)
print(f"  Model loaded ({sum(p.numel() for p in model.policy.parameters()):,} params)")

# ============================================================
# EPISODE
# ============================================================
print(f"\n[Episode] Running on {TARGET_TOWN} for max {MAX_STEPS_PER_EPISODE} steps...")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Per-step log
log_rows = []

# Reset
t_reset = time.time()
state = env.reset()
if isinstance(state, tuple):
    state = state[0]
print(f"  Reset done in {time.time()-t_reset:.1f}s")

# Episode başlangıç bilgileri
total_route_length = None
if hasattr(env, 'route_waypoints'):
    total_route_length = len(env.route_waypoints)
    print(f"  Route waypoints: {total_route_length}")

# Step döngüsü
episode_reward = 0.0
collisions = 0
last_collision_step = -1
collision_speeds = []
prev_collision_count = 0

t_episode = time.time()
for step in range(MAX_STEPS_PER_EPISODE):
    # Predict
    action, _ = model.predict(state, deterministic=True)

    # Step
    result = env.step(action)
    if len(result) == 4:
        next_state, reward, done, info = result
        terminated, truncated = done, False
    else:
        next_state, reward, terminated, truncated, info = result

    state = next_state
    episode_reward += reward

    # Vehicle stats
    speed_kmh = 0.0
    loc_x, loc_y = 0.0, 0.0
    center_dev = 0.0
    routes_completed = 0
    if hasattr(env, 'vehicle') and env.vehicle is not None:
        try:
            speed_kmh = env.vehicle.get_speed() * 3.6  # m/s -> km/h
        except Exception:
            pass
        try:
            loc = env.vehicle.get_location()
            loc_x, loc_y = loc.x, loc.y
        except Exception:
            pass
    if hasattr(env, 'distance_from_center'):
        center_dev = env.distance_from_center
    if hasattr(env, 'current_waypoint_index'):
        routes_completed = env.current_waypoint_index

    # Collision tracking
    if hasattr(env, 'collision_count'):
        if env.collision_count > prev_collision_count:
            collisions += 1
            collision_speeds.append(speed_kmh)
            last_collision_step = step
        prev_collision_count = env.collision_count

    # Log row
    log_rows.append({
        'step': step,
        'throttle': float(action[1]) if action[1] >= 0 else 0.0,
        'brake': float(-action[1]) if action[1] < 0 else 0.0,
        'steer': float(action[0]),
        'reward': float(reward),
        'speed_kmh': speed_kmh,
        'loc_x': loc_x,
        'loc_y': loc_y,
        'center_dev': center_dev,
        'routes_completed': routes_completed,
        'terminated': terminated,
        'truncated': truncated,
    })

    # Progress
    if (step + 1) % 50 == 0:
        avg_r = np.mean([r['reward'] for r in log_rows[-50:]])
        avg_s = np.mean([r['speed_kmh'] for r in log_rows[-50:]])
        print(f"  step {step+1:4d} | speed_avg(50)={avg_s:5.1f} | reward_avg(50)={avg_r:+.3f} | wp={routes_completed} | total_r={episode_reward:+.1f}")

    # Termination
    if terminated or truncated:
        print(f"\n  Episode ended at step {step+1}")
        if isinstance(info, dict):
            for k in ['terminal_reason', 'episode', 'mean_reward', 'avg_speed', 'avg_center_dev']:
                if k in info:
                    print(f"    info[{k}] = {info[k]}")
        break

episode_duration = time.time() - t_episode
final_step = log_rows[-1]['step'] + 1

# ============================================================
# OZET
# ============================================================
print("\n" + "=" * 60)
print("EPISODE ÖZETİ")
print("=" * 60)
print(f"  Town:                    {TARGET_TOWN}")
print(f"  Steps:                   {final_step}")
print(f"  Wall time:               {episode_duration:.1f}s")
print(f"  Effective FPS:           {final_step / episode_duration:.1f}")
print()

# Aggregate metrics
df = pd.DataFrame(log_rows)
print(f"  Total reward:            {episode_reward:+.2f}")
print(f"  Mean reward/step:        {df['reward'].mean():+.3f}")
print(f"  Mean speed:              {df['speed_kmh'].mean():.2f} km/h")
print(f"  Mean center deviation:   {df['center_dev'].mean():.3f} m")
print(f"  Final waypoint index:    {df['routes_completed'].iloc[-1]}")

if total_route_length:
    rc = df['routes_completed'].iloc[-1] / total_route_length
    print(f"  Route Completion (RC):   {rc*100:.1f}% ({df['routes_completed'].iloc[-1]}/{total_route_length})")

print(f"  Collisions:              {collisions}")
if collisions > 0:
    print(f"  Mean collision speed:    {np.mean(collision_speeds):.1f} km/h")

# Save CSV
df.to_csv(CSV_PATH, index=False)
print(f"\n  CSV saved: {CSV_PATH}")
print(f"  Total rows: {len(df)}")

# ============================================================
# CLEANUP
# ============================================================
print("\n[Cleanup] Closing env...")
try:
    env.close()
except Exception as e:
    print(f"  warning: {e}")

print("\n" + "=" * 60)
print("✅ MINIMAL EVAL TAMAMLANDI")
print("=" * 60)
print("\nPaper Tablo 2 (vlm_rl, Town02) bekledikleri:")
print("  CR (Collision Rate):     ~0.02")
print("  RC (Route Completion):   ~0.97")
print("  SR (Success Rate):       ~0.93")
print("\nBu tek bir episode olduğundan paper sayılarıyla doğrudan karşılaştırılamaz")
print("ama RC=%80+ ve collision=0 görürsek pipeline çalışıyor demektir.")
