"""
VLM-RL Eval Video Recorder.

Bir başarılı (Town02) ve bir başarısız (Town05) episode kaydeder.
Çıktı: .avi formatında 2 video (~30s/episode), pygame display'den alınmış RGB frame'ler.

KULLANIM:
    # CARLA çalışıyor olmalı
    python record_videos.py

    # Custom seçim
    python record_videos.py --success-town Town02 --success-route 0 \\
                            --fail-town Town05 --fail-route 3 \\
                            --seed 100

ÇIKTI:
    results/videos/
        success_Town02_route0_seed100.avi
        fail_Town05_route3_seed100.avi

NOTLAR:
    - activate_render=True (pygame window açılır, ama RenderOffScreen CARLA için)
    - Tek route koşar, eval_routes cycle'ından specific tuple'a sıçrar
    - reward overlay: VideoRecorder.add_frame_with_reward()
    - Her episode max 5000 step (yeterli, route auto-terminate)
"""

import sys
import argparse
import os
import zipfile
import io
import time
import warnings
import itertools
import random

import numpy as np
import pandas as pd
import torch
import pygame

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Record VLM-RL eval videos")
parser.add_argument("--success-town", default="Town02", choices=['Town01','Town02','Town03','Town04','Town05'])
parser.add_argument("--success-route", type=int, default=0,
                    help="Index of route in eval_routes (0-9). Town02 route 0 = (48,21).")
parser.add_argument("--fail-town", default="Town05", choices=['Town01','Town02','Town03','Town04','Town05'])
parser.add_argument("--fail-route", type=int, default=3,
                    help="Default Town05 route 3 — deterministic collision in 3-seed run.")
parser.add_argument("--seed", type=int, default=100, help="Inference seed")
parser.add_argument("--max-steps", type=int, default=5000)
parser.add_argument("--output-dir",
                    default=r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\results\videos")
parser.add_argument("--checkpoint",
                    default=r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\checkpoints\hf_vlmrl\sac_model.zip")
args = parser.parse_args()

print("=" * 72)
print("VLM-RL VIDEO RECORDER")
print("=" * 72)
print(f"  Success: {args.success_town} route {args.success_route}, seed={args.seed}")
print(f"  Fail:    {args.fail_town} route {args.fail_route}, seed={args.seed}")
print(f"  Output:  {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)

# Reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ============================================================
# CARLA HEALTH CHECK
# ============================================================
print("\n[Pre-flight] CARLA bağlantı kontrolü...")
try:
    import carla
    _client = carla.Client('localhost', 2000)
    _client.set_timeout(10.0)
    _ver = _client.get_server_version()
    print(f"  [✓] CARLA: version={_ver}")
    del _client
except Exception as e:
    print(f"  [✗] CARLA yanıt vermiyor: {e}")
    sys.exit(1)

# ============================================================
# IMPORTS
# ============================================================
print("\n[Setup] Loading modules...")
import config
CONFIG = config.set_config("vlm_rl")
CONFIG.algorithm_params.device = "cuda:0"

from clip.clip_rewarded_sac import CLIPRewardedSAC
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions
from utils import VideoRecorder
import carla_env.envs.carla_route_env as cre

EVAL_ROUTES_TUPLES = [
    (48, 21), (0, 72), (28, 83), (61, 39), (27, 14),
    (6, 67), (61, 49), (37, 64), (33, 80), (12, 30),
]

observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
print(f"  [✓] CONFIG loaded")


# ============================================================
# HELPERS
# ============================================================
def set_eval_routes_to(start_idx):
    """eval_routes cycle'ı belirli bir route'tan başlatacak şekilde reset et."""
    rotated = EVAL_ROUTES_TUPLES[start_idx:] + EVAL_ROUTES_TUPLES[:start_idx]
    cre.eval_routes = itertools.cycle(rotated)


def create_env(town):
    """Pygame display açık env (video için)."""
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
        activate_spectator=True,        # video için: camera + BEV display'e blit eder
        activate_render=True,           # video için: pygame display + HUD
        activate_bev=CONFIG.use_rgb_bev,
        activate_seg_bev=CONFIG.use_seg_bev,
        activate_traffic_flow=True,
        start_carla=False,
        eval=True,
        town=town,
    )


def load_model(env):
    model = CLIPRewardedSAC(env=env, config=CONFIG, inference_only=True)
    with zipfile.ZipFile(args.checkpoint) as z:
        with z.open('policy.pth') as f:
            sd = torch.load(io.BytesIO(f.read()),
                            map_location='cuda:0',
                            weights_only=False)
    model.policy.load_state_dict(sd, strict=True)
    return model


def record_episode(town, route_idx, output_path, label):
    """Tek episode'u kaydet ve metriklerini döndür."""
    print(f"\n{'='*72}")
    print(f"[{label}] Recording: {town} route {route_idx} → {output_path}")
    print(f"{'='*72}")

    # eval_routes'ı bu route'tan başlatacak şekilde reset
    set_eval_routes_to(route_idx)

    # Fresh env (her episode için)
    print(f"  Creating env (town={town})...")
    t0 = time.time()
    env = create_env(town)
    print(f"  [✓] Env in {time.time()-t0:.1f}s")

    # Traffic manager seed
    if hasattr(env, 'traffic_manager') and env.traffic_manager is not None:
        try:
            env.traffic_manager.set_random_device_seed(args.seed)
            print(f"  [✓] traffic_manager seed = {args.seed}")
        except Exception as e:
            print(f"  [!] traffic_manager seed failed: {e}")

    # Model
    model = load_model(env)
    print(f"  [✓] Model loaded")

    # Reset env (will pick first eval_routes tuple, which is now the desired route)
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    # İlk frame'i render et + dump al
    env.render(mode="human")     # camera+BEV display'e blit + flip
    rendered = env.render(mode="rgb_array")
    print(f"  Frame size: {rendered.shape}")
    video = VideoRecorder(output_path, frame_size=rendered.shape, fps=int(env.fps))

    # Episode loop
    total_reward = 0.0
    speeds = []
    last_info = {}
    natural_route_end = False

    print(f"  Recording {args.max_steps} max steps...")
    t_ep = time.time()
    for step in range(args.max_steps):
        action, _ = model.predict(state, deterministic=True)
        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
            terminated, truncated = done, False
        else:
            next_state, reward, terminated, truncated, info = result
        state = next_state
        last_info = info if isinstance(info, dict) else {}
        total_reward += float(reward)

        if hasattr(env, 'vehicle') and env.vehicle is not None:
            try:
                speeds.append(env.vehicle.get_speed())
            except Exception:
                pass

        # ÖNCE human mode render (camera + BEV display'e blit + flip)
        # SONRA rgb_array dump (display surface → np.array)
        try:
            env.render(mode="human")
            rendered = env.render(mode="rgb_array")
            video.add_frame_with_reward(rendered, total_reward)
        except Exception as e:
            print(f"  [!] Render failed at step {step}: {e}")
            break

        # Paper'ın eval termination hack'i
        wp_idx = getattr(env, 'current_waypoint_index', -1)
        if step >= 150 and wp_idx == 0:
            natural_route_end = True
            break

        if terminated or truncated:
            break

        # Progress
        if (step + 1) % 200 == 0:
            mean_s = np.mean(speeds[-200:]) if speeds else 0
            print(f"    step {step+1:>4} | speed={mean_s:5.1f}km/h | reward={total_reward:+.1f}")

    ep_dur = time.time() - t_ep
    n_steps = step + 1

    # Cleanup
    video.release()
    try:
        env.close()
    except Exception:
        pass
    pygame.quit()

    # Metrik özeti
    rc = float(last_info.get('routes_completed', 0.0))
    td = float(last_info.get('total_distance', 0.0))
    collision = bool(last_info.get('collision_state', False))
    cs = float(last_info.get('collision_speed', 0.0)) if collision else 0.0
    success = (not collision) and (rc >= 1.0)

    print(f"\n  >> {label} Sonuç:")
    print(f"     Steps:           {n_steps}")
    print(f"     Wall time:       {ep_dur:.1f}s")
    print(f"     Mean speed:      {np.mean(speeds):.2f} km/h" if speeds else "     Mean speed: N/A")
    print(f"     RC:              {rc:.3f}")
    print(f"     TD:              {td:.1f} m")
    print(f"     Collision:       {collision}")
    if collision:
        print(f"     Collision speed: {cs:.2f} km/h")
    print(f"     Success:         {success}")
    print(f"     Video saved:     {output_path}")

    return {
        'label': label,
        'town': town,
        'route_idx': route_idx,
        'steps': n_steps,
        'rc': rc,
        'td_m': td,
        'collision': collision,
        'collision_speed_kmh': cs,
        'success': success,
        'mean_speed_kmh': float(np.mean(speeds)) if speeds else 0.0,
        'video_path': output_path,
    }


# ============================================================
# RUN BOTH EPISODES
# ============================================================
results = []

# 1) Başarılı episode
success_filename = f"success_{args.success_town}_route{args.success_route}_seed{args.seed}.avi"
success_path = os.path.join(args.output_dir, success_filename)
try:
    r1 = record_episode(args.success_town, args.success_route, success_path, "SUCCESS")
    results.append(r1)
except Exception as e:
    print(f"  [✗] SUCCESS episode failed: {e}")
    import traceback; traceback.print_exc()

# CARLA stabilizasyon arası
print("\n[Pause] 5s waiting for CARLA stability...")
time.sleep(5)

# 2) Başarısız episode
fail_filename = f"fail_{args.fail_town}_route{args.fail_route}_seed{args.seed}.avi"
fail_path = os.path.join(args.output_dir, fail_filename)
try:
    r2 = record_episode(args.fail_town, args.fail_route, fail_path, "FAIL")
    results.append(r2)
except Exception as e:
    print(f"  [✗] FAIL episode failed: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 72)
print("VIDEO RECORDING COMPLETE")
print("=" * 72)
for r in results:
    print(f"\n  [{r['label']}] {r['town']} route {r['route_idx']}:")
    print(f"     File:     {r['video_path']}")
    print(f"     Outcome:  {'SUCCESS' if r['success'] else 'FAIL'} "
          f"(RC={r['rc']:.3f}, collision={r['collision']})")
    if r['collision']:
        print(f"     CS:       {r['collision_speed_kmh']:.2f} km/h")

print(f"\n  Videos: {args.output_dir}")
