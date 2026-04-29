"""
VLM-RL — Tek Town'un Tüm 10 Route'unun Video Kaydı.

Bir town için eval_routes cycle'ındaki 10 predefined route'u sırayla koşturur,
hepsini tek bir .avi dosyasında birleştirir. Route geçişlerinde 1.5 saniyelik
"Route X / 10 — SUCCESS|FAIL" overlay frame'leri eklenir.

KULLANIM:
    python record_town_routes.py --town Town02
    python record_town_routes.py --town Town05 --frame-skip 2
    python record_town_routes.py --town Town04 --max-steps 8000

ÇIKTI:
    results/videos/<Town>_all_routes_seed<N>.avi

ÖN KOŞUL:
    - CARLA çalışıyor (port 2000), -RenderOffScreen YOK
    - record_videos.py'deki düzeltmeler uygulanmış (activate_spectator=True,
      env.render(mode="human") çağrısı önce, sonra rgb_array dump)
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
import torch
import pygame

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Record all 10 routes of a town in one video")
parser.add_argument("--town", required=True,
                    choices=['Town01', 'Town02', 'Town03', 'Town04', 'Town05'],
                    help="Hangi town için kayıt yapılacak (zorunlu)")
parser.add_argument("--seed", type=int, default=100, help="Inference seed (default: 100)")
parser.add_argument("--max-steps", type=int, default=15000,
                    help="Episode başına max step (default: 15000, full run ile aynı)")
parser.add_argument("--frame-skip", type=int, default=1,
                    help="Her N step'te bir kayıt yap (1=her step, 2=yarı, 3=üçte bir)")
parser.add_argument("--transition-seconds", type=float, default=1.5,
                    help="Route arası geçiş overlay süresi (default: 1.5s)")
parser.add_argument("--output-dir",
                    default=r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\results\videos")
parser.add_argument("--checkpoint",
                    default=r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\checkpoints\hf_vlmrl\sac_model.zip")
args = parser.parse_args()

TOWN = args.town
SEED = args.seed

print("=" * 72)
print(f"VLM-RL — Tek Video: {TOWN}, 10 Route Ardışık")
print("=" * 72)
print(f"  Town:                 {TOWN}")
print(f"  Seed:                 {SEED}")
print(f"  Max steps/episode:    {args.max_steps}")
print(f"  Frame skip:           {args.frame_skip}")
print(f"  Transition overlay:   {args.transition_seconds}s")
print(f"  Output dir:           {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
def reset_eval_routes_iterator():
    """Module-level eval_routes cycle'ını sıfırla → ilk route (48,21)'den başlar."""
    cre.eval_routes = itertools.cycle(EVAL_ROUTES_TUPLES)


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
        activate_spectator=True,        # camera + BEV display'e blit
        activate_render=True,           # pygame window + HUD
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


def make_transition_frame(width, height, route_idx, total_routes,
                           prev_outcome=None, prev_metrics=None):
    """Route arası overlay frame'i: koyu arka plan + büyük route etiketi."""
    import cv2
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Lacivert arka plan (BGR formatında — VideoRecorder content'i öyle bekliyor)
    frame[:, :] = (40, 25, 15)

    # Üst metin: "ROUTE X / 10"
    title = f"ROUTE {route_idx + 1} / {total_routes}"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
    cv2.putText(frame, title,
                ((width - tw) // 2, height // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4, cv2.LINE_AA)

    # Önceki episode'un sonucu (eğer varsa)
    if prev_outcome is not None:
        color = (50, 200, 50) if prev_outcome else (50, 50, 220)  # yeşil/kırmızı (BGR)
        text = f"PREV: {'SUCCESS' if prev_outcome else 'FAIL'}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(frame, text,
                    ((width - tw) // 2, height // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        if prev_metrics:
            details = (f"RC={prev_metrics['rc']:.2f} | "
                       f"steps={prev_metrics['steps']} | "
                       f"speed={prev_metrics['mean_speed']:.1f} km/h")
            (tw, th), _ = cv2.getTextSize(details, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, details,
                        ((width - tw) // 2, height // 2 + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    return frame


def run_one_route(env, model, video, route_idx, max_steps, frame_skip):
    """Tek route'u koştur, frame'leri video'ya ekle, metrikleri döndür."""
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    # İlk render — display surface'in güncellendiğinden emin ol
    env.render(mode="human")

    total_reward = 0.0
    speeds = []
    last_info = {}
    natural_route_end = False
    terminated = False
    truncated = False

    t_ep = time.time()
    step = 0
    for step in range(max_steps):
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

        # Render + dump (frame_skip ile)
        if step % frame_skip == 0:
            try:
                env.render(mode="human")    # camera + BEV blit + flip
                rendered = env.render(mode="rgb_array")
                video.add_frame_with_reward(rendered, total_reward)
            except Exception as e:
                print(f"    [!] Render fail at step {step}: {e}")
                break

        # Paper'ın eval termination hack'i
        wp_idx = getattr(env, 'current_waypoint_index', -1)
        if step >= 150 and wp_idx == 0:
            natural_route_end = True
            break

        if terminated or truncated:
            break

        # Progress
        if (step + 1) % 500 == 0:
            mean_s = np.mean(speeds[-500:]) if speeds else 0
            wp = getattr(env, 'current_waypoint_index', -1)
            print(f"      step {step+1:>5} | speed={mean_s:5.1f}km/h "
                  f"| reward={total_reward:+.1f} | wp={wp}")

    ep_dur = time.time() - t_ep
    n_steps = step + 1

    rc = float(last_info.get('routes_completed', 0.0))
    td = float(last_info.get('total_distance', 0.0))
    collision = bool(last_info.get('collision_state', False))
    cs = float(last_info.get('collision_speed', 0.0)) if collision else 0.0
    success = (not collision) and (rc >= 1.0)

    return {
        'route_idx': route_idx,
        'steps': n_steps,
        'wall_time_s': ep_dur,
        'mean_speed': float(np.mean(speeds)) if speeds else 0.0,
        'rc': rc,
        'td_m': td,
        'collision': collision,
        'collision_speed_kmh': cs,
        'success': success,
        'natural_route_end': natural_route_end,
    }


# ============================================================
# MAIN
# ============================================================
output_filename = f"{TOWN}_all_routes_seed{SEED}.avi"
output_path = os.path.join(args.output_dir, output_filename)
print(f"\n[Output] {output_path}")

# eval_routes iterator reset (route 0'dan başlasın)
reset_eval_routes_iterator()
print(f"[Setup] eval_routes cycle reset → starts at route 0 (spawn 48→21)")

# Env yarat (sadece bir kez, tüm 10 route aynı env'de)
print(f"\n[Setup] Creating env (town={TOWN})...")
t0 = time.time()
env = create_env(TOWN)
print(f"  [✓] Env created in {time.time()-t0:.1f}s")

# Traffic manager seed
if hasattr(env, 'traffic_manager') and env.traffic_manager is not None:
    try:
        env.traffic_manager.set_random_device_seed(SEED)
        print(f"  [✓] traffic_manager seed = {SEED}")
    except Exception as e:
        print(f"  [!] traffic_manager seed failed: {e}")

# Model yükle
print(f"\n[Setup] Loading model...")
model = load_model(env)
print(f"  [✓] Model loaded")

# İlk reset → boyut için
state = env.reset()
if isinstance(state, tuple):
    state = state[0]
env.render(mode="human")
sample_frame = env.render(mode="rgb_array")
H, W = sample_frame.shape[:2]
fps = int(env.fps)
print(f"\n[Video] Frame: {W}x{H} @ {fps} fps")

# Video recorder
video = VideoRecorder(output_path, frame_size=sample_frame.shape, fps=fps)

# Tüm route'ları koştur
total_routes = len(EVAL_ROUTES_TUPLES)
all_metrics = []
overall_t0 = time.time()

# Eval_routes module-level cycle olduğu için reset() her çağrıda next() yapar.
# Yani ilk reset = route 0 (48,21), ikinci = route 1 (0,72), ...
# Şimdi sıfırladık, ilk reset'i yukarıda yaptık → route 0 koşuyor olmalı.

# Ancak yukarıdaki "sample_frame için reset" eval_routes'tan ilk tuple'ı çekti.
# eval_routes'ı tekrar reset edelim, sonra 10 route koşalım.
reset_eval_routes_iterator()

# Frame'leri sayalım (transition için)
transition_frames_count = max(1, int(args.transition_seconds * fps))
print(f"  Transition: {transition_frames_count} frame ({args.transition_seconds}s)")

prev_metrics = None
for route_idx in range(total_routes):
    print(f"\n{'='*72}")
    print(f"[Route {route_idx + 1}/{total_routes}]")
    print(f"{'='*72}")

    # Geçiş overlay frame'i (önceki sonucu da gösterir)
    print(f"  Adding transition overlay ({transition_frames_count} frames)...")
    transition_frame = make_transition_frame(
        W, H, route_idx, total_routes,
        prev_outcome=(prev_metrics['success'] if prev_metrics else None),
        prev_metrics=prev_metrics,
    )
    for _ in range(transition_frames_count):
        video.video_writer.write(transition_frame)  # zaten BGR

    # Route koşumu
    print(f"  Running route {route_idx} (max {args.max_steps} steps)...")
    metrics = run_one_route(env, model, video, route_idx,
                            args.max_steps, args.frame_skip)
    all_metrics.append(metrics)
    prev_metrics = metrics

    # Konsol özeti
    outcome = "SUCCESS" if metrics['success'] else "FAIL"
    print(f"\n  Route {route_idx + 1} sonuç: {outcome}")
    print(f"     steps:        {metrics['steps']}")
    print(f"     wall time:    {metrics['wall_time_s']:.1f}s")
    print(f"     mean speed:   {metrics['mean_speed']:.2f} km/h")
    print(f"     RC:           {metrics['rc']:.3f}")
    print(f"     TD:           {metrics['td_m']:.1f} m")
    print(f"     collision:    {metrics['collision']}")
    if metrics['collision']:
        print(f"     CS:           {metrics['collision_speed_kmh']:.2f} km/h")

# Son geçiş frame'i (10. route'un sonucu için)
print(f"\n[Final] Adding closing overlay...")
final_frame = make_transition_frame(
    W, H, total_routes, total_routes,
    prev_outcome=prev_metrics['success'],
    prev_metrics=prev_metrics,
)
# Final overlay biraz daha uzun (3s)
for _ in range(int(3.0 * fps)):
    video.video_writer.write(final_frame)

# Cleanup
video.release()
try:
    env.close()
except Exception:
    pass
pygame.quit()

# ============================================================
# FINAL SUMMARY
# ============================================================
total_dur = time.time() - overall_t0
n_success = sum(1 for m in all_metrics if m['success'])
n_collision = sum(1 for m in all_metrics if m['collision'])
total_steps = sum(m['steps'] for m in all_metrics)

print("\n" + "=" * 72)
print(f"VIDEO COMPLETE — {TOWN}")
print("=" * 72)
print(f"  Output:          {output_path}")
print(f"  Total wall time: {total_dur/60:.1f} min")
print(f"  Total steps:     {total_steps:,}")
print(f"  SR:              {n_success}/10 = {n_success/10:.1f}")
print(f"  CR:              {n_collision}/10 = {n_collision/10:.1f}")
print(f"\n  Per-route breakdown:")
print(f"  {'route':>5} {'steps':>6} {'speed':>7} {'RC':>5} {'TD':>7} {'collision':>10} {'outcome':>8}")
for m in all_metrics:
    out = 'SUCCESS' if m['success'] else 'FAIL'
    coll = f"yes ({m['collision_speed_kmh']:.1f}km/h)" if m['collision'] else 'no'
    print(f"  {m['route_idx']:>5} {m['steps']:>6} "
          f"{m['mean_speed']:>5.1f}km {m['rc']:>5.2f} {m['td_m']:>6.1f}m "
          f"{coll:>10} {out:>8}")
