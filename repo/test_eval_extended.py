"""
HF Checkpoint Genişletilmiş Eval Testi

Hedef: Policy gerçekten araba sürüyor mu, yoksa donmuş mu?

Önceki test'te action 5 step boyunca sabitti. Bu durum iki şeyden olabilir:
  1. Policy aslında düzgün, sadece aynı observation'ları görüyor (araba yerinde)
  2. Policy ölü/saturated, action sabit

Bu test 30 step koşturup vehicle hız + position'a bakıyor:
  - Eğer hız yükseliyorsa ve observation değişiyorsa → policy çalışıyor
  - Eğer hız sıfır kalıyor ya da action sürekli aynı ise → sorun var

KULLANIM:
    cd C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16\\PythonAPI\\VLM-RL-Stress\\repo
    python test_eval_extended.py
"""

import warnings
import os
import sys
import zipfile
import io
import traceback
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CHECKPOINT_PATH = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\checkpoints\hf_vlmrl\sac_model.zip"

print("=" * 60)
print("EXTENDED EVAL TEST - Policy Genuinely Driving?")
print("=" * 60)

# ============================================================
# SETUP
# ============================================================
print("\n[Setup] Loading env + model...")
import config
CONFIG = config.set_config("vlm_rl")
CONFIG.algorithm_params.device = "cuda:0"

from clip.clip_rewarded_sac import CLIPRewardedSAC
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions

observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
env = CarlaRouteEnv(
    obs_res=CONFIG.obs_res, host="localhost", port=2000,
    reward_fn=reward_functions[CONFIG.reward_fn],
    observation_space=observation_space,
    encode_state_fn=encode_state_fn,
    fps=15, action_smoothing=CONFIG.action_smoothing,
    action_space_type='continuous',
    activate_spectator=False, activate_render=False,
    activate_bev=CONFIG.use_rgb_bev,
    activate_seg_bev=CONFIG.use_seg_bev,
    activate_traffic_flow=False, start_carla=False,
)

model = CLIPRewardedSAC(env=env, config=CONFIG, inference_only=True)
with zipfile.ZipFile(CHECKPOINT_PATH) as z:
    with z.open('policy.pth') as f:
        state_dict = torch.load(io.BytesIO(f.read()), map_location='cuda:0', weights_only=False)
model.policy.load_state_dict(state_dict, strict=True)
print("  Model loaded with checkpoint weights.")

# ============================================================
# EXTENDED RUN: 30 STEP
# ============================================================
print("\n[Extended Run] 30 steps with state monitoring...")
result = env.reset()
obs = result[0] if isinstance(result, tuple) else result

# Vehicle stats logging
prev_action = None
action_changes = 0
prev_obs_seg = None
obs_changes = 0

print(f"\n  {'step':>4} | {'speed (km/h)':>12} | {'action':>30} | {'reward':>7} | {'note'}")
print("  " + "-" * 80)

for i in range(30):
    action, _ = model.predict(obs, deterministic=True)
    result = env.step(action)
    if len(result) == 4:
        obs, reward, done, info = result
        terminated, truncated = done, False
    else:
        obs, reward, terminated, truncated, info = result

    # Action değişimi
    notes = []
    if prev_action is not None:
        action_diff = np.linalg.norm(action - prev_action)
        if action_diff > 0.01:
            action_changes += 1
        else:
            notes.append("sameAction")
    prev_action = action.copy()

    # Observation değişimi (segmentation BEV ortalaması)
    if 'seg_camera' in obs:
        cur_obs_mean = float(np.mean(obs['seg_camera']))
        if prev_obs_seg is not None:
            if abs(cur_obs_mean - prev_obs_seg) > 0.5:
                obs_changes += 1
            else:
                notes.append("sameObs")
        prev_obs_seg = cur_obs_mean

    # Vehicle state - vehicle_measures'dan hız çek (genelde [vx, vy, vz, ...] formatında)
    speed_kmh = "?"
    if 'vehicle_measures' in obs:
        vm = obs['vehicle_measures']
        if hasattr(vm, '__len__') and len(vm) > 0:
            # vehicle_measures genelde hız vektörü içeriyor
            try:
                v_arr = np.array(vm).flatten()
                # Standardize: ilk 3 değer hız vektörü olabilir, ya da tek bir hız değeri
                # Heuristik: en büyük absolute değer hız olabilir
                speed_raw = np.abs(v_arr[:3]).sum() if len(v_arr) >= 3 else float(v_arr[0])
                speed_kmh = f"{speed_raw:6.2f}"
            except Exception:
                speed_kmh = "err"

    note_str = ", ".join(notes) if notes else "OK"
    action_str = f"[{action[0]:+.3f}, {action[1]:+.3f}]"
    print(f"  {i+1:>4} | {speed_kmh:>12} | {action_str:>30} | {reward:+7.3f} | {note_str}")

    if terminated or truncated:
        print(f"\n  Episode ended at step {i+1}: terminated={terminated}, truncated={truncated}")
        if isinstance(info, dict):
            for k, v in info.items():
                if not isinstance(v, (np.ndarray, list)) or (hasattr(v, '__len__') and len(v) < 5):
                    print(f"    info[{k}] = {v}")
        break

# ============================================================
# ANALIZ
# ============================================================
print("\n" + "=" * 60)
print("ANALIZ")
print("=" * 60)
print(f"  Action degisimleri:      {action_changes}/{i+1}")
print(f"  Observation degisimleri: {obs_changes}/{i+1}")

if action_changes < 3:
    print("\n  ⚠ DİKKAT: Action neredeyse hiç değişmiyor.")
    print("    İki olasılık:")
    print("      a) Policy bozuk (state_dict load yanlış mapping)")
    print("      b) Observation pipeline aynı obs döndürüyor (env donmuş)")
elif obs_changes < 3:
    print("\n  ⚠ DİKKAT: Observation değişmiyor ama action değişiyor.")
    print("    Env step gerçek state ilerletmiyor olabilir.")
else:
    print("\n  ✓ Action ve observation ikisi de değişiyor — policy aktif görünüyor.")

env.close()
