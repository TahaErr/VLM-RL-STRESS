"""
VLM-RL Env Smoke Test — CARLA 0.9.16 uyumluluk testi.

ÖN KOŞUL:
    CARLA server zaten çalışıyor olmalı. Yeni bir Anaconda Prompt'ta:

        cd C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16
        .\\CarlaUE4.exe -quality_level=Low -benchmark -fps=15 -RenderOffScreen -prefernvidia -carla-world-port=2000

    Server pencerede "Listening on 2000" göründükten sonra bu scripti çalıştırın.

KULLANIM:
    cd C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16\\PythonAPI\\VLM-RL-Stress\\repo
    conda activate carla_env
    python test_env.py

NE YAPAR:
    - train.py'nin import zincirini birebir kopyalar
    - vlm_rl config'ini yükler
    - CarlaRouteEnv oluşturur (start_carla=False, activate_traffic_flow=False)
    - env.reset() çağırır, dönen değerin yapısını rapor eder
    - 5 random step atar, dönen tuple uzunluğunu rapor eder
    - env.close() ile temizler

ÇIKTI:
    Her aşamadan sonra "[N/6] OK" basar. Hata olursa hangi aşamada
    patladığı anında görülür.
"""

import warnings
import os
import sys
import traceback

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
print("=" * 60)
print("VLM-RL ENV SMOKE TEST — CARLA 0.9.16")
print("=" * 60)

# ============================================================
# 1/6 — IMPORT ZINCIRI
# ============================================================
print("\n[1/6] Importing modules in train.py order...")

try:
    print("  - import config ...", end=" ")
    import config
    print("OK")

    print("  - config.set_config('vlm_rl') ...", end=" ")
    CONFIG = config.set_config("vlm_rl")
    print(f"OK (algorithm={CONFIG.algorithm})")

    # SB3 ana modülleri (train.py satir 27)
    print("  - from stable_baselines3 import PPO, DDPG, SAC ...", end=" ")
    from stable_baselines3 import PPO, DDPG, SAC
    print("OK")

    # KRITIK: bu en muhtemel patlama noktasi - eski gym + SB3 2.8 + gymnasium 1.2
    print("  - from clip.clip_rewarded_sac import CLIPRewardedSAC ...", end=" ")
    from clip.clip_rewarded_sac import CLIPRewardedSAC
    print("OK")

    print("  - from clip.clip_rewarded_ppo import CLIPRewardedPPO ...", end=" ")
    from clip.clip_rewarded_ppo import CLIPRewardedPPO
    print("OK")

    print("  - from carla_env.envs.carla_route_env import CarlaRouteEnv ...", end=" ")
    from carla_env.envs.carla_route_env import CarlaRouteEnv
    print("OK")

    print("  - from carla_env.state_commons import create_encode_state_fn ...", end=" ")
    from carla_env.state_commons import create_encode_state_fn
    print("OK")

    print("  - from carla_env.rewards import reward_functions ...", end=" ")
    from carla_env.rewards import reward_functions
    print("OK")

    print("  - from utils import parse_wrapper_class ...", end=" ")
    from utils import parse_wrapper_class
    print("OK")

except Exception as e:
    print(f"\n  ✗ IMPORT FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 2/6 — OBSERVATION SPACE + ENCODE FN
# ============================================================
print("\n[2/6] Building observation_space and encode_state_fn...")
try:
    observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
    print(f"  observation_space type: {type(observation_space).__name__}")
    if hasattr(observation_space, 'spaces'):
        print(f"  obs keys: {list(observation_space.spaces.keys())}")
    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(2)

# ============================================================
# 3/6 — ENV YARATMA
# ============================================================
print("\n[3/6] Creating CarlaRouteEnv (start_carla=False, traffic_flow=False)...")
try:
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
        activate_traffic_flow=False,  # smoke test'te trafik kapali
        start_carla=False,            # server elle baslatildi
    )
    print(f"  env class: {type(env).__name__}")
    print(f"  action_space: {env.action_space}")
    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(3)

# ============================================================
# 4/6 — RESET
# ============================================================
print("\n[4/6] Calling env.reset()...")
try:
    result = env.reset()
    if isinstance(result, tuple):
        print(f"  reset returned tuple of length {len(result)} (gymnasium API)")
        obs = result[0]
    else:
        print(f"  reset returned single value, type={type(result).__name__} (legacy gym API)")
        obs = result

    if isinstance(obs, dict):
        print(f"  obs is dict with keys: {list(obs.keys())}")
        for k, v in obs.items():
            shape_str = getattr(v, 'shape', type(v).__name__)
            print(f"    {k}: {shape_str}")
    else:
        print(f"  obs type: {type(obs).__name__}, shape: {getattr(obs, 'shape', '?')}")
    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(4)

# ============================================================
# 5/6 — STEP x 5
# ============================================================
print("\n[5/6] Running 5 random env.step() calls...")
try:
    for i in range(5):
        action = env.action_space.sample()
        result = env.step(action)
        if i == 0:
            # ilk adimda detayli bas
            print(f"  step return tuple length: {len(result)}")
            if len(result) == 4:
                print("  → legacy gym API: (obs, reward, done, info)")
            elif len(result) == 5:
                print("  → gymnasium API: (obs, reward, terminated, truncated, info)")
            else:
                print(f"  → UNUSUAL: {len(result)}-tuple")

        # done/terminated cikartip rapor et
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        elif len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            terminated = truncated = False

        print(f"  step {i+1}: reward={reward:+.3f} term={terminated} trunc={truncated}")

        if terminated or truncated:
            print(f"  episode ended at step {i+1}, breaking")
            break
except Exception as e:
    print(f"  ✗ FAILED at step iteration: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(5)

# ============================================================
# 6/6 — CLOSE
# ============================================================
print("\n[6/6] Closing env...")
try:
    env.close()
    print("  OK")
except Exception as e:
    print(f"  ✗ CLOSE FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(6)

# ============================================================
print("\n" + "=" * 60)
print("✅ SMOKE TEST PASSED — env CARLA 0.9.16'da çalışıyor")
print("=" * 60)
