"""
HF Checkpoint Load Test — VLM-RL eval.py satır 188 ile aynı load çağrısı.

Bu test, paper'ın sac_model.zip checkpoint'inin SB3 2.8 + PyTorch 2.11
ortamında yüklenip yüklenemediğini doğrular. Yüklenirse modelin
policy.predict() ile hareket önerebildiğini de teyit eder.

ÖN KOŞUL:
    CARLA server çalışıyor olmalı (test_env_full.py'deki gibi).

KULLANIM:
    cd C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16\\PythonAPI\\VLM-RL-Stress\\repo
    conda activate carla_env
    python test_load.py

NE YAPAR:
    1. Env oluşturur (smoke test'teki ile aynı)
    2. CLIPRewardedSAC.load(...) ile HF checkpoint'i yükler
    3. env.reset() yapar, 5 step boyunca policy.predict() ile action seç
    4. Cleanup

OLASI HATALAR:
    - "Argument 'X' has changed shape from ... to ..." → SB3 2.0 vs 2.8
      observation/action space parametre uyumsuzluğu
    - "weights_only=True" → PyTorch 2.6+ pickle güvenlik kısıtlaması
    - "Missing key" → checkpoint policy state_dict farklı
    - Beklenen değil ama mümkün: NumPy/torch tipi uyumsuzluğu
"""

import warnings
import os
import sys
import time
import traceback
import numpy as np

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# HF checkpoint mutlak yolu
CHECKPOINT_PATH = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\checkpoints\hf_vlmrl\sac_model.zip"

print("=" * 60)
print("HF CHECKPOINT LOAD TEST")
print("=" * 60)
print(f"Checkpoint: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"✗ Checkpoint bulunamadı: {CHECKPOINT_PATH}")
    sys.exit(1)
else:
    size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

# ============================================================
# IMPORT + ENV
# ============================================================
print("\n[1/4] Importing and creating env...")
try:
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
        activate_traffic_flow=False,
        start_carla=False,
    )
    print("  Env ready")
except Exception as e:
    print(f"  ✗ Setup failed: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(2)

# ============================================================
# LOAD CHECKPOINT
# ============================================================
print("\n[2/4] Loading checkpoint via CLIPRewardedSAC.load(...) ...")
print("       (eval.py satır 188 ile birebir aynı çağrı)")
t0 = time.time()
try:
    model = CLIPRewardedSAC.load(
        CHECKPOINT_PATH,
        env=env,
        config=CONFIG,
        device="cuda:0",
        load_clip=False,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Model class: {type(model).__name__}")
    print(f"  Policy class: {type(model.policy).__name__}")
    print(f"  Device: {model.device}")
except Exception as e:
    print(f"  ✗ LOAD FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\n[!] Bu hata büyük ihtimalle SB3 versiyon farkından kaynaklanıyor.")
    print("    eval.py'da kullanılan signature ile bizimki aynı,")
    print("    yani yazarın 2.0.0'da kaydettiği checkpoint 2.8.0'da yüklenemiyor olabilir.")
    sys.exit(3)

# ============================================================
# PREDICT
# ============================================================
print("\n[3/4] Testing policy.predict() with real observations...")
try:
    print("  Calling env.reset()...")
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result

    print("  Running 5 steps with policy.predict()...")
    rewards = []
    for i in range(5):
        action, _states = model.predict(obs, deterministic=True)
        result = env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        rewards.append(reward)
        print(f"    step {i+1}: action={action}, reward={reward:+.3f}, term={terminated}")
        if terminated or truncated:
            break
    print(f"  Mean reward: {np.mean(rewards):+.3f}")
    print("  OK")
except Exception as e:
    print(f"  ✗ PREDICT FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(4)

# ============================================================
# CLEANUP
# ============================================================
print("\n[4/4] Cleanup...")
try:
    env.close()
    print("  OK")
except Exception as e:
    print(f"  cleanup warning: {e}")

print("\n" + "=" * 60)
print("✅ CHECKPOINT LOAD TEST PASSED")
print("=" * 60)
print("\nModel HF checkpoint'inden yüklendi ve gerçek observation'larda")
print("hareket önerebiliyor. run_eval.py'a geçilebilir.")
