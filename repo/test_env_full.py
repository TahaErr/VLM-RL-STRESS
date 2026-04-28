"""
VLM-RL Env Kapsamlı Smoke Test — CARLA 0.9.16

Bu test minimal smoke test'in genişletilmiş versiyonudur. Eğitim ve eval
sırasında karşılaşılabilecek senaryoları önceden test eder.

ÖN KOŞUL:
    CARLA server temiz başlatılmış olmalı. Önceki testten sonra hayalet
    aktörler kalmış olabilir, server'ı yeniden başlatmanız önerilir:

        taskkill /F /IM CarlaUE4-Win64-Shipping.exe /T
        taskkill /F /IM CarlaUE4.exe /T
        start "CARLA Server" /D "C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16" CarlaUE4.exe -quality_level=Low -benchmark -fps=15 -RenderOffScreen -prefernvidia -carla-world-port=2000

    15 saniye bekleyin.

KULLANIM:
    cd C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16\\PythonAPI\\VLM-RL-Stress\\repo
    conda activate carla_env
    python test_env_full.py

NE YAPAR:
    Aşama A: train.py akışında env oluşturma (trafik flow AÇIK)
    Aşama B: 100 step ileri-doğru sürüş (sensor'ler tetiklenir)
    Aşama C: ikinci reset + 50 step (reset stabilitesi testi)
    Aşama D: temizlik

ÇIKTI:
    Reset süresi, episode istatistikleri, info dict içeriği gibi
    eğitimde önemli metrikleri raporlar.
"""

import warnings
import os
import sys
import time
import traceback
import numpy as np

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60)
print("VLM-RL ENV KAPSAMLI SMOKE TEST — CARLA 0.9.16")
print("=" * 60)

# ============================================================
# IMPORT
# ============================================================
print("\n[INIT] Importing modules...")
try:
    import config
    CONFIG = config.set_config("vlm_rl")
    from carla_env.envs.carla_route_env import CarlaRouteEnv
    from carla_env.state_commons import create_encode_state_fn
    from carla_env.rewards import reward_functions
    print(f"  Config: {CONFIG.algorithm}, reward_fn={CONFIG.reward_fn}")
    print(f"  obs_res={CONFIG.obs_res}, use_seg_bev={CONFIG.use_seg_bev}, use_rgb_bev={CONFIG.use_rgb_bev}")
    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# AŞAMA A — ENV YARATMA (TRAFİK FLOW AÇIK)
# ============================================================
print("\n[A] Creating env with activate_traffic_flow=True ...")
t0 = time.time()
try:
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
        activate_traffic_flow=True,   # FARKLI: trafik açık
        start_carla=False,
    )
    print(f"  Env created in {time.time()-t0:.1f}s")
    print(f"  action_space: {env.action_space}")
    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED at env creation: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(2)

# ============================================================
# AŞAMA B — EPISODE 1: 100 STEP FORWARD DRIVING
# ============================================================
print("\n[B] Episode 1: 100 forward-driving steps...")
try:
    # Eğer reset constructor içinde çağrılmışsa step ile başlayabiliriz
    # Ama temiz başlangıç için yine reset çağıralım
    print("  Calling reset()...")
    t_reset = time.time()
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    print(f"  Reset completed in {time.time()-t_reset:.1f}s")

    # Forward driving: throttle=0.5, steer=0.0
    # action_space Box(-1,1,(2,)): [steer, throttle/brake]
    # Eski VLM-RL standardına göre genelde [throttle/brake, steer]
    # İkisini de deniyoruz - güvenli orta yol: hafif gaz, sıfır direksiyon
    forward_action = np.array([0.0, 0.5], dtype=np.float32)

    rewards = []
    speeds = []
    distances = []
    terminated_at = None

    print("  Running 100 steps with action=[0.0, 0.5]...")
    t_steps = time.time()
    for i in range(100):
        result = env.step(forward_action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result

        rewards.append(reward)

        # Bazı bilgileri info'dan çekmeye çalış
        if isinstance(info, dict):
            if 'speed' in info or 'avg_speed' in info:
                speeds.append(info.get('speed', info.get('avg_speed', 0)))

        # Her 20 step'te durum yazdır
        if (i + 1) % 20 == 0:
            avg_r = np.mean(rewards[-20:])
            print(f"    step {i+1:3d}: avg_reward(last 20) = {avg_r:+.3f}, term={terminated}, trunc={truncated}")

        if terminated or truncated:
            terminated_at = i + 1
            break

    duration = time.time() - t_steps
    actual_fps = (terminated_at or 100) / duration

    print(f"  Episode 1 stats:")
    print(f"    duration:      {duration:.1f}s ({actual_fps:.1f} effective FPS)")
    print(f"    steps:         {terminated_at or 100}")
    print(f"    total reward:  {sum(rewards):+.2f}")
    print(f"    mean reward:   {np.mean(rewards):+.3f}")
    print(f"    reward range:  [{min(rewards):+.3f}, {max(rewards):+.3f}]")
    if terminated_at:
        print(f"    ended early at step {terminated_at} (term/trunc)")
    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED during episode 1: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(3)

# ============================================================
# AŞAMA C — EPISODE 2: RESET STABILITY TEST
# ============================================================
print("\n[C] Episode 2: reset stability test (50 random steps)...")
try:
    print("  Calling reset() #2...")
    t_reset2 = time.time()
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    print(f"  Reset #2 completed in {time.time()-t_reset2:.1f}s")

    rewards2 = []
    print("  Running 50 random steps...")
    for i in range(50):
        action = env.action_space.sample()
        result = env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        rewards2.append(reward)
        if terminated or truncated:
            print(f"    episode 2 ended at step {i+1}")
            break

    print(f"  Episode 2 stats:")
    print(f"    steps:        {len(rewards2)}")
    print(f"    mean reward:  {np.mean(rewards2):+.3f}")
    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED during episode 2: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(4)

# ============================================================
# AŞAMA D — CLEANUP
# ============================================================
print("\n[D] Cleanup...")
try:
    env.close()
    print("  OK")
except Exception as e:
    print(f"  ✗ Cleanup warning (genelde sorun değil): {type(e).__name__}: {e}")

# ============================================================
print("\n" + "=" * 60)
print("✅ KAPSAMLI SMOKE TEST PASSED")
print("=" * 60)
print("\nEnv eğitim ve eval için hazır görünüyor. Sıradaki adım:")
print("  - HF checkpoint ile reprodüksiyon (eval.py / run_eval.py)")
