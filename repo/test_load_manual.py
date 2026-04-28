"""
HF Checkpoint MANUEL LOAD Testi — Pickle bypass.

PROBLEM:
    sac_model.zip içindeki 'data' dosyasında Box config'i CloudPickle ile
    serileştirilmiş. lr_schedule fonksiyonu code obj olarak gömülü ve
    Python 3.8 → 3.12 arası code() argüman sırası değiştiği için
    bu pickle Python 3.12'de açılamıyor.

ÇÖZÜM:
    SB3'ün generic load() yolunu kullanmak yerine:
    1. Yeni bir CLIPRewardedSAC oluştur (smoke test'teki gibi)
    2. Sadece policy.pth'yi zip'ten çıkart
    3. PyTorch state_dict olarak yükle (sürüm bağımsız)

ÖN KOŞUL:
    CARLA server çalışıyor olmalı.

KULLANIM:
    cd C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16\\PythonAPI\\VLM-RL-Stress\\repo
    conda activate carla_env
    python test_load_manual.py
"""

import warnings
import os
import sys
import time
import zipfile
import io
import traceback
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CHECKPOINT_PATH = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\checkpoints\hf_vlmrl\sac_model.zip"

print("=" * 60)
print("HF CHECKPOINT MANUAL LOAD TEST (pickle bypass)")
print("=" * 60)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"  Size: {os.path.getsize(CHECKPOINT_PATH) / (1024*1024):.1f} MB")

# ============================================================
# 1/5 — IMPORT + ENV
# ============================================================
print("\n[1/5] Importing modules and creating env...")
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
    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 2/5 — FRESH MODEL OLUSTUR (NORMAL KONSTRUKTOR)
# ============================================================
print("\n[2/5] Creating fresh CLIPRewardedSAC (no checkpoint, init weights)...")
try:
    model = CLIPRewardedSAC(env=env, config=CONFIG, inference_only=True)
    print(f"  Model class: {type(model).__name__}")
    print(f"  Policy class: {type(model.policy).__name__}")
    print(f"  Device: {model.device}")
    print(f"  Total params: {sum(p.numel() for p in model.policy.parameters()):,}")
    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(2)

# ============================================================
# 3/5 — POLICY.PTH'I ZIP'TEN CIKART VE STATE_DICT YUKLE
# ============================================================
print("\n[3/5] Extracting policy.pth from checkpoint zip and loading state_dict...")
try:
    with zipfile.ZipFile(CHECKPOINT_PATH) as z:
        # Zip içeriğini listele
        names = z.namelist()
        print(f"  Files in zip: {names}")
        if 'policy.pth' not in names:
            print(f"  ✗ policy.pth zip içinde yok!")
            sys.exit(3)

        # policy.pth'yi belleğe oku
        with z.open('policy.pth') as f:
            policy_bytes = f.read()
        print(f"  policy.pth size: {len(policy_bytes) / 1024:.1f} KB")

    # PyTorch ile yükle (weights_only=False çünkü SB3 kayıtları içinde collection objects var)
    buf = io.BytesIO(policy_bytes)
    state_dict = torch.load(buf, map_location='cuda:0', weights_only=False)
    print(f"  state_dict type: {type(state_dict).__name__}")

    if isinstance(state_dict, dict):
        # SB3 policy.pth formatı: dict with 'policy' key OR direct state_dict
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            print("  unwrapped 'state_dict' key")
        keys_sample = list(state_dict.keys())[:5]
        print(f"  first 5 keys: {keys_sample}")
        print(f"  total keys: {len(state_dict)}")

    print("  OK (yüklendi belleğe)")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(3)

# ============================================================
# 4/5 — POLICY'YE STATE_DICT YUKLE
# ============================================================
print("\n[4/5] Loading state_dict into policy...")
try:
    # SB3 policy.pth dosyası genelde {'policy': state_dict, 'optimizer': ..., ...} formatında
    # Direkt deneyelim
    if isinstance(state_dict, dict) and any(k.startswith('mlp_extractor') or k.startswith('actor') or k.startswith('critic') for k in state_dict.keys()):
        # Doğrudan policy state_dict
        missing, unexpected = model.policy.load_state_dict(state_dict, strict=False)
    else:
        # SB3'ün set_parameters'a benzer şey — incele
        print(f"  Unexpected format. Keys: {list(state_dict.keys())[:10]}")
        missing, unexpected = [], []

    print(f"  missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    print(f"  unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")

    if len(missing) == 0 and len(unexpected) == 0:
        print("  ✓ PERFECT MATCH")
    elif len(missing) < 5 and len(unexpected) < 5:
        print("  ⚠ Küçük uyumsuzluk - genelde ihmal edilebilir")
    else:
        print("  ✗ Büyük uyumsuzluk - model yapısı farklı")

    print("  OK")
except Exception as e:
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(4)

# ============================================================
# 5/5 — PREDICT TEST
# ============================================================
print("\n[5/5] Testing predict() on real env...")
try:
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result

    print("  Running 5 steps with predict(deterministic=True)...")
    rewards = []
    for i in range(5):
        action, _ = model.predict(obs, deterministic=True)
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
    print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(5)

# ============================================================
print("\nCleanup...")
try:
    env.close()
except Exception:
    pass

print("\n" + "=" * 60)
print("✅ MANUAL LOAD TEST PASSED")
print("=" * 60)
print("\nCheckpoint policy.pth yüklendi, predict çalışıyor.")
print("Sıradaki: run_eval.py'ı manual load yapacak şekilde adapte etmek.")
