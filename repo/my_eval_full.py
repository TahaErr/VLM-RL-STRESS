"""
VLM-RL Full Eval — 5 Town × 10 Route Reprodüksiyon.

Paper Tablo 4 protokolünü uygular:
- 5 town (Town01-05) × 10 deterministic route (eval_routes cycle)
- eval=True modunda env: route sonuna ulaşılınca otomatik success_state=True
  (1 episode = 1 route, paper protokolü)
- Episode termination: collision OR route end (env auto) OR step>=5000 cap
- Deterministic policy

Çıktı: paper Tablo 4 ile karşılaştırılabilir per-town metrikler.

ÖN KOŞUL:
    CARLA server çalışıyor olmalı (port 2000).
    my_eval_minimal.py çalıştırıldı ve düzgün sonuç verdi.

KULLANIM:
    cd C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16\\PythonAPI\\VLM-RL-Stress\\repo
    conda activate carla_env
    python my_eval_full.py

ÇIKTI:
    - results/eval_full_episodes.csv  (50 satır, 1 row per episode, her ep'den sonra güncellenir)
    - results/eval_full_summary.csv   (5+ satır, 1 row per town + overall)
    - Konsola: per-town progress + final summary

TAHMİNİ SÜRE: 40-80 dakika (donanıma + agent hızına göre)
    - Town02: ~10-15s/ep × 10 = 2-3 min
    - Town01/03/04/05: 30-200s/ep × 10 = 5-30 min/town

NOTLAR:
    - eval_routes module-level itertools.cycle, town değişiminde manuel reset edilir
    - h5 cache (Town01-05) 0.9.13→0.9.16 stabil olduğu için ek check gerekmez
    - Vehicle.get_speed() zaten km/h döndürür
    - Bir town fail olursa diğerlerine devam edilir
    - Partial CSV her EPISODE'dan sonra güncellenir → Ctrl+C ile abort durumunda veri kaybolmaz
    - eval=True modunda env, route sonunda otomatik success_state=True yapar (1 ep = 1 route)
    - 'max_steps' terminal_reason: agent route'u tamamlayamadı → partial-success vakası
"""

import warnings
import os
import sys
import zipfile
import io
import time
import traceback
import itertools
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# CONFIG
# ============================================================
CHECKPOINT_PATH = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\checkpoints\hf_vlmrl\sac_model.zip"
RESULTS_DIR = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\results"
EPISODES_CSV = os.path.join(RESULTS_DIR, "eval_full_episodes.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "eval_full_summary.csv")

TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05"]
EPISODES_PER_TOWN = 10
MAX_STEPS_PER_EPISODE = 5000       # safety cap; eval=True'da env route sonunda auto-terminate
                                    # 5000 = ~333s @15fps, paper Town01-04 uzun rotaları için yeterli
ROUTE_END_DETECT_AFTER = 150       # hack: step >= 150 AND wp_idx == 0 (eval=True'da gereksiz ama safety net)

# eval_routes değerleri (paper'ın predefined 10 route'u, repo'dan kopyalandı)
EVAL_ROUTES_TUPLES = [
    (48, 21), (0, 72), (28, 83), (61, 39), (27, 14),
    (6, 67), (61, 49), (37, 64), (33, 80), (12, 30),
]

print("=" * 72)
print("VLM-RL FULL EVAL — 5 TOWNS × 10 ROUTES")
print("=" * 72)
print(f"  Towns:          {TOWNS}")
print(f"  Eps per town:   {EPISODES_PER_TOWN}")
print(f"  Max steps/ep:   {MAX_STEPS_PER_EPISODE}")
print(f"  Total episodes: {len(TOWNS) * EPISODES_PER_TOWN}")

# ============================================================
# IMPORTS (module-level config'i bekler)
# ============================================================
print("\n[Setup] Loading config and modules...")
import config
CONFIG = config.set_config("vlm_rl")
CONFIG.algorithm_params.device = "cuda:0"

from clip.clip_rewarded_sac import CLIPRewardedSAC
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn
from carla_env.rewards import reward_functions
import carla_env.envs.carla_route_env as cre  # eval_routes monkey-patch için

observation_space, encode_state_fn = create_encode_state_fn(CONFIG.state, CONFIG)
print(f"  [✓] CONFIG loaded (algorithm={CONFIG.algorithm}, reward_fn={CONFIG.reward_fn})")


# ============================================================
# HELPERS
# ============================================================
def reset_eval_routes_iterator():
    """Module-level eval_routes cycle'ını her town başında sıfırla.

    Paper protokolünde her town 10 route'a maruz kalır ve route'lar her town'da
    aynı sırayla başlamalı. Module-level itertools.cycle bunu garanti etmez —
    biz iterator'ı manuel olarak yeniliyoruz.
    """
    cre.eval_routes = itertools.cycle(EVAL_ROUTES_TUPLES)


def create_env(town):
    """eval=True olan fresh env yarat."""
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
        activate_traffic_flow=True,   # paper: 20 NPC autopilot
        start_carla=False,
        eval=True,                    # paper protokolü
        town=town,
    )


def load_model(env):
    """Manual state_dict yükleme (Python 3.8 → 3.12 pickle bypass)."""
    model = CLIPRewardedSAC(env=env, config=CONFIG, inference_only=True)
    with zipfile.ZipFile(CHECKPOINT_PATH) as z:
        with z.open('policy.pth') as f:
            sd = torch.load(io.BytesIO(f.read()),
                            map_location='cuda:0',
                            weights_only=False)
    model.policy.load_state_dict(sd, strict=True)
    return model


def run_episode(env, model, town, ep_idx):
    """Tek episode koştur, metrik dict'i döndür."""
    t_reset = time.time()
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    reset_dur = time.time() - t_reset

    speeds = []
    devs = []
    rewards = []
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
                speeds.append(env.vehicle.get_speed())  # km/h
            except Exception:
                pass
        if hasattr(env, 'distance_from_center'):
            devs.append(env.distance_from_center)

        # Paper'ın eval termination hack'i (eval.py satır ~75)
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

    # Terminal sebep teşhisi
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
        'ep_idx': ep_idx,
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


def aggregate_town(eps):
    """10 episode → 1 town summary satırı (paper Tablo 4 formatı).

    NOT: Paper Tablo 4'teki RC değerleri 1.0'a clip'lenmiştir (eval_plots.py'de
    `df['routes_completed'].clip(upper=1)` görülür). Env'in step() içindeki
    waypoint update logic eval=True modunda wp_idx'in route bittikten sonra
    geometric olarak artmasına izin veriyor (modulo wraparound), bu yüzden
    raw RC > 1.0 değerleri görüyoruz. Paper karşılaştırması için RC_paper
    (clipped) sütununu kullan.
    """
    df = pd.DataFrame(eps)
    n = len(df)
    if n == 0:
        return None

    # RC raw vs clipped
    rc_raw = df['routes_completed']
    rc_clipped = rc_raw.clip(upper=1.0)

    collisions = df[df['collision_state']]
    n_col = len(collisions)
    total_steps = int(df['steps'].sum())
    total_dist_km = float(df['total_distance_m'].sum()) / 1000.0

    return {
        'town': df['town'].iloc[0],
        'n_episodes': n,
        'AS_mean_kmh': round(df['mean_speed_kmh'].mean(), 2),
        'AS_std': round(df['mean_speed_kmh'].std(), 2),
        'RC_paper': round(rc_clipped.mean(), 3),       # clipped, paper Tablo 4 formatı
        'RC_paper_std': round(rc_clipped.std(), 3),
        'RC_raw_mean': round(rc_raw.mean(), 3),         # ham değer (geometric wrap dahil)
        'TD_mean_m': round(df['total_distance_m'].mean(), 1),
        'TD_std': round(df['total_distance_m'].std(), 1),
        'CR_collision_rate': round(n_col / n, 3),
        'CS_mean_kmh': round(collisions['collision_speed_kmh'].mean(), 2) if n_col > 0 else 0.0,
        'TCF_per_1k_steps': round(n_col / total_steps * 1000, 4) if total_steps > 0 else 0.0,
        'DCF_per_km': round(n_col / total_dist_km, 4) if total_dist_km > 0 else 0.0,
        'ICT_mean': round(collisions['collision_interval'].mean(), 1) if n_col > 0 else 0.0,
        'SR_success_rate': round(df['success'].sum() / n, 3),
        'mean_center_dev_m': round(df['mean_center_dev_m'].mean(), 4),
    }


def save_partial(all_eps, town_sums):
    """Her town'dan sonra CSV'leri güncelle (crash safety)."""
    if all_eps:
        pd.DataFrame(all_eps).to_csv(EPISODES_CSV, index=False)
    if town_sums:
        pd.DataFrame(town_sums).to_csv(SUMMARY_CSV, index=False)


# ============================================================
# MAIN
# ============================================================
os.makedirs(RESULTS_DIR, exist_ok=True)

# RESUME LOGIC: mevcut CSV varsa oku, tamamlanmış town'ları atla
existing_by_town = {}
if os.path.exists(EPISODES_CSV):
    try:
        existing_df = pd.read_csv(EPISODES_CSV)
        for ep in existing_df.to_dict('records'):
            existing_by_town.setdefault(ep['town'], []).append(ep)
        completed = {t: len(eps) for t, eps in existing_by_town.items()
                     if len(eps) >= EPISODES_PER_TOWN}
        if completed:
            print(f"\n[RESUME] Mevcut data bulundu:")
            for t, n in existing_by_town.items():
                status = "complete, skip" if len(n) >= EPISODES_PER_TOWN else "partial, redo"
                print(f"    {t}: {len(n)} episode ({status})")
    except Exception as e:
        print(f"\n[!] Mevcut CSV okunamadı, baştan başlıyor: {e}")
        existing_by_town = {}

all_episodes = []
town_summaries = []
overall_t0 = time.time()

for town in TOWNS:
    print("\n" + "=" * 72)
    print(f"[{town}]")
    print("=" * 72)

    # Resume: bu town zaten tamamlanmışsa atla, mevcut data'yı koru
    if town in existing_by_town and len(existing_by_town[town]) >= EPISODES_PER_TOWN:
        print(f"  [SKIP] {town} zaten tamamlanmış ({len(existing_by_town[town])} ep), mevcut data korunuyor")
        all_episodes.extend(existing_by_town[town])
        town_summaries.append(aggregate_town(existing_by_town[town]))
        continue

    # 1. eval_routes iterator reset (her town aynı 10 route'la başlasın)
    reset_eval_routes_iterator()
    print(f"  [✓] eval_routes iterator reset to position 0")

    # 2. Fresh env yarat
    print(f"  Creating env (town={town}, eval=True, traffic_flow=True)...")
    try:
        t0 = time.time()
        env = create_env(town)
        print(f"  [✓] Env created in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  [✗] Env creation FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"  [!] Skipping {town}, continuing with next town")
        continue

    # 3. Model yükle (her town'da fresh — env-model coupling güvenli)
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
        continue

    # 4. 10 episode koş
    print(f"  Running {EPISODES_PER_TOWN} episodes...")
    print(f"    {'ep':>3} {'steps':>5} {'speed':>7} {'RC':>6} {'TD':>8} {'CS':>3} {'SR':>3} {'time':>6} {'reason':<15}")
    town_eps = []
    for ep_idx in range(EPISODES_PER_TOWN):
        try:
            metrics = run_episode(env, model, town, ep_idx)
            town_eps.append(metrics)

            print(f"    {ep_idx+1:>3} {metrics['steps']:>5} "
                  f"{metrics['mean_speed_kmh']:>5.1f}km "
                  f"{metrics['routes_completed']:>6.2f} "
                  f"{metrics['total_distance_m']:>7.1f}m "
                  f"{int(metrics['collision_state']):>3} "
                  f"{int(metrics['success']):>3} "
                  f"{metrics['wall_time_s']:>5.1f}s "
                  f"{metrics['terminal_reason']:<15}")

            # Per-episode partial save (crash safety, ucuz işlem)
            save_partial(all_episodes + town_eps, town_summaries)
        except Exception as e:
            print(f"    {ep_idx+1:>3} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Sonraki episode'a devam

    # 5. Env kapat (sonraki town için temiz başlangıç)
    try:
        env.close()
    except Exception as e:
        print(f"  warning: env.close() exception: {e}")

    # 6. Town özeti
    if town_eps:
        all_episodes.extend(town_eps)
        town_sum = aggregate_town(town_eps)
        town_summaries.append(town_sum)

        print(f"\n  >> {town} özet (n={len(town_eps)}):")
        print(f"     AS = {town_sum['AS_mean_kmh']:>6.2f} ± {town_sum['AS_std']:.2f} km/h")
        print(f"     RC = {town_sum['RC_paper']:>6.3f} ± {town_sum['RC_paper_std']:.3f}  "
              f"[clipped to 1.0; raw mean: {town_sum['RC_raw_mean']:.3f}]")
        print(f"     TD = {town_sum['TD_mean_m']:>6.1f} ± {town_sum['TD_std']:.1f} m")
        print(f"     CR = {town_sum['CR_collision_rate']:>6.3f}")
        print(f"     CS = {town_sum['CS_mean_kmh']:>6.2f} km/h "
              f"(collisions: {int(town_sum['CR_collision_rate']*town_sum['n_episodes'])})")
        print(f"     SR = {town_sum['SR_success_rate']:>6.3f}")
        print(f"     center dev = {town_sum['mean_center_dev_m']:.4f} m")

    # 7. Partial save
    save_partial(all_episodes, town_summaries)
    print(f"  [✓] Partial CSVs saved")

# ============================================================
# FINAL SUMMARY
# ============================================================
total_time = time.time() - overall_t0
print("\n" + "=" * 72)
print("FINAL — TÜM TOWN'LAR")
print("=" * 72)
print(f"  Toplam süre:           {total_time/60:.1f} dakika")
print(f"  Tamamlanan episode:    {len(all_episodes)} / {len(TOWNS)*EPISODES_PER_TOWN}")
print(f"  Tamamlanan town:       {len(town_summaries)} / {len(TOWNS)}")

if all_episodes:
    df_all = pd.DataFrame(all_episodes)
    n = len(df_all)
    n_col = int(df_all['collision_state'].sum())
    n_succ = int(df_all['success'].sum())

    # RC clipping for paper compatibility
    rc_paper = df_all['routes_completed'].clip(upper=1.0)

    print(f"\n  Overall metrikler (n={n}):")
    print(f"    AS = {df_all['mean_speed_kmh'].mean():.2f} km/h "
          f"(std {df_all['mean_speed_kmh'].std():.2f})")
    print(f"    RC = {rc_paper.mean():.3f} (paper, clipped) | "
          f"raw mean: {df_all['routes_completed'].mean():.3f}")
    print(f"    TD = {df_all['total_distance_m'].mean():.1f} m "
          f"(std {df_all['total_distance_m'].std():.1f})")
    print(f"    CR = {n_col/n:.3f} ({n_col}/{n} episode)")
    print(f"    SR = {n_succ/n:.3f} ({n_succ}/{n} episode)")
    print(f"    Mean center dev = {df_all['mean_center_dev_m'].mean():.4f} m")

    # Per-town tablo (paper Tablo 4 formatına yakın)
    print(f"\n  Per-town breakdown (paper Tablo 4 formatı, RC clipped at 1.0):")
    df_sum = pd.DataFrame(town_summaries)
    cols_to_show = ['town', 'n_episodes', 'AS_mean_kmh', 'RC_paper',
                    'TD_mean_m', 'CR_collision_rate', 'CS_mean_kmh', 'SR_success_rate']
    print(df_sum[cols_to_show].to_string(index=False))

    print(f"\n  CSV dosyaları:")
    print(f"    {EPISODES_CSV}")
    print(f"    {SUMMARY_CSV}")
else:
    print("\n  [!] Hiç episode tamamlanmadı — CARLA server / kurulum sorunu olabilir")

print("\n" + "=" * 72)
print("✅ FULL EVAL COMPLETED" if all_episodes else "⚠️  FULL EVAL FAILED")
print("=" * 72)

print("\nPaper Tablo 4 ile karşılaştırma için bekledikleri (vlm_rl):")
print("  Town01: AS≈22.9 km/h | RC≈1.00 | SR≈1.00 | CS≈0.03 km/h")
print("  Town02: AS≈19.3 km/h | RC≈0.97 | SR≈0.93 | CS≈0.02 km/h")
print("  Town03: AS değişken | RC≈0.91 | SR≈0.87 | CS düşük")
print("  Town04: AS≈22.0 km/h | RC dengeli | TD>12000m (uzun route)")
print("  Town05: RC≈0.91 | CS≈0.46 km/h")
print("\nTolerans: CR ±20%, RC ±5%, SR ±10% (CARLA 0.9.13→0.9.16 + seed varyansı)")
