"""
VLM-RL Full Eval Orchestrator — Subprocess Wrapper.

Her town için CARLA server'ı temiz başlatır, eval_one_town.py'yi subprocess
olarak çağırır, sonra CARLA'yı kapatır. Multi-map switching bug'ını
(MultiStreamState assertion) tamamen bypass eder.

KULLANIM:
    python run_full_eval.py
    python run_full_eval.py --towns Town01 Town04
    python run_full_eval.py --max-steps 8000

ÇIKTI:
    - results/eval_full_episodes.csv (eval_one_town.py tarafından)
    - results/eval_full_summary.csv  (bu script regenerate eder)
    - Konsola: per-town progress + final summary

ÖN KOŞUL:
    - CARLA 0.9.16 binary `CARLA_DIR` altında olmalı
    - eval_one_town.py aynı klasörde
    - conda carla_env aktif

NOTLAR:
    - Her town için CARLA fresh restart (multi-map bug bypass)
    - eval_one_town.py kendi resume logic'ine sahip → kopuş halinde re-run mümkün
    - --towns parametresi ile selective rerun
"""

import sys
import os
import time
import subprocess
import argparse
import traceback

# ============================================================
# CONFIG
# ============================================================
CARLA_DIR = r"C:\Users\TAHA\Desktop\CARLA_0.9.16"
CARLA_EXE_NAME = "CarlaUE4.exe"
RESULTS_DIR = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\results"
REPO_DIR = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\repo"

EPISODES_CSV = os.path.join(RESULTS_DIR, "eval_full_episodes.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "eval_full_summary.csv")

CARLA_STARTUP_WAIT = 25
CARLA_KILL_WAIT = 5
CARLA_HEALTH_TIMEOUT = 10
CARLA_MAX_STARTUP_RETRIES = 3

CARLA_PROCESSES = ["CarlaUE4-Win64-Shipping.exe", "CarlaUE4.exe"]


# ============================================================
# CARLA LIFECYCLE
# ============================================================
def kill_carla(verbose=True):
    """Tüm CARLA process'lerini öldür."""
    if verbose:
        print("  [CARLA] Killing existing processes...")
    for proc_name in CARLA_PROCESSES:
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", proc_name, "/T"],
                capture_output=True, timeout=10
            )
        except Exception:
            pass
    time.sleep(CARLA_KILL_WAIT)


def start_carla():
    """CARLA server'ı arka planda başlat."""
    print("  [CARLA] Starting fresh server...")
    carla_exe = os.path.join(CARLA_DIR, CARLA_EXE_NAME)
    if not os.path.exists(carla_exe):
        print(f"  [CARLA] [✗] Binary not found at {carla_exe}")
        return False

    flags = 0
    if os.name == 'nt':
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        subprocess.Popen(
            [carla_exe, "-quality_level=Low", "-benchmark", "-fps=15",
             "-RenderOffScreen", "-prefernvidia", "-carla-world-port=2000"],
            cwd=CARLA_DIR,
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"  [CARLA] [✗] Popen failed: {e}")
        return False

    print(f"  [CARLA] Waiting {CARLA_STARTUP_WAIT}s for ramp-up...")
    time.sleep(CARLA_STARTUP_WAIT)
    return True


def carla_healthy():
    """Hızlı bağlantı testi."""
    try:
        import carla
        c = carla.Client('localhost', 2000)
        c.set_timeout(CARLA_HEALTH_TIMEOUT)
        version = c.get_server_version()
        return True, version
    except Exception as e:
        return False, str(e)


def ensure_carla_running():
    """Kill → start → verify döngüsü, max retry'a kadar."""
    for attempt in range(1, CARLA_MAX_STARTUP_RETRIES + 1):
        print(f"\n  [CARLA] Attempt {attempt}/{CARLA_MAX_STARTUP_RETRIES} — kill, start, verify")
        kill_carla()
        if not start_carla():
            continue
        ok, info = carla_healthy()
        if ok:
            print(f"  [CARLA] [✓] Healthy: version={info}")
            return True
        else:
            print(f"  [CARLA] [✗] Health check failed: {info}")
            if attempt < CARLA_MAX_STARTUP_RETRIES:
                print(f"  [CARLA] Waiting 10s before retry...")
                time.sleep(10)
    return False


# ============================================================
# WORKER INVOCATION
# ============================================================
def run_eval_for_town(town, seed, episodes, max_steps):
    """eval_one_town.py'yi subprocess olarak çağır."""
    cmd = [
        sys.executable, "eval_one_town.py", town,
        "--seed", str(seed),
        "--episodes", str(episodes),
        "--max-steps", str(max_steps),
    ]
    print(f"  [Worker] Subprocess: {' '.join(cmd)}")
    print(f"  [Worker] cwd: {REPO_DIR}")
    print()

    # UTF-8 zorla (Windows cp1254 cmd için)
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONUTF8"] = "1"

    try:
        result = subprocess.run(cmd, cwd=REPO_DIR, env=sub_env)
        return result.returncode
    except Exception as e:
        print(f"  [Worker] [✗] Subprocess error: {e}")
        traceback.print_exc()
        return -1


# ============================================================
# AGGREGATION
# ============================================================
def aggregate_summary():
    """Paper-faithful aggregation: per (town, seed) mean → 3-seed mean ± std."""
    import pandas as pd

    if not os.path.exists(EPISODES_CSV):
        print(f"\n[!] No episodes CSV at {EPISODES_CSV}, skipping summary")
        return None

    df = pd.read_csv(EPISODES_CSV)
    if len(df) == 0:
        print(f"\n[!] Episodes CSV is empty")
        return None

    # 'seed' kolonu yoksa legacy run, sentinel olarak işaretle
    if 'seed' not in df.columns:
        df['seed'] = -1

    summaries = []
    for town in sorted(df['town'].unique()):
        town_df = df[df['town'] == town]
        n_total = len(town_df)
        seeds = sorted(town_df['seed'].unique())
        n_seeds = len(seeds)

        # Her seed için per-seed mean hesapla
        per_seed_metrics = []
        for s in seeds:
            sd = town_df[town_df['seed'] == s]
            rc_clipped = sd['routes_completed'].clip(upper=1.0)
            sd_collisions = sd[sd['collision_state']]
            per_seed_metrics.append({
                'seed': s,
                'n_eps': len(sd),
                'AS_mean': sd['mean_speed_kmh'].mean(),
                'RC_paper_mean': rc_clipped.mean(),
                'TD_mean': sd['total_distance_m'].mean(),
                'CR': len(sd_collisions) / len(sd),
                'CS_mean': sd_collisions['collision_speed_kmh'].mean() if len(sd_collisions) > 0 else 0.0,
                'SR': sd['success'].sum() / len(sd),
                'center_dev': sd['mean_center_dev_m'].mean(),
            })

        # Paper-faithful: 3-seed mean'lerinin mean ± std (her metrik için)
        seed_df = pd.DataFrame(per_seed_metrics)

        # Aynı zamanda flat-N özet (tüm episode'lar birlikte) — ekstra bilgi
        rc_raw = town_df['routes_completed']
        rc_clipped_all = rc_raw.clip(upper=1.0)
        collisions = town_df[town_df['collision_state']]
        n_col = len(collisions)
        total_steps = int(town_df['steps'].sum())
        total_dist_km = float(town_df['total_distance_m'].sum()) / 1000.0

        summary = {
            'town': town,
            'n_seeds': n_seeds,
            'n_total_episodes': n_total,
            'seeds': str(seeds),

            # PAPER-FAITHFUL: 3-seed agregasyonu (her seed mean'inin mean ± std)
            'AS_mean_kmh': round(seed_df['AS_mean'].mean(), 2),
            'AS_std': round(seed_df['AS_mean'].std(), 2),
            'RC_paper': round(seed_df['RC_paper_mean'].mean(), 3),
            'RC_paper_std': round(seed_df['RC_paper_mean'].std(), 3),
            'TD_mean_m': round(seed_df['TD_mean'].mean(), 1),
            'TD_std': round(seed_df['TD_mean'].std(), 1),
            'CR_collision_rate': round(seed_df['CR'].mean(), 3),
            'CR_std': round(seed_df['CR'].std(), 3),
            'CS_mean_kmh': round(seed_df['CS_mean'].mean(), 2),
            'CS_std': round(seed_df['CS_mean'].std(), 2),
            'SR_success_rate': round(seed_df['SR'].mean(), 3),
            'SR_std': round(seed_df['SR'].std(), 3),
            'mean_center_dev_m': round(seed_df['center_dev'].mean(), 4),

            # FLAT (tüm episode'lar birleşik) — bilgilendirici
            'TCF_per_1k_steps_flat': round(n_col / total_steps * 1000, 4) if total_steps > 0 else 0.0,
            'DCF_per_km_flat': round(n_col / total_dist_km, 4) if total_dist_km > 0 else 0.0,
            'ICT_mean_flat': round(collisions['collision_interval'].mean(), 1) if n_col > 0 else 0.0,
        }
        summaries.append(summary)

    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[✓] Summary CSV regenerated: {SUMMARY_CSV}")
    print(f"    Schema: paper-faithful aggregation = mean ± std across {n_seeds} seeds (mean of per-seed means)")
    return sum_df


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="VLM-RL Multi-Seed Eval Orchestrator (paper-faithful 3-seed reproduction)")
    parser.add_argument("--towns", nargs="+",
                        default=["Town01", "Town02", "Town03", "Town04", "Town05"],
                        choices=['Town01', 'Town02', 'Town03', 'Town04', 'Town05'])
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 42, 2024],
                        help="Inference seed listesi (default: 100 42 2024). "
                             "100 = paper config default. "
                             "Her seed × town ayrı subprocess olarak koşar.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per (town, seed) cell — paper N=10")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--skip-completed", action="store_true",
                        help="Mevcut CSV'de tamamlanmış (town, seed) hücrelerini atla")
    args = parser.parse_args()

    print("=" * 72)
    print("VLM-RL MULTI-SEED EVAL — Paper-Faithful 3-Seed Reproduction")
    print("=" * 72)
    print(f"  Towns:           {args.towns}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Episodes/cell:   {args.episodes}")
    print(f"  Max steps/ep:    {args.max_steps}")
    print(f"  Total cells:     {len(args.towns) * len(args.seeds)}")
    print(f"  Total episodes:  {len(args.towns) * len(args.seeds) * args.episodes}")

    # Resume check — (town, seed) çiftleri için
    completed_cells = set()
    if args.skip_completed and os.path.exists(EPISODES_CSV):
        try:
            import pandas as pd
            existing_df = pd.read_csv(EPISODES_CSV)
            if 'seed' not in existing_df.columns:
                existing_df['seed'] = -1
            counts = existing_df.groupby(['town', 'seed']).size()
            for (t, s), n in counts.items():
                if n >= args.episodes:
                    completed_cells.add((t, int(s)))
            if completed_cells:
                print(f"\n[Resume] Tamamlanmış (town, seed) hücreleri:")
                for cell in sorted(completed_cells):
                    print(f"    {cell}")
        except Exception as e:
            print(f"\n[!] CSV okuma uyarı: {e}")

    overall_t0 = time.time()
    results = {}

    # Outer loop: SEED (her seed kendi CARLA session'ına sahip)
    # Inner loop: TOWN (her town aynı CARLA session'da map switch ile)
    # NOT: CARLA multi-map bug nedeniyle aslında her (seed, town) için fresh CARLA daha güvenli.
    # O yüzden: outer = seed, inner = town, her cell için fresh CARLA.
    for seed in args.seeds:
        for town in args.towns:
            cell = (town, seed)

            print("\n" + "=" * 72)
            print(f"[CELL: town={town}, seed={seed}]")
            print("=" * 72)
            cell_t0 = time.time()

            if cell in completed_cells:
                print(f"  [SKIP] (town={town}, seed={seed}) zaten tamamlanmış")
                results[cell] = "skipped"
                continue

            # 1. CARLA fresh start
            if not ensure_carla_running():
                print(f"  [✗] CARLA başlatılamadı, {cell} atlanıyor")
                results[cell] = "carla_failed"
                continue

            # 2. eval_one_town.py subprocess
            rc = run_eval_for_town(town, seed, args.episodes, args.max_steps)
            if rc == 0:
                results[cell] = "success"
                print(f"\n  [✓] {cell} eval completed (subprocess exit 0)")
            else:
                results[cell] = f"failed (exit {rc})"
                print(f"\n  [!] {cell} eval failed (subprocess exit {rc})")

            # 3. CARLA kill (sonraki cell için temiz)
            kill_carla(verbose=False)

            elapsed = time.time() - cell_t0
            print(f"  [{cell}] Wall time: {elapsed/60:.1f} min")

    # Final cleanup
    print("\n" + "=" * 72)
    print("FINAL CLEANUP")
    print("=" * 72)
    kill_carla()

    # Aggregate summary
    print("\n" + "=" * 72)
    print("REGENERATING SUMMARY (paper-faithful 3-seed aggregation)")
    print("=" * 72)
    sum_df = aggregate_summary()

    # Final report
    total_time = time.time() - overall_t0
    print("\n" + "=" * 72)
    print("ORCHESTRATOR COMPLETE")
    print("=" * 72)
    print(f"  Total wall time: {total_time/60:.1f} min")
    print(f"\n  Per-cell status:")
    for cell, status in sorted(results.items()):
        print(f"    {cell}: {status}")

    if sum_df is not None:
        print(f"\n  Summary tablosu (paper Tab 4 formatında, 3-seed mean ± std):")
        cols = ['town', 'n_seeds', 'n_total_episodes',
                'AS_mean_kmh', 'AS_std',
                'RC_paper', 'RC_paper_std',
                'CR_collision_rate', 'CR_std',
                'CS_mean_kmh', 'CS_std',
                'SR_success_rate', 'SR_std']
        print(sum_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
