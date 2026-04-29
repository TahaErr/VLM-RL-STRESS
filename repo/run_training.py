"""
VLM-RL Training Orchestrator — 3 Seed Sequential.

Her seed için:
1. CARLA tüm process'leri kill
2. CARLA fresh start
3. train_one_seed.py subprocess (1M step)
4. CARLA kill

Resume desteklenir: bir seed yarıda kalmışsa otomatik kaldığı yerden devam eder.

KULLANIM:
    python run_training.py
    python run_training.py --seeds 42 100 2024
    python run_training.py --seeds 42 --total-timesteps 50000   # smoke test

ÇIKTI:
    tensorboard/seed_<N>/
        model_*_steps.zip
        replay_buffer.pkl
        config.json
        progress.csv
        events.out.tfevents.*
        training_log.txt   (subprocess stdout/stderr)

ÖN KOŞUL:
    - CARLA 0.9.16 binary CARLA_DIR'da
    - train_one_seed.py aynı klasörde
    - conda carla_env aktif
    - GPU 16+ GB VRAM (ViT-bigG-14 için)

NOTLAR:
    - 3 seed × 1M step ≈ 36-72 saat sequential
    - Her seed'in kendi log dosyası: tensorboard/seed_<N>/training_log.txt
    - Kopuş halinde: aynı komutu tekrar çalıştır → resume otomatik
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
REPO_DIR = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\repo"
OUTPUT_BASE = r"C:\Users\TAHA\Desktop\CARLA_0.9.16\PythonAPI\VLM-RL-Stress\repo\tensorboard"

CARLA_STARTUP_WAIT = 25
CARLA_KILL_WAIT = 5
CARLA_HEALTH_TIMEOUT = 10
CARLA_MAX_STARTUP_RETRIES = 3

CARLA_PROCESSES = ["CarlaUE4-Win64-Shipping.exe", "CarlaUE4.exe"]


# ============================================================
# CARLA LIFECYCLE
# ============================================================
def kill_carla(verbose=True):
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
    try:
        import carla
        c = carla.Client('localhost', 2000)
        c.set_timeout(CARLA_HEALTH_TIMEOUT)
        version = c.get_server_version()
        return True, version
    except Exception as e:
        return False, str(e)


def ensure_carla_running():
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
# TRAINING SUBPROCESS
# ============================================================
def is_training_complete(seed_dir, target_steps):
    """seed_dir'da target_steps'lik final model var mı?"""
    if not os.path.isdir(seed_dir):
        return False
    final = os.path.join(seed_dir, f"model_{target_steps}_steps.zip")
    return os.path.exists(final)


def has_partial_training(seed_dir):
    """Resume için checkpoint var mı?"""
    if not os.path.isdir(seed_dir):
        return False
    for f in os.listdir(seed_dir):
        if f.startswith("model_") and f.endswith("_steps.zip"):
            return True
    return False


def run_training_for_seed(seed, total_timesteps, output_dir, log_file_path):
    """train_one_seed.py'yi subprocess olarak çağır, log'u dosyaya da yaz."""
    cmd = [
        sys.executable, "train_one_seed.py",
        "--seed", str(seed),
        "--total-timesteps", str(total_timesteps),
        "--output-dir", output_dir,
    ]
    if has_partial_training(output_dir):
        cmd.append("--resume")
        print(f"  [Worker] Partial training detected, will resume")

    print(f"  [Worker] Subprocess: {' '.join(cmd)}")
    print(f"  [Worker] cwd: {REPO_DIR}")
    print(f"  [Worker] Log file: {log_file_path}")
    print()

    # Subprocess'e UTF-8 zorla (Windows cp1254 cmd için kritik)
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONUTF8"] = "1"

    # Stream stdout to console + file
    try:
        with open(log_file_path, 'a', encoding='utf-8') as logf:
            logf.write(f"\n{'='*72}\n")
            logf.write(f"Training start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            logf.write(f"Seed: {seed}, Total steps: {total_timesteps:,}\n")
            logf.write(f"{'='*72}\n\n")
            logf.flush()

            process = subprocess.Popen(
                cmd, cwd=REPO_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                bufsize=1,
                env=sub_env,
            )

            for line in process.stdout:
                print(line, end='')
                logf.write(line)
                logf.flush()

            return_code = process.wait()
            logf.write(f"\nExit code: {return_code}\n")
            return return_code
    except Exception as e:
        print(f"  [Worker] [✗] Subprocess error: {e}")
        traceback.print_exc()
        return -1


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024],
                        help="Seed listesi (default: 42 100 2024)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Step sayısı per seed (default: 1M, paper-faithful)")
    parser.add_argument("--output-base", type=str, default=OUTPUT_BASE,
                        help="Output root directory")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Tamamlanmış seed'leri atla (default: True)")
    args = parser.parse_args()

    print("=" * 72)
    print("VLM-RL TRAINING ORCHESTRATOR — 3 Seed Sequential")
    print("=" * 72)
    print(f"  Seeds:           {args.seeds}")
    print(f"  Steps per seed:  {args.total_timesteps:,}")
    print(f"  Output base:     {args.output_base}")
    print(f"  Total estimate:  {len(args.seeds) * args.total_timesteps / 15 / 3600:.0f}h "
          f"@ 15 fps (lower bound, real wall-time longer)")

    os.makedirs(args.output_base, exist_ok=True)

    overall_t0 = time.time()
    results = {}

    for seed in args.seeds:
        seed_dir = os.path.join(args.output_base, f"seed_{seed}")
        log_file = os.path.join(seed_dir, "training_log.txt")
        os.makedirs(seed_dir, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"[seed={seed}]")
        print("=" * 72)
        seed_t0 = time.time()

        # Skip if already complete
        if args.skip_completed and is_training_complete(seed_dir, args.total_timesteps):
            print(f"  [SKIP] seed={seed} already complete "
                  f"(model_{args.total_timesteps}_steps.zip exists)")
            results[seed] = "already_complete"
            continue

        # 1. CARLA fresh start
        if not ensure_carla_running():
            print(f"  [✗] CARLA başlatılamadı, seed={seed} atlanıyor")
            results[seed] = "carla_failed"
            continue

        # 2. Run training subprocess
        rc = run_training_for_seed(seed, args.total_timesteps, seed_dir, log_file)

        if rc == 0:
            results[seed] = "success"
            print(f"\n  [✓] seed={seed} training completed")
        else:
            results[seed] = f"failed (exit {rc})"
            print(f"\n  [!] seed={seed} training exit code {rc}")
            print(f"  [!] Check log: {log_file}")

        # 3. CARLA kill (sonraki seed için temiz)
        kill_carla(verbose=False)

        elapsed = time.time() - seed_t0
        print(f"  [seed={seed}] Wall time: {elapsed/3600:.2f}h")

    # Final cleanup
    print("\n" + "=" * 72)
    print("FINAL CLEANUP")
    print("=" * 72)
    kill_carla()

    # Summary
    total_time = time.time() - overall_t0
    print("\n" + "=" * 72)
    print("ORCHESTRATOR COMPLETE")
    print("=" * 72)
    print(f"  Total wall time: {total_time/3600:.2f}h")
    print(f"\n  Per-seed status:")
    for seed, status in results.items():
        print(f"    seed={seed}: {status}")

    print(f"\n  Outputs: {args.output_base}/seed_<N>/")
    print(f"\n  Test için: python run_full_eval.py --towns Town01 Town02")
    print(f"            (her seed'in checkpoint'i için ayrı eval gerekir,")
    print(f"             eval orchestrator'ı seed-aware hale getirilecek)")


if __name__ == "__main__":
    main()
