"""
VLM-RL Smoke Test — Throughput Measurement.

5000 step kısa training koşturur, gerçek throughput'u (steps/saniye) ölçer
ve 1M step full training'in ne kadar süreceğini extrapolate eder.

Bu test:
- Pipeline çalışıyor mu (env + model + CLIP + RL update) doğrular
- VRAM yeterli mi (OOM olur mu) test eder
- Throughput → wall-time tahmini sağlar
- Full training (run_training.py) için açık go/no-go önerisi verir

KULLANIM:
    # CARLA çalışıyor olmalı
    python train_smoke_test.py
    python train_smoke_test.py --steps 10000   # daha uzun test

ÖN KOŞUL:
    - CARLA server çalışıyor (port 2000)
    - train_one_seed.py aynı klasörde
    - conda carla_env aktif
"""

import sys
import os
import time
import subprocess
import argparse
import re

# ============================================================
# CONFIG
# ============================================================
SMOKE_OUTPUT_DIR = "tensorboard/smoke_test"
SMOKE_SEED = 42
DEFAULT_STEPS = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Smoke test step count (default: {DEFAULT_STEPS})")
    parser.add_argument("--clean", action="store_true",
                        help="Mevcut smoke_test dizinini sil, baştan başla")
    args = parser.parse_args()

    print("=" * 72)
    print("VLM-RL SMOKE TEST — Throughput Measurement")
    print("=" * 72)
    print(f"  Steps:       {args.steps:,}")
    print(f"  Output:      {SMOKE_OUTPUT_DIR}")
    print(f"  Seed:        {SMOKE_SEED}")

    # Clean if requested
    if args.clean and os.path.exists(SMOKE_OUTPUT_DIR):
        import shutil
        print(f"\n[Clean] Removing existing {SMOKE_OUTPUT_DIR}...")
        shutil.rmtree(SMOKE_OUTPUT_DIR)

    # ============================================================
    # PRE-FLIGHT
    # ============================================================
    print("\n[Pre-flight] CARLA bağlantı kontrolü...")
    try:
        import carla
        c = carla.Client('localhost', 2000)
        c.set_timeout(10.0)
        version = c.get_server_version()
        print(f"  [✓] CARLA: version={version}")
    except Exception as e:
        print(f"  [✗] CARLA yanıt vermiyor: {e}")
        print("\nÖnce CARLA server'ı başlat:")
        print("  taskkill /F /IM CarlaUE4-Win64-Shipping.exe /T")
        print("  start \"CARLA\" /D \"C:\\Users\\TAHA\\Desktop\\CARLA_0.9.16\" \\")
        print("    CarlaUE4.exe -quality_level=Low -benchmark -fps=15 \\")
        print("    -RenderOffScreen -prefernvidia -carla-world-port=2000")
        print("  timeout /t 25")
        sys.exit(1)

    # ============================================================
    # RUN TRAINING SUBPROCESS
    # ============================================================
    cmd = [
        sys.executable, "train_one_seed.py",
        "--seed", str(SMOKE_SEED),
        "--total-timesteps", str(args.steps),
        "--output-dir", SMOKE_OUTPUT_DIR,
        "--checkpoint-freq", str(args.steps),     # tek checkpoint sonda
        "--replay-buffer-freq", str(args.steps),
    ]

    print(f"\n[Smoke] Running: {' '.join(cmd)}")
    print(f"[Smoke] Bu test ~{args.steps/30:.0f}-{args.steps/15:.0f} saniye sürecek (15-30 step/s tahmin)")
    print()

    # Subprocess'in stdout'unu UTF-8'e zorla (Windows cp1254 cmd için kritik)
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    sub_env["PYTHONUTF8"] = "1"

    t_start = time.time()
    setup_done_time = None        # env + model load süresi
    sb3_fps_values = []            # SB3'ün rapor ettiği fps'ler

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace',
            bufsize=1,
            env=sub_env,
        )

        for line in process.stdout:
            print(line, end='')

            # "TRAINING START" satırından sonrası gerçek training
            if setup_done_time is None and "TRAINING START" in line:
                setup_done_time = time.time()

            # SB3'ün periyodik fps logunu yakala
            # Format: |    fps                  | 42    |
            m = re.search(r'\|\s*fps\s*\|\s*(\d+(?:\.\d+)?)\s*\|', line)
            if m:
                fps = float(m.group(1))
                sb3_fps_values.append(fps)

        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("\n[!] Test interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n[!] Subprocess error: {e}")
        sys.exit(1)

    t_total = time.time() - t_start
    t_setup = (setup_done_time - t_start) if setup_done_time else 0
    t_train = t_total - t_setup if setup_done_time else t_total

    # ============================================================
    # ANALYSIS
    # ============================================================
    print("\n" + "=" * 72)
    print("SMOKE TEST RESULTS")
    print("=" * 72)
    print(f"  Exit code:                  {return_code}")
    print(f"  Total wall time:            {t_total:.1f}s ({t_total/60:.1f} min)")

    if setup_done_time:
        print(f"  Setup overhead:             {t_setup:.1f}s (env + model + CLIP load)")
        print(f"  Pure training time:         {t_train:.1f}s")

    if return_code != 0:
        print(f"\n[✗] Smoke test FAILED (exit {return_code})")
        print(f"\nLog incele: {SMOKE_OUTPUT_DIR}/training_log.txt")
        print(f"Olası sebepler:")
        print(f"  - CUDA OOM (VRAM yetersiz, ViT-H'a downgrade gerekir)")
        print(f"  - CARLA server crashed")
        print(f"  - Pipeline import bug (yeni paket eksik?)")
        sys.exit(return_code)

    # Throughput hesapla
    throughput_total = args.steps / t_total
    throughput_pure = args.steps / t_train if t_train > 0 else 0
    sb3_avg_fps = sum(sb3_fps_values) / len(sb3_fps_values) if sb3_fps_values else 0

    print(f"\n  Throughput (toplam wall):   {throughput_total:.1f} steps/s")
    print(f"  Throughput (pure training): {throughput_pure:.1f} steps/s")
    if sb3_avg_fps > 0:
        print(f"  SB3 reported avg fps:       {sb3_avg_fps:.1f} steps/s "
              f"(N={len(sb3_fps_values)} samples)")

    # 1M extrapolation (pure training rate kullan, daha güvenilir)
    rate = throughput_pure if throughput_pure > 0 else throughput_total
    h_per_seed = (1_000_000 / rate) / 3600 + (t_setup / 3600)   # +setup
    h_3_seed = h_per_seed * 3
    h_2_seed = h_per_seed * 2

    print(f"\n  1M step extrapolation:")
    print(f"    Per seed:    {h_per_seed:.1f}h  ({h_per_seed/24:.1f} gün)")
    print(f"    2 seed:      {h_2_seed:.1f}h  ({h_2_seed/24:.1f} gün)")
    print(f"    3 seed:      {h_3_seed:.1f}h  ({h_3_seed/24:.1f} gün)")

    # ============================================================
    # RECOMMENDATION
    # ============================================================
    print(f"\n  {'='*68}")
    print(f"  RECOMMENDATION")
    print(f"  {'='*68}")

    if rate >= 30:
        print(f"  ✅ GO — {rate:.0f} step/s yeterince hızlı")
        print(f"     1M step ~{h_per_seed:.0f}h, 3 seed ~{h_3_seed:.0f}h")
        print(f"     Plan: tek gece × 3 + bir kontrol günü")
        print(f"\n     Komut:")
        print(f"       python run_training.py")

    elif rate >= 15:
        print(f"  ⚠️  MARGINAL — {rate:.0f} step/s orta hızda")
        print(f"     1M step ~{h_per_seed:.0f}h, 3 seed ~{h_3_seed:.0f}h ({h_3_seed/24:.1f} gün)")
        print(f"\n     Seçenekler:")
        print(f"       a) Devam et: 3 seed × {h_per_seed:.0f}h overnight koşumlar")
        print(f"       b) 2 seed: {h_2_seed:.0f}h ({h_2_seed/24:.1f} gün) — tasarruf, std güvenilirliği azalır")
        print(f"       c) Step sayısını azalt: 500K step (paper'dan farklı)")

    elif rate >= 8:
        print(f"  🔶 SLOW — {rate:.0f} step/s yavaş")
        print(f"     1M step ~{h_per_seed:.0f}h, 3 seed ~{h_3_seed:.0f}h ({h_3_seed/24:.1f} gün)")
        print(f"\n     Tavsiyeler:")
        print(f"       a) ViT-H'a downgrade (~3-5x hızlanır)")
        print(f"          config.py'de reward_clg.pretrained_model = 'ViT-H-14/laion2b_s32b_b79k'")
        print(f"          Paper Fig 12 ViT-H'ın bigG ile yakın performans gösterdiğini doğruluyor")
        print(f"       b) 2 seed × 500K step = {2 * h_per_seed/2:.0f}h")
        print(f"       c) Sadece 1 seed (paper iddiasını test etme)")

    else:
        print(f"  ❌ TOO SLOW — {rate:.0f} step/s")
        print(f"     1M step ~{h_per_seed:.0f}h kabul edilemez")
        print(f"\n     Zorunlu adımlar:")
        print(f"       1) ViT-H veya ViT-L'a downgrade (config.py)")
        print(f"       2) Smoke test'i tekrar çalıştır")
        print(f"       3) Yeni rate'e göre planı revize et")

    print(f"  {'='*68}\n")

    # VRAM hint
    print(f"  GPU VRAM kontrol için ayrı terminalde:")
    print(f"    nvidia-smi -l 5")
    print(f"  Eğer VRAM training sırasında 15+ GB'a yaklaştıysa OOM riski var.\n")

    # Final FPS değerlerinin trendi
    if len(sb3_fps_values) >= 3:
        first_third = sb3_fps_values[:len(sb3_fps_values)//3]
        last_third = sb3_fps_values[-len(sb3_fps_values)//3:]
        avg_first = sum(first_third) / len(first_third)
        avg_last = sum(last_third) / len(last_third)
        change = (avg_last - avg_first) / avg_first * 100

        if abs(change) > 20:
            print(f"  [!] FPS trendi: ilk üçte bir avg {avg_first:.1f} → son üçte bir avg {avg_last:.1f} "
                  f"({change:+.0f}%)")
            print(f"      Throughput zaman içinde {'düşüyor' if change < 0 else 'artıyor'}, "
                  f"1M extrapolation belirsizleşir")


if __name__ == "__main__":
    main()
