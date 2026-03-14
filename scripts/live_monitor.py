#!/usr/bin/env python3
"""
Live IOStat Monitor with ML-based Utilization Prediction
=========================================================

Monitors storage devices in real-time using the ``iostat`` command and
compares IOstat's reported utilization with the ML model's predicted
real utilization.

Workflow
--------
1. Runs ``iostat -txm -d`` once to discover available devices.
2. Lets the user choose which device to monitor.
3. Loads pre-trained ML models from a ``.joblib`` file.
4. Continuously reads ``iostat`` output at a configurable interval.
5. For each sample, computes model features (``read_percent``,
   ``block_size_kb``), predicts maximum IOPS, and calculates the
   real utilization.
6. Prints a live comparison table so the user can see how much more
   accurate the ML prediction is compared to the raw ``iostat``
   report.

Usage
-----
::

    # First, train models on your device benchmark data:
    python scripts/evaluation/13_execute_training.py data/AllDevices.xlsx \\
        --save models.joblib

    # Then run live monitoring:
    python scripts/live_monitor.py models.joblib
    python scripts/live_monitor.py models.joblib --interval 5 --model RANDOM_FOREST
    python scripts/live_monitor.py models.joblib --count 20
"""

import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys
import warnings

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Allow imports from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FEATURES  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADER = (
    f"{'Sample':<8} {'Time':<10} {'IOPS':>8} {'Read%':>7} "
    f"{'BS(kB)':>8} {'IOstat%':>9} {'Ours%':>8} {'Diff':>8}"
)
SEP = "-" * 70


# ---------------------------------------------------------------------------
# iostat helpers
# ---------------------------------------------------------------------------

def check_iostat_installed():
    """Exit with a helpful message if *iostat* is not on the PATH."""
    if shutil.which("iostat") is None:
        print("ERROR: 'iostat' is not installed on this system.")
        print("Install it with:")
        print("  Debian / Ubuntu : sudo apt install sysstat")
        print("  CentOS / RHEL   : sudo yum install sysstat")
        print("  Fedora          : sudo dnf install sysstat")
        sys.exit(1)


def run_iostat(interval, count=2):
    """
    Run ``iostat -txm -d <interval> <count>`` and return stdout.

    Using *count=2* produces two reports: the first contains averages
    since boot (which we discard) and the second covers the most recent
    *interval* seconds.
    """
    cmd = ["iostat", "-txm", "-d", str(interval), str(count)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=interval * count + 30,
        )
        return result.stdout
    except FileNotFoundError:
        print("ERROR: 'iostat' command not found.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("WARNING: iostat timed out.")
        return ""


def parse_iostat_output(output):
    """
    Parse ``iostat -txm -d`` output and return per-device statistics.

    When the output contains multiple report blocks (e.g. *count=2*),
    only the **last** block is used so that we get the most recent
    interval data rather than since-boot averages.

    Returns
    -------
    dict
        ``{device_name: {"r_s", "w_s", "rareq_sz", "wareq_sz", "util"}}``
    """
    lines = output.strip().split("\n")

    # Locate all header lines — we want the *last* report block.
    header_indices = [
        i for i, line in enumerate(lines)
        if "Device" in line and ("%util" in line or "util" in line.lower())
    ]
    if not header_indices:
        return {}

    header_idx = header_indices[-1]  # last header = most recent sample
    header_parts = lines[header_idx].split()

    def _col(name):
        """Return the column index for *name*, or None."""
        for j, h in enumerate(header_parts):
            if h.lower().rstrip(":") == name.lower():
                return j
        return None

    r_s_idx = _col("r/s")
    w_s_idx = _col("w/s")
    util_idx = _col("%util")

    # New-style columns (sysstat >= 11)
    rareq_idx = _col("rareq-sz")
    wareq_idx = _col("wareq-sz")
    # Old-style column (sysstat < 11, values in 512-byte sectors)
    avgrq_idx = _col("avgrq-sz")

    devices = {}
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line or "Device" in line:
            # Another header means we've passed into a new block — stop.
            break
        parts = line.split()
        if len(parts) < 3:
            continue
        dev = parts[0]
        try:
            r_s = float(parts[r_s_idx]) if r_s_idx is not None else 0.0
            w_s = float(parts[w_s_idx]) if w_s_idx is not None else 0.0
            util = float(parts[util_idx]) if util_idx is not None else 0.0

            if rareq_idx is not None and wareq_idx is not None:
                rareq = float(parts[rareq_idx])
                wareq = float(parts[wareq_idx])
            elif avgrq_idx is not None:
                # Old format: avgrq-sz is in 512-byte sectors → ÷ 2 for kB
                avgrq_sectors = float(parts[avgrq_idx])
                rareq = avgrq_sectors / 2.0
                wareq = avgrq_sectors / 2.0
            else:
                rareq = 4.0
                wareq = 4.0

            devices[dev] = {
                "r_s": r_s,
                "w_s": w_s,
                "rareq_sz": rareq,
                "wareq_sz": wareq,
                "util": util,
            }
        except (ValueError, IndexError):
            continue

    return devices


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(stats):
    """
    Derive ML model features from a single iostat sample.

    Returns
    -------
    tuple
        ``(read_percent, block_size_kb, total_iops)``
    """
    r_s = stats["r_s"]
    w_s = stats["w_s"]
    total_iops = r_s + w_s

    if total_iops > 0:
        read_percent = (r_s / total_iops) * 100.0
        block_size_kb = (
            (stats["rareq_sz"] * r_s + stats["wareq_sz"] * w_s)
            / total_iops
        )
    else:
        read_percent = 100.0
        block_size_kb = 4.0

    # Ensure block_size_kb is at least 1 (avoid degenerate input)
    block_size_kb = max(block_size_kb, 1.0)
    return read_percent, block_size_kb, total_iops


# ---------------------------------------------------------------------------
# Device / model matching
# ---------------------------------------------------------------------------

def match_model_device(device_name, models):
    """
    Map an iostat device name (e.g. ``nvme0n1``) to a model key
    (e.g. ``nvme``).  Returns *None* when no match is found.
    """
    if device_name in models:
        return device_name

    # Strip trailing partition / namespace numbers (e.g. nvme0n1 → nvme)
    stripped = re.sub(r"\d+$", "", device_name)
    stripped = re.sub(r"n\d+$", "", stripped)
    if stripped in models:
        return stripped

    # Prefix matching
    for key in models:
        if device_name.startswith(key) or key.startswith(device_name):
            return key

    return None


def select_device_interactive(devices):
    """Display available devices and let the user pick one."""
    dev_list = sorted(devices.keys())
    if not dev_list:
        print("ERROR: No storage devices found in iostat output.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Available Storage Devices")
    print("=" * 60)
    for i, dev in enumerate(dev_list, 1):
        s = devices[dev]
        print(
            f"  [{i}] {dev:<16}  "
            f"r/s={s['r_s']:<8.1f} w/s={s['w_s']:<8.1f} "
            f"%util={s['util']:.1f}"
        )
    print("=" * 60)

    while True:
        try:
            choice = input(
                f"\nSelect device number [1-{len(dev_list)}]: "
            ).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(dev_list):
                return dev_list[idx]
        except (ValueError, EOFError):
            pass
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
        print(f"  Please enter a number between 1 and {len(dev_list)}.")


def select_model_device_interactive(models):
    """Let user manually pick a model device when auto-match fails."""
    keys = sorted(models.keys())
    print("\nAvailable model devices:")
    for i, k in enumerate(keys, 1):
        print(f"  [{i}] {k}")

    while True:
        try:
            choice = input(
                f"\nSelect model device [1-{len(keys)}] or 'q' to quit: "
            ).strip()
            if choice.lower() == "q":
                sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        except (ValueError, EOFError):
            pass
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Live IOStat Monitor — compares iostat utilization with "
            "ML-predicted real utilization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Train models first:\n"
            "  python scripts/evaluation/13_execute_training.py "
            "data/AllDevices.xlsx --save models.joblib\n\n"
            "  # Run live monitor:\n"
            "  python scripts/live_monitor.py models.joblib\n"
            "  python scripts/live_monitor.py models.joblib "
            "--interval 5 --model RANDOM_FOREST\n"
        ),
    )
    parser.add_argument(
        "models_path",
        help="Path to trained models (.joblib file).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Sampling interval in seconds (default: 5).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of samples to collect (0 = unlimited, default: 0).",
    )
    parser.add_argument(
        "--model",
        default="LASSO_POLY",
        help="Model name to use for prediction (default: LASSO_POLY).",
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    check_iostat_installed()

    if not os.path.exists(args.models_path):
        print(f"ERROR: Models file not found: {args.models_path}")
        print(
            "Train models first with:\n"
            "  python scripts/evaluation/13_execute_training.py "
            "data/AllDevices.xlsx --save models.joblib"
        )
        sys.exit(1)

    print(f"Loading models from: {args.models_path}")
    models = joblib.load(args.models_path)
    print(f"Model devices: {sorted(models.keys())}")

    # ------------------------------------------------------------------
    # Discover devices via a quick iostat run
    # ------------------------------------------------------------------
    discover_interval = min(args.interval, 2)
    print(f"\nDiscovering devices (sampling for {discover_interval}s)...")
    initial_output = run_iostat(interval=discover_interval, count=2)
    devices = parse_iostat_output(initial_output)

    if not devices:
        print("ERROR: iostat returned no device data.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Interactive device & model selection
    # ------------------------------------------------------------------
    selected_device = select_device_interactive(devices)
    print(f"\n  → Selected device: {selected_device}")

    model_key = match_model_device(selected_device, models)
    if model_key is None:
        print(
            f"\nWARNING: No trained model found matching '{selected_device}'."
        )
        model_key = select_model_device_interactive(models)

    dev_models = models[model_key]

    if args.model not in dev_models:
        print(
            f"ERROR: Model '{args.model}' not available for device "
            f"'{model_key}'."
        )
        print(f"Available models: {sorted(dev_models.keys())}")
        sys.exit(1)

    chosen_model = dev_models[args.model]
    print(
        f"  → Using model: {args.model}  "
        f"(trained on device: {model_key})"
    )

    # ------------------------------------------------------------------
    # Live monitoring loop
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(
        f"  Live Monitor | Device: {selected_device} | "
        f"Model: {args.model} | Interval: {args.interval}s"
    )
    print(f"{'=' * 70}")
    print(HEADER)
    print(SEP)

    sample_num = 0
    iostat_errors = []
    ours_errors = []

    try:
        while True:
            if 0 < args.count <= sample_num:
                break

            output = run_iostat(interval=args.interval, count=2)
            current_devices = parse_iostat_output(output)

            if selected_device not in current_devices:
                print(
                    f"  [!] Device '{selected_device}' not found in this "
                    f"sample — skipping."
                )
                continue

            stats = current_devices[selected_device]
            read_pct, bs_kb, total_iops = compute_features(stats)
            iostat_util = stats["util"]

            if total_iops > 0:
                X = np.array([[read_pct, bs_kb]])
                predicted_max_iops = float(chosen_model.predict(X)[0])
                predicted_max_iops = max(predicted_max_iops, 1.0)
                our_util = (total_iops / predicted_max_iops) * 100.0
                our_util = min(our_util, 100.0)
            else:
                our_util = 0.0

            diff = iostat_util - our_util
            sample_num += 1
            ts = datetime.datetime.now().strftime("%H:%M:%S")

            print(
                f"{sample_num:<8} {ts:<10} {total_iops:>8.1f} "
                f"{read_pct:>6.1f}% {bs_kb:>7.1f} "
                f"{iostat_util:>8.1f}% {our_util:>7.1f}% "
                f"{diff:>+7.1f}%"
            )

            # Track for summary statistics
            if total_iops > 0:
                iostat_errors.append(iostat_util)
                ours_errors.append(our_util)

    except KeyboardInterrupt:
        pass

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Monitoring stopped — {sample_num} sample(s) collected.")
    print(f"{'=' * 70}")

    if iostat_errors:
        iostat_arr = np.array(iostat_errors)
        ours_arr = np.array(ours_errors)
        print(f"\n  Summary Statistics (samples with I/O activity):")
        print(f"  {'':>20} {'IOstat':>12} {'Ours':>12}")
        print(f"  {'Mean Util%':>20} {iostat_arr.mean():>11.2f}% "
              f"{ours_arr.mean():>11.2f}%")
        print(f"  {'Std  Util%':>20} {iostat_arr.std():>11.2f}% "
              f"{ours_arr.std():>11.2f}%")
        print(f"  {'Min  Util%':>20} {iostat_arr.min():>11.2f}% "
              f"{ours_arr.min():>11.2f}%")
        print(f"  {'Max  Util%':>20} {iostat_arr.max():>11.2f}% "
              f"{ours_arr.max():>11.2f}%")
        mean_diff = (iostat_arr - ours_arr).mean()
        print(
            f"\n  Average IOstat overestimation: {mean_diff:+.2f} "
            f"percentage points"
        )

    print()


if __name__ == "__main__":
    main()
