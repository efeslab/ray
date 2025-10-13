#!/usr/bin/env python3
import argparse
import csv
from math import isfinite

def parse_float(x):
    try:
        return float(x)
    except:
        return float('nan')

def main():
    ap = argparse.ArgumentParser(description="Transform progress-by-second CSV into (time_from_start, rows_finished) with interpolation.")
    ap.add_argument("input_csv", help="Input CSV with headers: epoch_sec,completed_in_bin,cumulative_completed")
    ap.add_argument("output_csv", help="Output CSV path")
    ap.add_argument("--batch-size", type=int, default=32, help="Step (rows) to output at each multiple (default: 32)")
    ap.add_argument("--start-at-first-completion", action="store_true",
                    help="Start time_from_start at the first *completion threshold* rather than first epoch in file")
    ap.add_argument("--use-cumulative", action="store_true",
                    help="Prefer the 'cumulative_completed' column instead of summing 'completed_in_bin'")
    args = ap.parse_args()

    # Read and normalize rows
    rows = []
    with open(args.input_csv, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            t = parse_float(r.get("epoch_sec", ""))
            bin_done = parse_float(r.get("completed_in_bin", ""))
            cum = parse_float(r.get("cumulative_completed", ""))
            rows.append({"t": t, "bin": bin_done, "cum": cum})

    # Filter invalid rows and sort by time
    rows = [r for r in rows if isfinite(r["t"])]
    rows.sort(key=lambda r: r["t"])
    if not rows:
        raise SystemExit("No valid rows found.")

    # If cumulative is NaN (or we choose not to trust it), rebuild from bins
    have_good_cum = args.use_cumulative and all(isfinite(r["cum"]) for r in rows)
    if have_good_cum:
        # Also derive per-bin from cumulative (in case provided bin counts are zero)
        prev = None
        for r in rows:
            if prev is None or not isfinite(prev):
                r["bin"] = r["cum"] if isfinite(r["cum"]) else 0.0
            else:
                r["bin"] = max(0.0, r["cum"] - prev)
            prev = r["cum"]
    else:
        # Build cumulative from bins
        cum = 0.0
        for r in rows:
            b = r["bin"] if isfinite(r["bin"]) else 0.0
            cum += b
            r["cum"] = cum

    # Determine bin durations (for interpolation). Assume each row covers [t_i, t_{i+1}] (or 1s if missing next).
    for i in range(len(rows)):
        if i + 1 < len(rows):
            dt = max(0.0, rows[i+1]["t"] - rows[i]["t"])
        else:
            # Last bin: assume 1 second if positive bin, else 0
            dt = 1.0 if rows[i]["bin"] > 0 else 0.0
        rows[i]["dt"] = dt

    # Find threshold crossing times via linear interpolation within each bin
    batch = max(1, args.batch_size)
    results = []
    # Start from the first threshold higher than zero
    next_threshold = batch

    # Optional: set t0 (start) at first epoch or at first completion threshold
    start_time = rows[0]["t"]  # temporary; may be replaced later

    prev_cum = 0.0
    prev_time = rows[0]["t"]
    for i, r in enumerate(rows):
        t0 = r["t"]
        dt = r["dt"]
        bin_done = max(0.0, r["bin"] if isfinite(r["bin"]) else 0.0)
        cum_before = prev_cum
        cum_after = prev_cum + bin_done

        # While we have crossed one or more thresholds within this bin
        while next_threshold <= cum_after:
            if bin_done > 0 and dt > 0:
                # fraction within the bin where crossing occurs
                frac = (next_threshold - cum_before) / bin_done
                frac = max(0.0, min(1.0, frac))
                crossing_time = t0 + frac * dt
            else:
                # No time elapses or no progress: assign at bin start
                crossing_time = t0

            results.append((crossing_time, next_threshold))
            next_threshold += batch

        prev_cum = cum_after
        prev_time = t0 + dt

    if not results:
        # No thresholds reached; nothing to write
        # Still write an empty CSV with headers
        with open(args.output_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["time_from_start", "number_of_rows_finished"])
        return

    # If requested, move the time origin to the time of first threshold crossing
    if args.start_at_first_completion:
        start_time = results[0][0]  # first crossing time

    # Write output
    with open(args.output_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["time_from_start", "number_of_rows_finished"])
        for t_cross, n_rows in results:
            wr.writerow([t_cross - start_time, int(n_rows)])

if __name__ == "__main__":
    main()
