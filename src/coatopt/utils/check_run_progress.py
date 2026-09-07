#!/usr/bin/env python3
"""Quick progress checker for coatopt runs (supports local and rclone/FUSE mounted dirs).

Usage:
    python -m coatopt.utils.check_run_progress [options]
    python src/coatopt/utils/check_run_progress.py [options]
    python paper_runs/check_progress.py [options]
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False


def classify_run_dir(run_path: Path, current_time: float, active_window_hours: float = 6.0):
    """Classify a run directory based on existing output files without reading file bodies.
    
    This keeps execution fast over network mounts (rclone / macFUSE).
    """
    try:
        files = set(os.listdir(run_path))
    except Exception as e:
        return "Unknown", f"Error reading dir: {e}", 0

    has_meta = "run_metadata.json" in files
    has_pareto_front = "pareto_front.csv" in files
    has_pareto_designs = "pareto_designs.csv" in files
    has_ckpt = any("checkpoint" in f for f in files)
    has_diversity = (
        "design_diversity_tsne.png" in files or "design_clusters_tsne.png" in files
    )
    has_history = "training_history.csv" in files or "generations.csv" in files
    has_config = "config.ini" in files
    has_materials = "materials.json" in files

    # 1. Done (Fast path: if Pareto front or metadata exists, no mtime check needed)
    if has_meta or has_pareto_front:
        return "Done", "Completed (Pareto front & metadata generated)", 0.0
    if has_diversity and has_pareto_designs:
        return "Done", "Completed (Final diversity plot & Pareto designs generated)", 0.0

    # For remaining incomplete/unstarted runs, check directory date or mtime
    # Directory names typically start with YYYYMMDD (e.g. 20260828, 20260902)
    dir_name = run_path.name
    is_today = False
    if len(dir_name) >= 8 and dir_name[:8].isdigit():
        run_date_str = dir_name[:8]
        today_str = time.strftime("%Y%m%d", time.localtime(current_time))
        is_today = (run_date_str == today_str)

    # Check mtime only when necessary
    age_hours = 999.0
    if is_today or has_ckpt or has_history:
        try:
            # Check directory modification time first (fast)
            age_hours = (current_time - os.path.getmtime(run_path)) / 3600.0
        except Exception:
            pass

    # 2. In Progress vs Failed (Incomplete)
    if has_ckpt or has_history or has_pareto_designs:
        if is_today and age_hours <= active_window_hours:
            return "In Progress", f"Active (updated {age_hours:.1f}h ago)", age_hours
        else:
            return (
                "Failed",
                "Incomplete / Halted (aborted mid-training before completion)",
                age_hours,
            )

    # 3. Startup Failure vs Not Started
    if len(files) <= 3 and (has_config or has_materials):
        if is_today and age_hours <= 2.0:
            return "Not Started", "Initialized recently (awaiting execution)", age_hours
        else:
            return (
                "Failed",
                "Failed at startup (created with config/materials only, never began training)",
                age_hours,
            )

    if len(files) == 0:
        return "Not Started", "Empty directory", age_hours

    return "Unknown", f"Unrecognized file set ({len(files)} files)", age_hours


from concurrent.futures import ThreadPoolExecutor


def scan_results(results_dir: Path, active_window_hours: float = 6.0, max_workers: int = 16):
    current_time = time.time()
    results = {}

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    sub_entries = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name != "plots"])

    def process_dir(r_dir):
        status, details, age = classify_run_dir(
            r_dir, current_time=current_time, active_window_hours=active_window_hours
        )
        return {
            "name": r_dir.name,
            "status": status,
            "details": details,
            "age_hours": age,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for suite_dir in sub_entries:
            suite_name = suite_dir.name
            run_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir()])
            runs = list(executor.map(process_dir, run_dirs))
            results[suite_name] = runs

    return results


def print_report(results, show_details: str = "failed", filter_suite: str = None):
    console = Console() if HAVE_RICH else None

    # Status color formatting
    def fmt_status(st):
        if not HAVE_RICH:
            return st
        if st == "Done":
            return "[bold green]Done[/bold green]"
        elif st == "In Progress":
            return "[bold yellow]In Progress[/bold yellow]"
        elif st == "Failed":
            return "[bold red]Failed[/bold red]"
        elif st == "Not Started":
            return "[dim]Not Started[/dim]"
        return st

    # 1. Summary Table across all Suites
    if HAVE_RICH:
        summary_table = Table(
            title="Coatopt Run Progress Summary",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        summary_table.add_column("Suite / Problem Setup", style="bold")
        summary_table.add_column("Total", justify="right")
        summary_table.add_column("Done", justify="right", style="green")
        summary_table.add_column("In Progress", justify="right", style="yellow")
        summary_table.add_column("Failed", justify="right", style="red")
        summary_table.add_column("Not Started", justify="right", style="dim")
    else:
        print("\n=== Coatopt Run Progress Summary ===")
        print(f"{'Suite':<35} | {'Total':>5} | {'Done':>5} | {'In Prog':>7} | {'Failed':>6} | {'Not St':>6}")
        print("-" * 75)

    grand_total = defaultdict(int)

    for suite, runs in results.items():
        if filter_suite and filter_suite.lower() not in suite.lower():
            continue
        counts = defaultdict(int)
        for r in runs:
            counts[r["status"]] += 1
            grand_total[r["status"]] += 1
        grand_total["Total"] += len(runs)

        if HAVE_RICH:
            summary_table.add_row(
                suite,
                str(len(runs)),
                str(counts["Done"]),
                str(counts["In Progress"]),
                str(counts["Failed"]),
                str(counts["Not Started"]),
            )
        else:
            print(
                f"{suite:<35} | {len(runs):>5} | {counts['Done']:>5} | {counts['In Progress']:>7} | {counts['Failed']:>6} | {counts['Not Started']:>6}"
            )

    if HAVE_RICH:
        summary_table.add_section()
        summary_table.add_row(
            "TOTAL",
            str(grand_total["Total"]),
            str(grand_total["Done"]),
            str(grand_total["In Progress"]),
            str(grand_total["Failed"]),
            str(grand_total["Not Started"]),
            style="bold",
        )
        console.print()
        console.print(summary_table)
    else:
        print("-" * 75)
        print(
            f"{'TOTAL':<35} | {grand_total['Total']:>5} | {grand_total['Done']:>5} | {grand_total['In Progress']:>7} | {grand_total['Failed']:>6} | {grand_total['Not Started']:>6}\n"
        )

    # 2. Detailed listing if requested
    # show_details: "none", "failed", "all", "in-progress"
    if show_details == "none":
        return

    for suite, runs in results.items():
        if filter_suite and filter_suite.lower() not in suite.lower():
            continue

        matching_runs = []
        for r in runs:
            if show_details == "all":
                matching_runs.append(r)
            elif show_details == "failed" and r["status"] == "Failed":
                matching_runs.append(r)
            elif show_details == "in-progress" and r["status"] == "In Progress":
                matching_runs.append(r)
            elif show_details == "not-done" and r["status"] != "Done":
                matching_runs.append(r)

        if not matching_runs:
            continue

        if HAVE_RICH:
            detail_table = Table(
                title=f"{suite} ({show_details.upper()})",
                box=box.SIMPLE_HEAVY,
                header_style="bold magenta",
                expand=True,
            )
            detail_table.add_column("Run Directory", style="bold", ratio=5, overflow="fold")
            detail_table.add_column("Status", justify="center", ratio=1, no_wrap=True)
            detail_table.add_column("Details", ratio=4)
            for r in matching_runs:
                detail_table.add_row(r["name"], fmt_status(r["status"]), r["details"])
            console.print()
            console.print(detail_table)
        else:
            print(f"\n--- {suite} ({show_details.upper()}) ---")
            for r in matching_runs:
                print(f"  [{r['status']:<11}] {r['name']} -> {r['details']}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick progress checker for coatopt runs (local or mounted directory)."
    )
    parser.add_argument(
        "--results-dir",
        "-d",
        type=str,
        default="paper_runs/results",
        help="Path to paper_runs/results directory (default: paper_runs/results)",
    )
    parser.add_argument(
        "--show",
        "-s",
        choices=["all", "failed", "in-progress", "not-done", "none"],
        default="not-done",
        help="Which runs to list in detail: not-done (default), failed, in-progress, all, or none.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Filter by suite substring (e.g. '50layer' or '2obj')",
    )
    parser.add_argument(
        "--active-window",
        type=float,
        default=6.0,
        help="Hours since last file modification to consider a run actively In Progress (default: 6.0h)",
    )

    args = parser.parse_args()

    results_path = Path(args.results_dir)
    # Check if relative path needs resolving from repo root
    if not results_path.exists():
        script_dir = Path(__file__).resolve().parent
        candidate = script_dir.parents[2] / args.results_dir
        if candidate.exists():
            results_path = candidate

    try:
        results = scan_results(results_path, active_window_hours=args.active_window)
        print_report(results, show_details=args.show, filter_suite=args.suite)
    except Exception as e:
        print(f"Error checking progress: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
