#!/usr/bin/env python3
"""Plot TGS throughput over time from worker logs.

This script reads the worker log format produced by ``worker.py``:

    __main__:INFO [2026-04-27 12:57:29,542] worker, report, 2, 53.55, 41

It also reads the trace CSV so it can label each job as high or low priority.
The default output is a PNG that compares throughput over time for high- and
low-priority jobs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(
    r"worker, report, (?P<job_id>\d+), (?P<throughput>[0-9.]+), (?P<finished_iterations>\d+)"
)
START_PATTERN = re.compile(r"trainer, start, (?P<start_time>[0-9.]+)")
TIME_PATTERN = re.compile(r"\[(?P<timestamp>[^\]]+)\]")


@dataclass
class ReportPoint:
    time_seconds: float
    throughput: float
    finished_iterations: int
    job_id: int
    priority: str


@dataclass
class StartPoint:
    time_seconds: float
    job_id: int
    priority: str


def parse_timestamp(text: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported timestamp format: {text}")


def load_priorities(trace_path: Path) -> Dict[int, str]:
    priorities: Dict[int, str] = {}
    with trace_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            priority = row.get("priority", "unknown").strip() or "unknown"
            job_id_text = row.get("job_id")
            if job_id_text is not None and job_id_text.strip():
                job_id = int(job_id_text)
            else:
                job_id = index
            priorities[job_id] = priority
    return priorities


def expand_log_inputs(log_paths: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for log_path in log_paths:
        if log_path.is_dir():
            for pattern in ("*.log", "*.txt"):
                resolved.extend(sorted(log_path.glob(pattern)))
            continue

        matches = glob.glob(str(log_path))
        if matches and ("*" in str(log_path) or "?" in str(log_path) or "[" in str(log_path)):
            resolved.extend(Path(match) for match in sorted(matches))
            continue

        resolved.append(log_path)

    unique_paths: List[Path] = []
    seen = set()
    for path in resolved:
        normalized = path.resolve() if path.exists() else path
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(path)
    return unique_paths


def parse_worker_logs(log_paths: Iterable[Path], priorities: Dict[int, str]) -> List[ReportPoint]:
    records: List[Tuple[datetime, ReportPoint]] = []

    for log_path in log_paths:
        with log_path.open() as handle:
            for line in handle:
                match = LOG_PATTERN.search(line)
                if not match:
                    continue

                time_match = TIME_PATTERN.search(line)
                if not time_match:
                    continue

                timestamp = parse_timestamp(time_match.group("timestamp"))
                job_id = int(match.group("job_id"))
                throughput = float(match.group("throughput"))
                finished_iterations = int(match.group("finished_iterations"))
                priority = priorities.get(job_id, "unknown")

                records.append(
                    (
                        timestamp,
                        ReportPoint(
                            time_seconds=0.0,
                            throughput=throughput,
                            finished_iterations=finished_iterations,
                            job_id=job_id,
                            priority=priority,
                        ),
                    )
                )

    if not records:
        return []

    records.sort(key=lambda item: item[0])
    start_time = records[0][0]

    parsed: List[ReportPoint] = []
    for timestamp, point in records:
        point.time_seconds = (timestamp - start_time).total_seconds()
        parsed.append(point)
    return parsed


def parse_start_times(log_paths: Iterable[Path], priorities: Dict[int, str]) -> List[StartPoint]:
    records: List[Tuple[datetime, StartPoint]] = []

    for log_path in log_paths:
        with log_path.open() as handle:
            for line in handle:
                time_match = TIME_PATTERN.search(line)
                if not time_match:
                    continue

                start_match = START_PATTERN.search(line)
                if not start_match:
                    continue

                job_match = re.search(r"job (?P<job_id>\d+), trainer, start", line)
                if not job_match:
                    continue

                timestamp = parse_timestamp(time_match.group("timestamp"))
                job_id = int(job_match.group("job_id"))
                priority = priorities.get(job_id, "unknown")
                records.append(
                    (
                        timestamp,
                        StartPoint(time_seconds=0.0, job_id=job_id, priority=priority),
                    )
                )

    if not records:
        return []

    records.sort(key=lambda item: item[0])
    start_time = records[0][0]

    parsed: List[StartPoint] = []
    for timestamp, point in records:
        point.time_seconds = (timestamp - start_time).total_seconds()
        parsed.append(point)
    return parsed


def build_series(points: Iterable[ReportPoint]) -> Dict[str, List[ReportPoint]]:
    grouped: Dict[str, List[ReportPoint]] = defaultdict(list)
    for point in points:
        grouped[point.priority].append(point)
    return grouped


def smooth_values(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values

    smoothed: List[float] = []
    running_total = 0.0
    for index, value in enumerate(values):
        running_total += value
        if index >= window:
            running_total -= values[index - window]
        smoothed.append(running_total / min(index + 1, window))
    return smoothed


def plot_points(
    points: List[ReportPoint],
    start_points: List[StartPoint],
    output_path: Path,
    title: str,
    smooth_window: int,
) -> None:
    grouped = build_series(points)
    colors = {
        "high": "#d1495b",
        "low": "#2d6cdf",
        "mps": "#1f9d8f",
        "Co-ex": "#8a5cf6",
        "unknown": "#666666",
    }

    plt.figure(figsize=(10, 5.5))

    for start_point in sorted(start_points, key=lambda point: point.time_seconds):
        plt.axvline(
            x=start_point.time_seconds,
            linestyle="--",
            linewidth=1.5,
            color=colors.get(start_point.priority, colors["unknown"]),
            alpha=0.35,
        )
        plt.text(
            start_point.time_seconds,
            plt.ylim()[1],
            f"{start_point.priority} start",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=colors.get(start_point.priority, colors["unknown"]),
            alpha=0.8,
        )

    for priority, series in sorted(grouped.items(), key=lambda item: item[0]):
        series = sorted(series, key=lambda point: point.time_seconds)
        x_values = [point.time_seconds for point in series]
        y_values = [point.throughput for point in series]
        y_values = smooth_values(y_values, smooth_window)
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            markersize=4,
            label=f"{priority} priority",
            color=colors.get(priority, colors["unknown"]),
        )

    plt.title(title)
    plt.xlabel("Time since first worker report (s)")
    plt.ylabel("Throughput")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot TGS throughput over time from worker logs.")
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("config/test_tgs.csv"),
        help="Trace CSV used to map job IDs to priorities.",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        nargs="*",
        default=[Path("backup_logs/test_tgs.log"), Path("job_logs")],
        help="Worker log files, directories, or glob patterns to parse.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tgs_throughput.png"),
        help="Where to write the plot image.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TGS throughput over time",
        help="Plot title.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving-average window for smoothing throughput lines; use 1 to disable.",
    )
    args = parser.parse_args()

    priorities = load_priorities(args.trace)
    log_paths = expand_log_inputs(args.logs)
    points = parse_worker_logs(log_paths, priorities)
    start_points = parse_start_times(log_paths, priorities)

    if not points:
        raise SystemExit("No worker report lines found in the supplied log files. Try backup_logs/test_tgs.log.")

    if not start_points:
        raise SystemExit("No trainer start lines found in the supplied log files. Try job_logs/*.txt.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_points(points, start_points, args.output, args.title, args.smooth_window)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
