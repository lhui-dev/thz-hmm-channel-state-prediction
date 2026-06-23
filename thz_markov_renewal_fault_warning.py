"""
Simulation for:
Markov renewal process based fault warning in THz industrial IoT.

Run:
    python thz_markov_renewal_fault_warning.py

Outputs:
    - thz_markov_renewal_timeseries.csv
    - thz_state_timeline.png
    - thz_metrics.png
    - thz_fault_risk.png
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATE_NAMES = ["Normal", "Minor abnormal", "Severe abnormal", "Fault/repair"]
NORMAL = 0
MINOR = 1
SEVERE = 2
FAULT = 3


@dataclass
class Interval:
    start: float
    end: float
    state: int


def sample_sojourn_time(rng: np.random.Generator, state: int) -> float:
    """Sample holding time for each state in a Markov renewal process."""
    if state == NORMAL:
        # Long stable operation. Weibull can describe aging-related degradation.
        return 80.0 * rng.weibull(2.2) + 20.0
    if state == MINOR:
        return rng.exponential(35.0) + 5.0
    if state == SEVERE:
        return rng.gamma(shape=2.0, scale=8.0) + 3.0
    if state == FAULT:
        # Repair duration.
        return rng.lognormal(mean=2.7, sigma=0.35)
    raise ValueError(f"Unknown state: {state}")


def next_state(rng: np.random.Generator, state: int, transition: np.ndarray) -> int:
    return int(rng.choice(len(STATE_NAMES), p=transition[state]))


def simulate_markov_renewal(
    rng: np.random.Generator,
    transition: np.ndarray,
    horizon: float,
    initial_state: int = NORMAL,
) -> list[Interval]:
    """Generate continuous-time state intervals."""
    intervals: list[Interval] = []
    t = 0.0
    state = initial_state

    while t < horizon:
        holding = sample_sojourn_time(rng, state)
        end = min(t + holding, horizon)
        intervals.append(Interval(t, end, state))
        t = end
        if t >= horizon:
            break
        state = next_state(rng, state, transition)

    return intervals


def discretize_intervals(intervals: list[Interval], dt: float, horizon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0.0, horizon + 1e-9, dt)
    states = np.zeros_like(times, dtype=int)
    dwell = np.zeros_like(times, dtype=float)

    idx = 0
    for k, t in enumerate(times):
        while idx < len(intervals) - 1 and t >= intervals[idx].end:
            idx += 1
        interval = intervals[idx]
        states[k] = interval.state
        dwell[k] = t - interval.start

    return times, states, dwell


def generate_thz_observations(
    rng: np.random.Generator,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate THz link observations: SNR, packet loss and latency."""
    snr_mean = np.array([28.0, 20.0, 11.0, 2.0])
    snr_std = np.array([1.8, 2.8, 3.5, 1.2])

    loss_mean = np.array([0.005, 0.025, 0.12, 0.65])
    latency_mean = np.array([1.0, 2.5, 8.0, 30.0])

    snr = rng.normal(snr_mean[states], snr_std[states])
    snr = np.clip(snr, -5.0, 35.0)

    packet_loss = rng.normal(loss_mean[states], loss_mean[states] * 0.35 + 0.003)
    packet_loss = np.clip(packet_loss, 0.0, 1.0)

    latency = rng.normal(latency_mean[states], latency_mean[states] * 0.20 + 0.3)
    latency = np.clip(latency, 0.2, 60.0)

    return snr, packet_loss, latency


def sample_residual_time(
    rng: np.random.Generator,
    state: int,
    age: float,
    max_attempts: int = 200,
) -> float:
    """Sample residual holding time conditional on the current age."""
    for _ in range(max_attempts):
        holding = sample_sojourn_time(rng, state)
        if holding >= age:
            return holding - age

    # Fallback for very old states. The residual time is small but positive.
    return rng.exponential(2.0)


def estimate_fault_probability(
    rng: np.random.Generator,
    current_state: int,
    age: float,
    transition: np.ndarray,
    lookahead: float,
    trials: int = 250,
) -> float:
    """Monte Carlo estimate of P(fault occurs within lookahead)."""
    if current_state == FAULT:
        return 1.0

    hit = 0
    for _ in range(trials):
        t = 0.0
        state = current_state
        residual = sample_residual_time(rng, state, age)

        while t < lookahead:
            t += residual
            if t > lookahead:
                break
            state = next_state(rng, state, transition)
            if state == FAULT:
                hit += 1
                break
            residual = sample_sojourn_time(rng, state)

    return hit / trials


def compute_risk_series(
    rng: np.random.Generator,
    states: np.ndarray,
    dwell: np.ndarray,
    transition: np.ndarray,
    lookahead: float,
) -> np.ndarray:
    risk = np.zeros(len(states), dtype=float)
    cache: dict[tuple[int, int], float] = {}

    for i, (state, age) in enumerate(zip(states, dwell)):
        age_bin = int(age // 5) * 5
        key = (int(state), age_bin)
        if key not in cache:
            cache[key] = estimate_fault_probability(
                rng=rng,
                current_state=int(state),
                age=float(age_bin),
                transition=transition,
                lookahead=lookahead,
            )
        risk[i] = cache[key]

    return risk


def alarm_rising_edges(
    times: np.ndarray,
    risk: np.ndarray,
    threshold: float,
    states: np.ndarray,
    min_interval: float = 0.0,
) -> np.ndarray:
    active = (risk >= threshold) & (states != FAULT)
    previous = np.r_[False, active[:-1]]
    raw_alarms = times[active & ~previous]

    alarms = []
    for alarm in raw_alarms:
        if not alarms or alarm - alarms[-1] >= min_interval:
            alarms.append(float(alarm))
    return np.array(alarms, dtype=float)


def fault_start_times(intervals: list[Interval]) -> np.ndarray:
    return np.array([item.start for item in intervals if item.state == FAULT], dtype=float)


def evaluate_warning(
    alarms: np.ndarray,
    faults: np.ndarray,
    lead_window: float,
) -> dict[str, float]:
    true_positive = 0
    missed = 0

    for ft in faults:
        has_alarm = np.any((alarms >= ft - lead_window) & (alarms < ft))
        if has_alarm:
            true_positive += 1
        else:
            missed += 1

    false_alarm = 0
    lead_times = []
    for alarm in alarms:
        future_faults = faults[(faults > alarm) & (faults <= alarm + lead_window)]
        if len(future_faults) == 0:
            false_alarm += 1
        else:
            lead_times.append(float(future_faults[0] - alarm))

    precision = true_positive / max(true_positive + false_alarm, 1)
    recall = true_positive / max(true_positive + missed, 1)
    false_alarm_rate = false_alarm / max(len(alarms), 1)
    avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

    return {
        "fault_count": float(len(faults)),
        "alarm_count": float(len(alarms)),
        "true_positive": float(true_positive),
        "missed": float(missed),
        "false_alarm": float(false_alarm),
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm_rate,
        "avg_lead_time": avg_lead_time,
    }


def save_csv(
    output_dir: Path,
    times: np.ndarray,
    states: np.ndarray,
    dwell: np.ndarray,
    snr: np.ndarray,
    packet_loss: np.ndarray,
    latency: np.ndarray,
    risk: np.ndarray,
) -> None:
    csv_path = output_dir / "thz_markov_renewal_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "state_id", "state_name", "dwell_time", "snr_db", "packet_loss", "latency_ms", "fault_risk"])
        for row in zip(times, states, dwell, snr, packet_loss, latency, risk):
            t, state, age, s, loss, delay, r = row
            writer.writerow([f"{t:.2f}", int(state), STATE_NAMES[int(state)], f"{age:.2f}", f"{s:.3f}", f"{loss:.5f}", f"{delay:.3f}", f"{r:.4f}"])


def make_plots(
    output_dir: Path,
    times: np.ndarray,
    states: np.ndarray,
    snr: np.ndarray,
    packet_loss: np.ndarray,
    latency: np.ndarray,
    risk: np.ndarray,
    threshold: float,
    alarms: np.ndarray,
    faults: np.ndarray,
) -> None:
    plt.figure(figsize=(10, 3.6))
    plt.step(times, states, where="post", linewidth=1.8)
    plt.yticks(range(len(STATE_NAMES)), STATE_NAMES)
    plt.xlabel("Time")
    plt.ylabel("Device state")
    plt.title("THz Industrial IoT Device State Evolution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "thz_state_timeline.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, snr, color="#20639b", linewidth=1.1)
    axes[0].set_ylabel("SNR (dB)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(times, packet_loss, color="#b23a48", linewidth=1.1)
    axes[1].set_ylabel("Packet loss")
    axes[1].grid(alpha=0.3)
    axes[2].plot(times, latency, color="#2a9d8f", linewidth=1.1)
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_xlabel("Time")
    axes[2].grid(alpha=0.3)
    fig.suptitle("Simulated THz Link Observations")
    fig.tight_layout()
    fig.savefig(output_dir / "thz_metrics.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(10, 4))
    plt.plot(times, risk, color="#7b2cbf", linewidth=1.6, label="Fault risk")
    plt.axhline(threshold, color="#e76f51", linestyle="--", label="Warning threshold")
    for alarm in alarms:
        plt.axvline(alarm, color="#f4a261", alpha=0.45, linewidth=1.0)
    for fault in faults:
        plt.axvline(fault, color="#111111", alpha=0.35, linewidth=1.0)
    plt.xlabel("Time")
    plt.ylabel("Risk")
    plt.ylim(-0.03, 1.03)
    plt.title("Fault Warning Risk Based on Markov Renewal Process")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "thz_fault_risk.png", dpi=180)
    plt.close()


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(2028)

    horizon = 1200.0
    dt = 1.0
    lookahead = 80.0
    warning_threshold = 0.35
    lead_window = 120.0
    min_alarm_interval = 90.0

    transition = np.array(
        [
            [0.78, 0.17, 0.04, 0.01],
            [0.18, 0.47, 0.27, 0.08],
            [0.04, 0.14, 0.42, 0.40],
            [0.80, 0.10, 0.02, 0.08],
        ],
        dtype=float,
    )

    intervals = simulate_markov_renewal(rng, transition, horizon)
    times, states, dwell = discretize_intervals(intervals, dt, horizon)
    snr, packet_loss, latency = generate_thz_observations(rng, states)
    risk = compute_risk_series(rng, states, dwell, transition, lookahead)

    alarms = alarm_rising_edges(times, risk, warning_threshold, states, min_alarm_interval)
    faults = fault_start_times(intervals)
    metrics = evaluate_warning(alarms, faults, lead_window)

    save_csv(output_dir, times, states, dwell, snr, packet_loss, latency, risk)
    make_plots(output_dir, times, states, snr, packet_loss, latency, risk, warning_threshold, alarms, faults)

    print("=== Markov renewal THz IIoT fault warning simulation ===")
    print(f"Output directory: {output_dir}")
    print(f"Fault count: {int(metrics['fault_count'])}")
    print(f"Alarm count: {int(metrics['alarm_count'])}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"False alarm rate: {metrics['false_alarm_rate']:.3f}")
    print(f"Average warning lead time: {metrics['avg_lead_time']:.2f}")


if __name__ == "__main__":
    main()
