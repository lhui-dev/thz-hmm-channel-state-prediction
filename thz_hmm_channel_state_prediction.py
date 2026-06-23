"""
HMM-based THz channel state recognition and one-step prediction.

Topic:
    基于隐马尔可夫模型的太赫兹信道状态识别与预测研究

Run:
    python thz_hmm_channel_state_prediction.py

Outputs:
    - thz_hmm_timeseries.csv
    - thz_hmm_summary.txt
    - thz_hmm_state_recognition.png
    - thz_hmm_observations.png
    - thz_hmm_prediction_risk.png
    - thz_hmm_confusion_matrix.png

The implementation intentionally avoids third-party HMM libraries. Viterbi
decoding and forward filtering are implemented directly so the code can be
explained in a course report appendix.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATE_NAMES = ["LoS", "NLoS", "Light blockage", "Severe blockage"]
OBS_NAMES = ["Excellent", "Good", "Weak", "Outage"]
BLOCKAGE_RISK_THRESHOLD = 0.30

LOS = 0
NLOS = 1
LIGHT_BLOCK = 2
SEVERE_BLOCK = 3


@dataclass(frozen=True)
class HMMModel:
    initial: np.ndarray
    transition: np.ndarray
    emission: np.ndarray


def normalize(prob: np.ndarray) -> np.ndarray:
    total = np.sum(prob)
    if total <= 0:
        return np.ones_like(prob) / len(prob)
    return prob / total


def build_thz_hmm_model() -> HMMModel:
    """Build an HMM model reflecting typical THz channel behavior."""
    initial = np.array([0.74, 0.16, 0.08, 0.02], dtype=float)

    # Rows: current hidden state; columns: next hidden state.
    # THz LoS links are usually stable, while blockage states are more likely
    # to remain abnormal or recover through beam adjustment.
    transition = np.array(
        [
            [0.86, 0.09, 0.04, 0.01],
            [0.18, 0.66, 0.12, 0.04],
            [0.10, 0.24, 0.52, 0.14],
            [0.18, 0.12, 0.28, 0.42],
        ],
        dtype=float,
    )

    # Rows: hidden channel state; columns: observed quality symbol.
    # Observations are discretized from SNR/RSS/BER measurements.
    emission = np.array(
        [
            [0.780, 0.180, 0.035, 0.005],
            [0.120, 0.620, 0.210, 0.050],
            [0.020, 0.150, 0.620, 0.210],
            [0.005, 0.040, 0.180, 0.775],
        ],
        dtype=float,
    )

    return HMMModel(
        initial=normalize(initial),
        transition=transition / transition.sum(axis=1, keepdims=True),
        emission=emission / emission.sum(axis=1, keepdims=True),
    )


def generate_hidden_states_and_observations(
    rng: np.random.Generator,
    model: HMMModel,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate hidden THz channel states and discrete observations."""
    states = np.zeros(steps, dtype=int)
    observations = np.zeros(steps, dtype=int)

    states[0] = int(rng.choice(len(STATE_NAMES), p=model.initial))
    observations[0] = int(rng.choice(len(OBS_NAMES), p=model.emission[states[0]]))

    for t in range(1, steps):
        states[t] = int(rng.choice(len(STATE_NAMES), p=model.transition[states[t - 1]]))
        observations[t] = int(rng.choice(len(OBS_NAMES), p=model.emission[states[t]]))

    return states, observations


def generate_thz_measurements(
    rng: np.random.Generator,
    states: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate continuous THz link measurements for visualization."""
    snr_mean = np.array([31.0, 22.0, 13.0, 4.0])
    snr_std = np.array([2.0, 3.0, 3.8, 2.0])

    rss_mean = np.array([-42.0, -56.0, -70.0, -84.0])
    rss_std = np.array([2.0, 3.0, 4.0, 3.0])

    log10_ber_mean = np.array([-6.2, -4.6, -2.8, -1.1])
    log10_ber_std = np.array([0.30, 0.42, 0.48, 0.35])

    snr = rng.normal(snr_mean[states], snr_std[states])
    rss = rng.normal(rss_mean[states], rss_std[states])
    log10_ber = rng.normal(log10_ber_mean[states], log10_ber_std[states])
    ber = np.clip(10.0**log10_ber, 1e-8, 0.9)

    # Observation quality adds a small discrete penalty/bonus to throughput.
    quality_factor = np.array([1.00, 0.78, 0.42, 0.08])
    spectral_efficiency = np.log2(1.0 + np.maximum(10.0 ** (snr / 10.0), 0.0))
    throughput = 12.0 * spectral_efficiency * quality_factor[observations]
    throughput += rng.normal(0.0, 3.0, size=len(states))
    throughput = np.clip(throughput, 0.0, 110.0)

    return snr, rss, ber, throughput


def viterbi_decode(model: HMMModel, observations: np.ndarray) -> np.ndarray:
    """Most likely hidden state sequence given observations."""
    eps = 1e-12
    log_pi = np.log(model.initial + eps)
    log_a = np.log(model.transition + eps)
    log_b = np.log(model.emission + eps)

    steps = len(observations)
    state_count = len(model.initial)
    delta = np.zeros((steps, state_count), dtype=float)
    psi = np.zeros((steps, state_count), dtype=int)

    delta[0] = log_pi + log_b[:, observations[0]]

    for t in range(1, steps):
        for j in range(state_count):
            candidates = delta[t - 1] + log_a[:, j]
            psi[t, j] = int(np.argmax(candidates))
            delta[t, j] = candidates[psi[t, j]] + log_b[j, observations[t]]

    decoded = np.zeros(steps, dtype=int)
    decoded[-1] = int(np.argmax(delta[-1]))

    for t in range(steps - 2, -1, -1):
        decoded[t] = psi[t + 1, decoded[t + 1]]

    return decoded


def forward_filter_and_predict(
    model: HMMModel,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter current state posterior and predict the next hidden state.

    predicted_next[t] is the estimated state at t+1 using observations up to t.
    """
    steps = len(observations)
    state_count = len(model.initial)

    posterior = np.zeros((steps, state_count), dtype=float)
    next_state_prob = np.zeros((steps, state_count), dtype=float)
    predicted_next = np.full(steps, -1, dtype=int)

    alpha = normalize(model.initial * model.emission[:, observations[0]])
    posterior[0] = alpha
    next_state_prob[0] = alpha @ model.transition
    predicted_next[0] = int(np.argmax(next_state_prob[0]))

    for t in range(1, steps):
        alpha = normalize((alpha @ model.transition) * model.emission[:, observations[t]])
        posterior[t] = alpha
        next_state_prob[t] = alpha @ model.transition
        predicted_next[t] = int(np.argmax(next_state_prob[t]))

    predicted_next[-1] = -1
    next_state_prob[-1] = np.nan
    return posterior, next_state_prob, predicted_next


def confusion_matrix(true_states: np.ndarray, decoded_states: np.ndarray, state_count: int) -> np.ndarray:
    matrix = np.zeros((state_count, state_count), dtype=int)
    for true_state, decoded_state in zip(true_states, decoded_states):
        matrix[int(true_state), int(decoded_state)] += 1
    return matrix


def compute_rolling_accuracy(correct: np.ndarray, window: int) -> np.ndarray:
    values = correct.astype(float)
    rolling = np.zeros_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        rolling[i] = float(np.mean(values[start : i + 1]))
    return rolling


def evaluate_results(
    states: np.ndarray,
    decoded: np.ndarray,
    predicted_next: np.ndarray,
    next_state_prob: np.ndarray,
) -> dict[str, float]:
    recognition_accuracy = float(np.mean(states == decoded))

    valid = predicted_next[:-1] >= 0
    prediction_accuracy = float(np.mean(predicted_next[:-1][valid] == states[1:][valid]))

    true_block_next = np.isin(states[1:], [LIGHT_BLOCK, SEVERE_BLOCK])
    pred_block_next = np.isin(predicted_next[:-1], [LIGHT_BLOCK, SEVERE_BLOCK])

    block_tp = int(np.sum(true_block_next & pred_block_next))
    block_fp = int(np.sum(~true_block_next & pred_block_next))
    block_fn = int(np.sum(true_block_next & ~pred_block_next))

    block_precision = block_tp / max(block_tp + block_fp, 1)
    block_recall = block_tp / max(block_tp + block_fn, 1)

    next_blockage_prob = next_state_prob[:-1, LIGHT_BLOCK] + next_state_prob[:-1, SEVERE_BLOCK]
    risk_warning = next_blockage_prob >= BLOCKAGE_RISK_THRESHOLD
    risk_tp = int(np.sum(true_block_next & risk_warning))
    risk_fp = int(np.sum(~true_block_next & risk_warning))
    risk_fn = int(np.sum(true_block_next & ~risk_warning))
    risk_tn = int(np.sum(~true_block_next & ~risk_warning))

    risk_accuracy = (risk_tp + risk_tn) / max(len(true_block_next), 1)
    risk_precision = risk_tp / max(risk_tp + risk_fp, 1)
    risk_recall = risk_tp / max(risk_tp + risk_fn, 1)
    risk_false_alarm_rate = risk_fp / max(risk_fp + risk_tn, 1)

    severe_mask = states == SEVERE_BLOCK
    severe_recognition_accuracy = float(np.mean(decoded[severe_mask] == SEVERE_BLOCK)) if np.any(severe_mask) else 0.0

    return {
        "recognition_accuracy": recognition_accuracy,
        "prediction_accuracy": prediction_accuracy,
        "blockage_prediction_precision": block_precision,
        "blockage_prediction_recall": block_recall,
        "risk_threshold": BLOCKAGE_RISK_THRESHOLD,
        "risk_warning_accuracy": risk_accuracy,
        "risk_warning_precision": risk_precision,
        "risk_warning_recall": risk_recall,
        "risk_warning_false_alarm_rate": risk_false_alarm_rate,
        "severe_blockage_recognition_accuracy": severe_recognition_accuracy,
        "los_ratio": float(np.mean(states == LOS)),
        "nlos_ratio": float(np.mean(states == NLOS)),
        "light_blockage_ratio": float(np.mean(states == LIGHT_BLOCK)),
        "severe_blockage_ratio": float(np.mean(states == SEVERE_BLOCK)),
    }


def save_csv(
    output_dir: Path,
    times: np.ndarray,
    states: np.ndarray,
    observations: np.ndarray,
    snr: np.ndarray,
    rss: np.ndarray,
    ber: np.ndarray,
    throughput: np.ndarray,
    decoded: np.ndarray,
    posterior: np.ndarray,
    predicted_next: np.ndarray,
    next_state_prob: np.ndarray,
) -> None:
    csv_path = output_dir / "thz_hmm_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time",
                "true_state_id",
                "true_state",
                "observation_id",
                "observation",
                "snr_db",
                "rss_dbm",
                "ber",
                "throughput_gbps",
                "decoded_state_id",
                "decoded_state",
                "p_los",
                "p_nlos",
                "p_light_blockage",
                "p_severe_blockage",
                "predicted_next_state_id",
                "predicted_next_state",
                "next_blockage_probability",
            ]
        )

        for i in range(len(times)):
            pred = int(predicted_next[i])
            pred_name = "" if pred < 0 else STATE_NAMES[pred]
            next_block_prob = np.nan
            if i < len(times) - 1:
                next_block_prob = float(next_state_prob[i, LIGHT_BLOCK] + next_state_prob[i, SEVERE_BLOCK])
            writer.writerow(
                [
                    f"{times[i]:.0f}",
                    int(states[i]),
                    STATE_NAMES[int(states[i])],
                    int(observations[i]),
                    OBS_NAMES[int(observations[i])],
                    f"{snr[i]:.3f}",
                    f"{rss[i]:.3f}",
                    f"{ber[i]:.8f}",
                    f"{throughput[i]:.3f}",
                    int(decoded[i]),
                    STATE_NAMES[int(decoded[i])],
                    f"{posterior[i, LOS]:.5f}",
                    f"{posterior[i, NLOS]:.5f}",
                    f"{posterior[i, LIGHT_BLOCK]:.5f}",
                    f"{posterior[i, SEVERE_BLOCK]:.5f}",
                    "" if pred < 0 else pred,
                    pred_name,
                    "" if np.isnan(next_block_prob) else f"{next_block_prob:.5f}",
                ]
            )


def save_summary(
    output_dir: Path,
    model: HMMModel,
    metrics: dict[str, float],
    matrix: np.ndarray,
) -> None:
    summary_path = output_dir / "thz_hmm_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("HMM-based THz channel state recognition and prediction\n")
        file.write("=======================================================\n\n")
        file.write("Hidden states:\n")
        for i, name in enumerate(STATE_NAMES):
            file.write(f"  {i}: {name}\n")

        file.write("\nObservation symbols:\n")
        for i, name in enumerate(OBS_NAMES):
            file.write(f"  {i}: {name}\n")

        file.write("\nInitial distribution pi:\n")
        file.write(np.array2string(model.initial, precision=3) + "\n")

        file.write("\nTransition matrix A:\n")
        file.write(np.array2string(model.transition, precision=3) + "\n")

        file.write("\nEmission matrix B:\n")
        file.write(np.array2string(model.emission, precision=3) + "\n")

        file.write("\nMetrics:\n")
        for key, value in metrics.items():
            file.write(f"  {key}: {value:.4f}\n")

        file.write("\nConfusion matrix, rows=true state, columns=decoded state:\n")
        file.write(str(matrix) + "\n")


def plot_state_recognition(
    output_dir: Path,
    times: np.ndarray,
    states: np.ndarray,
    decoded: np.ndarray,
) -> None:
    plt.figure(figsize=(11, 4.4))
    plt.step(times, states + 0.06, where="post", linewidth=1.6, label="True state")
    plt.step(times, decoded - 0.06, where="post", linewidth=1.3, alpha=0.85, label="Viterbi decoded")
    plt.yticks(range(len(STATE_NAMES)), STATE_NAMES)
    plt.xlabel("Time slot")
    plt.ylabel("Channel state")
    plt.title("THz Channel State Recognition by HMM")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "thz_hmm_state_recognition.png", dpi=180)
    plt.close()


def plot_observations(
    output_dir: Path,
    times: np.ndarray,
    snr: np.ndarray,
    rss: np.ndarray,
    ber: np.ndarray,
    throughput: np.ndarray,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(times, snr, color="#20639b", linewidth=1.0)
    axes[0].set_ylabel("SNR (dB)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, rss, color="#6a4c93", linewidth=1.0)
    axes[1].set_ylabel("RSS (dBm)")
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(times, ber, color="#b23a48", linewidth=1.0)
    axes[2].set_ylabel("BER")
    axes[2].grid(alpha=0.3)

    axes[3].plot(times, throughput, color="#2a9d8f", linewidth=1.0)
    axes[3].set_ylabel("Gbps")
    axes[3].set_xlabel("Time slot")
    axes[3].grid(alpha=0.3)

    fig.suptitle("Simulated THz Link Measurements")
    fig.tight_layout()
    fig.savefig(output_dir / "thz_hmm_observations.png", dpi=180)
    plt.close(fig)


def plot_prediction_risk(
    output_dir: Path,
    times: np.ndarray,
    states: np.ndarray,
    posterior: np.ndarray,
    next_state_prob: np.ndarray,
    predicted_next: np.ndarray,
) -> None:
    current_block_prob = posterior[:, LIGHT_BLOCK] + posterior[:, SEVERE_BLOCK]
    next_block_prob = next_state_prob[:, LIGHT_BLOCK] + next_state_prob[:, SEVERE_BLOCK]
    next_block_prob[-1] = np.nan

    correct_next = (predicted_next[:-1] == states[1:]).astype(float)
    rolling_accuracy = compute_rolling_accuracy(correct_next, window=40)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(times, current_block_prob, color="#7b2cbf", linewidth=1.4, label="Current blockage posterior")
    axes[0].plot(times, next_block_prob, color="#f77f00", linewidth=1.2, label="Next-slot blockage probability")
    axes[0].axhline(
        BLOCKAGE_RISK_THRESHOLD,
        color="#d62828",
        linestyle="--",
        linewidth=1.0,
        label="Risk threshold",
    )
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times[:-1], rolling_accuracy, color="#264653", linewidth=1.4)
    axes[1].set_ylabel("Rolling accuracy")
    axes[1].set_xlabel("Time slot")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(alpha=0.3)

    fig.suptitle("One-step Channel State Prediction")
    fig.tight_layout()
    fig.savefig(output_dir / "thz_hmm_prediction_risk.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(output_dir: Path, matrix: np.ndarray) -> None:
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, np.maximum(row_sum, 1))

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(STATE_NAMES)), STATE_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(len(STATE_NAMES)), STATE_NAMES)
    ax.set_xlabel("Decoded state")
    ax.set_ylabel("True state")
    ax.set_title("HMM Recognition Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]}\n{normalized[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.55 else "#222222",
                fontsize=9,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "thz_hmm_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(2026)

    steps = 900
    times = np.arange(steps)
    model = build_thz_hmm_model()

    states, observations = generate_hidden_states_and_observations(rng, model, steps)
    snr, rss, ber, throughput = generate_thz_measurements(rng, states, observations)

    decoded = viterbi_decode(model, observations)
    posterior, next_state_prob, predicted_next = forward_filter_and_predict(model, observations)

    matrix = confusion_matrix(states, decoded, len(STATE_NAMES))
    metrics = evaluate_results(states, decoded, predicted_next, next_state_prob)

    save_csv(
        output_dir=output_dir,
        times=times,
        states=states,
        observations=observations,
        snr=snr,
        rss=rss,
        ber=ber,
        throughput=throughput,
        decoded=decoded,
        posterior=posterior,
        predicted_next=predicted_next,
        next_state_prob=next_state_prob,
    )
    save_summary(output_dir, model, metrics, matrix)
    plot_state_recognition(output_dir, times, states, decoded)
    plot_observations(output_dir, times, snr, rss, ber, throughput)
    plot_prediction_risk(output_dir, times, states, posterior, next_state_prob, predicted_next)
    plot_confusion_matrix(output_dir, matrix)

    print("=== HMM THz channel state recognition and prediction ===")
    print(f"Output directory: {output_dir}")
    print(f"Recognition accuracy: {metrics['recognition_accuracy']:.3f}")
    print(f"One-step prediction accuracy: {metrics['prediction_accuracy']:.3f}")
    print(f"Blockage prediction precision: {metrics['blockage_prediction_precision']:.3f}")
    print(f"Blockage prediction recall: {metrics['blockage_prediction_recall']:.3f}")
    print(f"Risk-warning accuracy: {metrics['risk_warning_accuracy']:.3f}")
    print(f"Risk-warning recall: {metrics['risk_warning_recall']:.3f}")
    print(f"Severe blockage recognition accuracy: {metrics['severe_blockage_recognition_accuracy']:.3f}")


if __name__ == "__main__":
    main()
