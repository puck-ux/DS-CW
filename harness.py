"""Part B Test Harness: seeded, config-driven multi-worker benchmark and fault injector."""

import argparse
import json
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from config import LR, NUM_KEYS, SEED, STEPS
from ps_client import PSClient


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class MetricsCollector:
    lock: threading.Lock = field(default_factory=threading.Lock)

    # End-to-end Pull/Wait and Push/Wait operation latencies.
    latencies_ms: list = field(default_factory=list)

    # Estimated application payload sizes, not total network wire bytes.
    bytes_sent: int = 0
    bytes_received: int = 0

    completed_operations: int = 0
    completed_iterations: int = 0
    dropped_iterations: int = 0

    convergence_history: list = field(default_factory=list)
    scenario_events: list = field(default_factory=list)
    worker_errors: list = field(default_factory=list)

    def record_operation(self, latency_ms, sent, received):
        with self.lock:
            self.latencies_ms.append(float(latency_ms))
            self.bytes_sent += int(sent)
            self.bytes_received += int(received)
            self.completed_operations += 1

    def record_iteration(self):
        with self.lock:
            self.completed_iterations += 1

    def record_drop(self):
        with self.lock:
            self.dropped_iterations += 1

    def record_convergence(self, elapsed_s, l2_error, rmse):
        with self.lock:
            self.convergence_history.append({
                "elapsed_sec": float(elapsed_s),
                "l2_error": float(l2_error),
                "rmse": float(rmse),
            })

    def record_scenario_event(self, description, handling_ms):
        with self.lock:
            self.scenario_events.append({
                "event": description,
                "handling_ms": float(handling_ms),
            })

    def record_worker_error(self, worker_id, error):
        with self.lock:
            self.worker_errors.append({
                "worker_id": int(worker_id),
                "error": str(error),
            })


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

def sample_zipf_keys(
    rng: np.random.Generator,
    num_keys: int,
    sample_size: int,
    alpha: float,
) -> np.ndarray:
    """Sample unique keys with a Zipf-like probability distribution."""

    ranks = np.arange(1, num_keys + 1, dtype=np.float64)
    probabilities = 1.0 / np.power(ranks, alpha)
    probabilities /= probabilities.sum()

    size = min(sample_size, num_keys)

    return np.sort(
        rng.choice(
            num_keys,
            size=size,
            replace=False,
            p=probabilities,
        )
    )


# ---------------------------------------------------------------------------
# Scenario controller
# ---------------------------------------------------------------------------

class ScenarioController:
    """Controls worker termination, worker-side delay and iteration dropping."""

    def __init__(self, initial_workers):
        self.lock = threading.Lock()
        self.killed_workers = set()
        self.extra_delay_ms = 0.0
        self.drop_prob = 0.0
        self.next_worker_id = initial_workers

    def is_killed(self, worker_id):
        with self.lock:
            return worker_id in self.killed_workers

    def kill_worker(self, worker_id):
        with self.lock:
            self.killed_workers.add(worker_id)

    def set_delay(self, delay_ms):
        with self.lock:
            self.extra_delay_ms = float(delay_ms)

    def set_drop_prob(self, probability):
        with self.lock:
            self.drop_prob = float(probability)

    def get_faults(self):
        with self.lock:
            return self.extra_delay_ms, self.drop_prob

    def allocate_worker_id(self):
        with self.lock:
            worker_id = self.next_worker_id
            self.next_worker_id += 1
            return worker_id


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker_loop(
    worker_id,
    cfg,
    w_star,
    metrics,
    controller,
    start_time,
    straggler_ids,
):
    """Run one synthetic parameter-server worker."""

    rng = np.random.default_rng(cfg["seed"] + worker_id)

    iterations = cfg["iterations"]
    mode = cfg["mode"]

    compute_ms = cfg["compute_ms"]

    if worker_id in straggler_ids:
        compute_ms *= cfg["straggler_multiplier"]

    # Precompute range-based key selection.
    if mode == "range":
        start = cfg["range_start"]
        length = cfg["range_length"]
        static_keys = np.arange(start, start + length)
    elif mode == "full":
        static_keys = np.arange(NUM_KEYS)
    else:
        static_keys = None

    try:
        with PSClient(cfg["address"]) as ps:

            for iteration in range(1, iterations + 1):

                # Kill events take effect at the next iteration boundary.
                if controller.is_killed(worker_id):
                    print(
                        f"[scenario] worker {worker_id} "
                        f"terminated at iteration {iteration}"
                    )
                    return

                extra_delay_ms, drop_prob = controller.get_faults()

                # This simulates a skipped worker iteration, not a lost RPC.
                if rng.random() < drop_prob:
                    metrics.record_drop()
                    continue

                # This is worker-side delay, not network-level delay.
                if extra_delay_ms > 0:
                    time.sleep(extra_delay_ms / 1000.0)

                # -------------------------------------------------------
                # Select keys
                # -------------------------------------------------------

                if mode == "zipf":
                    keys = sample_zipf_keys(
                        rng,
                        NUM_KEYS,
                        cfg["zipf_sample_size"],
                        cfg["zipf_alpha"],
                    )

                    pull_kwargs = {"keys": keys}
                    pull_request_bytes = len(keys) * 8

                elif mode == "range":
                    keys = static_keys
                    pull_kwargs = {
                        "start": cfg["range_start"],
                        "length": cfg["range_length"],
                    }
                    pull_request_bytes = 16

                else:
                    keys = static_keys
                    pull_kwargs = {
                        "start": 0,
                        "length": NUM_KEYS,
                    }
                    pull_request_bytes = 16

                # -------------------------------------------------------
                # 1. Pull
                # -------------------------------------------------------

                t0 = time.perf_counter()

                pull_handle = ps.pull(**pull_kwargs)
                weights = ps.wait(pull_handle)

                pull_ms = (time.perf_counter() - t0) * 1000.0

                # Approximate payload sizes:
                # key indices/range request and returned double weights.
                metrics.record_operation(
                    latency_ms=pull_ms,
                    sent=pull_request_bytes,
                    received=len(weights) * 8,
                )

                # -------------------------------------------------------
                # 2. Simulated local computation
                # -------------------------------------------------------

                if cfg["hetero_compute"]:
                    sleep_ms = rng.exponential(scale=compute_ms)
                else:
                    sleep_ms = compute_ms

                time.sleep(sleep_ms / 1000.0)

                # Synthetic gradient towards the shared target.
                gradients = weights - w_star[keys]

                # -------------------------------------------------------
                # 3. Push
                # -------------------------------------------------------

                t1 = time.perf_counter()

                push_handle = ps.push(
                    gradients,
                    deps=[pull_handle],
                )

                ps.wait(push_handle)

                push_ms = (time.perf_counter() - t1) * 1000.0

                metrics.record_operation(
                    latency_ms=push_ms,
                    sent=len(gradients) * 8,
                    received=0,
                )

                metrics.record_iteration()

                # -------------------------------------------------------
                # 4. Periodic convergence evaluation
                # -------------------------------------------------------

                if (
                    worker_id == 0
                    and (
                        iteration % cfg["eval_every"] == 0
                        or iteration == iterations
                    )
                ):
                    all_weights = ps.wait(
                        ps.pull(start=0, length=NUM_KEYS)
                    )

                    l2_error = float(
                        np.linalg.norm(all_weights - w_star)
                    )

                    rmse = l2_error / np.sqrt(NUM_KEYS)

                    metrics.record_convergence(
                        time.perf_counter() - start_time,
                        l2_error,
                        rmse,
                    )

    except Exception as error:
        metrics.record_worker_error(worker_id, error)
        print(f"[worker {worker_id}] error: {error}")


# ---------------------------------------------------------------------------
# Scenario script
# ---------------------------------------------------------------------------

def run_scenario_script(
    script,
    cfg,
    w_star,
    metrics,
    controller,
    threads,
    start_time,
    straggler_ids,
):
    """Execute time-triggered kill, add, delay and drop events."""

    for event in sorted(script, key=lambda item: item["at_sec"]):

        wait_s = event["at_sec"] - (
            time.perf_counter() - start_time
        )

        if wait_s > 0:
            time.sleep(wait_s)

        event_start = time.perf_counter()
        action = event["action"]

        if action == "kill":
            worker_id = event["worker_id"]
            controller.kill_worker(worker_id)
            description = f"kill worker {worker_id}"

        elif action == "add":
            worker_id = controller.allocate_worker_id()

            thread = threading.Thread(
                target=worker_loop,
                args=(
                    worker_id,
                    cfg,
                    w_star,
                    metrics,
                    controller,
                    start_time,
                    straggler_ids,
                ),
                name=f"worker-{worker_id}",
            )

            thread.start()
            threads.append(thread)

            description = f"add worker {worker_id}"

        elif action == "delay":
            delay_ms = float(event.get("delay_ms", 0.0))
            controller.set_delay(delay_ms)
            description = f"set worker-side delay to {delay_ms} ms"

        elif action == "drop":
            probability = float(event.get("drop_prob", 0.0))
            controller.set_drop_prob(probability)
            description = f"set iteration drop probability to {probability}"

        else:
            print(f"[scenario] unknown action: {action}")
            continue

        handling_ms = (
            time.perf_counter() - event_start
        ) * 1000.0

        metrics.record_scenario_event(
            description,
            handling_ms,
        )

        print(
            f"[scenario @ {event['at_sec']:.2f}s] "
            f"{description} "
            f"(handled in {handling_ms:.2f} ms)"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config():
    parser = argparse.ArgumentParser(
        description="Part B Parameter Server Test Harness"
    )

    parser.add_argument("--config", type=str)
    parser.add_argument("--address", type=str, default="localhost:8000")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=STEPS)

    parser.add_argument(
        "--mode",
        choices=["full", "range", "zipf"],
        default="zipf",
    )

    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "LENGTH"),
        default=[0, NUM_KEYS],
    )

    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument(
        "--zipf-sample-size",
        type=int,
        default=min(20, NUM_KEYS),
    )

    parser.add_argument("--compute-ms", type=float, default=10.0)
    parser.add_argument("--hetero-compute", action="store_true")
    parser.add_argument("--straggler-fraction", type=float, default=0.0)
    parser.add_argument("--straggler-multiplier", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--output-json", type=str)

    args = parser.parse_args()

    cfg = {
        "address": args.address,
        "seed": args.seed,
        "workers": args.workers,
        "iterations": args.iterations,
        "mode": args.mode,
        "range_start": args.range[0],
        "range_length": args.range[1],
        "zipf_alpha": args.zipf_alpha,
        "zipf_sample_size": args.zipf_sample_size,
        "compute_ms": args.compute_ms,
        "hetero_compute": args.hetero_compute,
        "straggler_fraction": args.straggler_fraction,
        "straggler_multiplier": args.straggler_multiplier,
        "eval_every": args.eval_every,
        "scenario": [],
        "output_json": args.output_json,
    }

    # JSON configuration values override command-line defaults.
    if args.config:
        with open(args.config, "r", encoding="utf-8") as file:
            cfg.update(json.load(file))

    validate_config(cfg)

    return cfg


def validate_config(cfg):
    """Reject invalid benchmark configurations."""

    if cfg["workers"] < 1:
        raise ValueError("workers must be at least 1")

    if cfg["iterations"] < 1:
        raise ValueError("iterations must be at least 1")

    if cfg["mode"] not in ("full", "range", "zipf"):
        raise ValueError("mode must be full, range or zipf")

    if cfg["compute_ms"] < 0:
        raise ValueError("compute_ms cannot be negative")

    if not 0 <= cfg["straggler_fraction"] <= 1:
        raise ValueError("straggler_fraction must be between 0 and 1")

    if cfg["straggler_multiplier"] < 1:
        raise ValueError("straggler_multiplier must be at least 1")

    if cfg["zipf_alpha"] <= 0:
        raise ValueError("zipf_alpha must be positive")

    if cfg["zipf_sample_size"] < 1:
        raise ValueError("zipf_sample_size must be positive")

    if cfg["eval_every"] < 1:
        raise ValueError("eval_every must be at least 1")

    start = cfg["range_start"]
    length = cfg["range_length"]

    if start < 0 or length < 1 or start + length > NUM_KEYS:
        raise ValueError("Configured range is outside the available keys")

    for event in cfg["scenario"]:
        if event["at_sec"] < 0:
            raise ValueError("Scenario event times cannot be negative")

        if event["action"] not in ("kill", "add", "delay", "drop"):
            raise ValueError(f"Unknown scenario action: {event['action']}")

        if event["action"] == "kill":
            worker_id = event["worker_id"]
            if not 0 <= worker_id < cfg["workers"]:
                raise ValueError(
                    "kill events must target an initially configured worker"
                )

        if event["action"] == "delay":
            if float(event.get("delay_ms", 0.0)) < 0:
                raise ValueError("delay_ms cannot be negative")

        if event["action"] == "drop":
            probability = float(event.get("drop_prob", 0.0))
            if not 0 <= probability <= 1:
                raise ValueError("drop_prob must be between 0 and 1")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    # Shared deterministic target used by every worker.
    w_star = np.random.default_rng(
        cfg["seed"]
    ).normal(size=NUM_KEYS)

    metrics = MetricsCollector()

    controller = ScenarioController(
        initial_workers=cfg["workers"]
    )

    # Select a reproducible subset of initial workers as stragglers.
    num_stragglers = round(
        cfg["workers"] * cfg["straggler_fraction"]
    )

    straggler_rng = np.random.default_rng(
        cfg["seed"] + cfg["workers"]
    )

    straggler_ids = set(
        straggler_rng.choice(
            cfg["workers"],
            size=num_stragglers,
            replace=False,
        ).tolist()
    )

    print(f"Workers: {cfg['workers']}")
    print(f"Straggler workers: {sorted(straggler_ids)}")
    print(f"Workload mode: {cfg['mode']}")

    start_time = time.perf_counter()

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(
                worker_id,
                cfg,
                w_star,
                metrics,
                controller,
                start_time,
                straggler_ids,
            ),
            name=f"worker-{worker_id}",
        )
        for worker_id in range(cfg["workers"])
    ]

    for thread in threads:
        thread.start()

    # Run scenario events concurrently with the workers.
    if cfg["scenario"]:
        scenario_thread = threading.Thread(
            target=run_scenario_script,
            args=(
                cfg["scenario"],
                cfg,
                w_star,
                metrics,
                controller,
                threads,
                start_time,
                straggler_ids,
            ),
            name="scenario-controller",
        )

        scenario_thread.start()
        scenario_thread.join()

    # Also joins any workers added by the scenario script.
    for thread in threads:
        thread.join()

    elapsed_s = time.perf_counter() - start_time

    # ---------------------------------------------------------------
    # Final evaluation against the correctness oracle
    # ---------------------------------------------------------------

    final_l2 = None
    final_rmse = None

    try:
        with PSClient(cfg["address"]) as ps:
            final_weights = ps.wait(
                ps.pull(start=0, length=NUM_KEYS)
            )

        final_l2 = float(
            np.linalg.norm(final_weights - w_star)
        )

        final_rmse = final_l2 / np.sqrt(NUM_KEYS)

    except Exception as error:
        print(f"[evaluation] final evaluation failed: {error}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    with metrics.lock:
        latency_samples = list(metrics.latencies_ms)

        completed_operations = metrics.completed_operations
        completed_iterations = metrics.completed_iterations
        dropped_iterations = metrics.dropped_iterations

        bytes_sent = metrics.bytes_sent
        bytes_received = metrics.bytes_received

        convergence_history = list(metrics.convergence_history)
        scenario_events = list(metrics.scenario_events)
        worker_errors = list(metrics.worker_errors)

    if latency_samples:
        mean_latency = float(np.mean(latency_samples))
        p99_latency = float(np.percentile(latency_samples, 99))
    else:
        mean_latency = None
        p99_latency = None

    summary = {
        "config": cfg,
        "elapsed_sec": round(elapsed_s, 4),

        # Pull/Wait and Push/Wait operation cycles per second.
        "operation_throughput_per_sec": round(
            completed_operations / max(elapsed_s, 1e-9),
            2,
        ),

        # Successfully completed worker iterations per second.
        "iteration_throughput_per_sec": round(
            completed_iterations / max(elapsed_s, 1e-9),
            2,
        ),

        "completed_operations": completed_operations,
        "completed_iterations": completed_iterations,
        "dropped_iterations": dropped_iterations,

        "mean_operation_latency_ms": (
            round(mean_latency, 3)
            if mean_latency is not None else None
        ),

        "p99_operation_latency_ms": (
            round(p99_latency, 3)
            if p99_latency is not None else None
        ),

        # These are estimated application payload bytes, not wire bytes.
        "estimated_payload_bytes_sent": bytes_sent,
        "estimated_payload_bytes_received": bytes_received,

        "final_l2_distance": (
            round(final_l2, 6)
            if final_l2 is not None else None
        ),

        "final_rmse": (
            round(final_rmse, 6)
            if final_rmse is not None else None
        ),

        "convergence_history": convergence_history,
        "scenario_events": scenario_events,
        "worker_errors": worker_errors,
    }

    print("\n=== Part B Test Harness Summary ===")

    for key, value in summary.items():
        if key not in ("config", "convergence_history"):
            print(f"  {key}: {value}")

    if cfg.get("output_json"):
        with open(
            cfg["output_json"],
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2)

        print(f"\nMetrics saved to {cfg['output_json']}")

    return summary


if __name__ == "__main__":
    main()
