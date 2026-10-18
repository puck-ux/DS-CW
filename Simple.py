"""Smallest possible starting point: an in-process parameter server.
Run: python simple_ps.py
"""
import threading

import numpy as np


class ParameterServer:
    def __init__(self, num_keys, lr=0.1):
        self.w = np.zeros(num_keys)
        self.lr = lr
        self.lock = threading.Lock()

    def pull(self, keys):
        with self.lock:
            return self.w[keys].copy()

    def push(self, keys, grads):
        with self.lock:
            np.add.at(self.w, keys, -self.lr * grads)  # np.add.at sums duplicate keys


# --- demo: 4 workers all pull, compute a gradient, push ---
N, WORKERS, STEPS = 100, 4, 50
w_star = np.random.default_rng(0).normal(size=N)   # hidden target
ps = ParameterServer(N, lr=0.1)                    # lr * workers = 0.4 < 1, so it converges


def worker():
    keys = np.arange(N)
    for _ in range(STEPS):
        w = ps.pull(keys)
        ps.push(keys, w - w_star)


threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("distance to target:", np.linalg.norm(ps.w - w_star))