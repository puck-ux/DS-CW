import argparse
import time
import threading

import grpc
import numpy as np

import ps_pb2
import ps_pb2_grpc
from config import NUM_KEYS, SEED

w_star = np.random.default_rng(SEED).normal(size=NUM_KEYS)  # hidden target


def run_worker(worker_id, stub, access, keys, iterations, compute_ms, hetero_compute):
    for t in range(1, iterations + 1):
        t0 = time.perf_counter()
        pull = stub.Pull(ps_pb2.PullRequest(**access)) # pull             
        w = np.array(stub.Wait(pull).weights)
        if hetero_compute:
            time.sleep(np.random.exponential(scale=compute_ms / 1000))
        else:
            time.sleep(compute_ms / 1000)
        grads = w - w_star[keys] 
        push = stub.Push(ps_pb2.PushRequest(grads=grads.tolist(), deps=[pull.handle_id])) # push
        stub.Wait(push)
        print(f"worker {worker_id} iteration {t}: {(time.perf_counter() - t0) * 1000:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="B1 emulated worker")
    parser.add_argument("--workers", type=int, default=1) # number of workers
    parser.add_argument("--iterations", type=int, default=20) # number of iterations
    parser.add_argument("--compute-ms", type=float, default=10) # compute time
    parser.add_argument("--keys", nargs="+", type=int) # number of keys
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "LENGTH")) # range of keys each worker does
    parser.add_argument("--hetero_compute", action="store_true") # true or false for enabling heterogeneous compute times
    parser.add_argument("--straggler_fraction", type=float, default=0) # percentage of workers that are stragglers with 1 being 100%
    args = parser.parse_args()

    if args.range:
        start, length = args.range
        keys = np.arange(start, start + length)
        access = {"range": ps_pb2.KeyRange(start_idx=start, length=length)}
    else:
        keys = np.array(args.keys) if args.keys else np.arange(NUM_KEYS)
        access = {"sparse_keys": keys.tolist()}

    with grpc.insecure_channel("localhost:8000") as channel:
        stub = ps_pb2_grpc.ParameterServerStub(channel)
        stragglers = round(args.workers * args.straggler_fraction)
        threads = []
        for i in range(args.workers):
            if i < stragglers:
                compute_ms = args.compute_ms * 5 
            else:
                compute_ms = args.compute_ms
            threads.append(threading.Thread(target=run_worker, args=(i, stub, access, keys, args.iterations, compute_ms, args.hetero_compute)))
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        final = np.array(stub.Wait(stub.Pull(ps_pb2.PullRequest(**access))).weights)
        print("Distance to target:", np.linalg.norm(final - w_star[keys]))


if __name__ == "__main__":
    main()