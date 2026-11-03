import argparse
import time
import threading

import grpc
import numpy as np

import ps_pb2
import ps_pb2_grpc
from config import NUM_KEYS, SEED

w_star = np.random.default_rng(SEED).normal(size=NUM_KEYS)  # hidden target


def run_worker(worker_id, stub, access, keys, iterations, compute_ms):
    for t in range(1, iterations + 1):
        t0 = time.perf_counter()
        pull = stub.Pull(ps_pb2.PullRequest(**access)) # pull             
        w = np.array(stub.Wait(pull).weights)
        time.sleep(compute_ms / 1000)                              
        grads = w - w_star[keys] 
        push = stub.Push(ps_pb2.PushRequest(grads=grads.tolist(), deps=[pull.handle_id])) # push
        stub.Wait(push)
        print(f"worker {worker_id} iteration {t}: {(time.perf_counter() - t0) * 1000:.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="B1 emulated worker")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--compute-ms", type=float, default=10)
    parser.add_argument("--keys", nargs="+", type=int)
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "LENGTH"))
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
        threads = [threading.Thread(target=run_worker,
                                    args=(k, stub, access, keys, args.iterations, args.compute_ms))
                   for k in range(args.workers)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        final = np.array(stub.Wait(stub.Pull(ps_pb2.PullRequest(**access))).weights)
        print("Distance to target:", np.linalg.norm(final - w_star[keys]))


if __name__ == "__main__":
    main()