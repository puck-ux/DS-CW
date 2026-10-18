import grpc
import ps_pb2
import ps_pb2_grpc
import argparse
import numpy as np

from config import NUM_KEYS, STEPS, SEED

# hidden target: same seed everywhere, so every worker agrees on it
w_star = np.random.default_rng(SEED).normal(size=NUM_KEYS)

def run():
    # channel to connect to the server - can be made remote
    with grpc.insecure_channel('localhost:8000') as channel:
        stub = ps_pb2_grpc.ParameterServerStub(channel)
        
        keys = list(range(NUM_KEYS))
 
        for _ in range(STEPS):
            # pull the current weights
            response = stub.Pull(ps_pb2.PullRequest(keys=keys))
            w = np.array(response.weights)
 
            # compute a gradient
            grads = w - w_star
 
            # push it
            stub.Push(ps_pb2.PushRequest(keys=keys, grads=grads.tolist()))
 
        w = np.array(stub.Pull(ps_pb2.PullRequest(keys=keys)).weights)
        print("distance to target:", np.linalg.norm(w - w_star))
 
 
if __name__ == "__main__":
    run()
 