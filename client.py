import grpc
import ps_pb2
import ps_pb2_grpc
import argparse
import numpy as np

from config import NUM_KEYS, STEPS, SEED

# hidden target: same seed everywhere, so every worker agrees on it
w_star = np.random.default_rng(SEED).normal(size=NUM_KEYS) # Correctness Oracle

def run():
    parser = argparse.ArgumentParser(description="Parameter Server Client Worker Harness")
    parser.add_argument("--range",nargs=2,type=int,metavar=("START", "LENGTH"),
                        help="Range Access with a starting index and length")
    parser.add_argument("--keys",nargs="+",type=int,metavar="KEY",help="Keys for Sparse List Access")
    args = parser.parse_args()
    
    if args.range is not None and args.keys is not None:
        parser.error("Choose either --range or --keys, not both")
        
    if args.range is not None:
        start, length = args.range
        if start < 0 or length < 1 or start + length > NUM_KEYS:
            parser.error("Range must be within the available keys")
        print(f"Using Range Access: {start} to {start + length - 1}")
        keys = np.arange(start, start + length)
        access = {"range": ps_pb2.KeyRange(start_idx=start,length=length)}
    else:
        if args.keys is None:
            keys = np.arange(NUM_KEYS)
            print("Using Sparse List Access: all keys")
        else:
            keys = np.array(args.keys)
            print("Using Sparse List Access: specified keys")
        if np.any(keys < 0) or np.any(keys >= NUM_KEYS):
            parser.error(f"Keys must be between 0 and {NUM_KEYS - 1}")
        if len(np.unique(keys)) != len(keys):
            parser.error("Duplicate keys are not allowed")
        access = {"sparse_keys": keys.tolist()}
    
    # channel to connect to the server - can be made remote
    with grpc.insecure_channel('localhost:8000') as channel:
        stub = ps_pb2_grpc.ParameterServerStub(channel)
        for _ in range(STEPS):
            pull_handle = stub.Pull(ps_pb2.PullRequest(**access)) # Returns a tracking handle ID ticket immediately
            wait_response = stub.Wait(pull_handle) # wait to resolve and unpack raw weights slice data payload
            w = np.array(wait_response.weights)
            grads = w - w_star[keys] # compute a gradient
            push_handle = stub.Push(ps_pb2.PushRequest( # Asynchronous push request, returns handle ID immediately
                grads=grads.tolist(), deps=[pull_handle.handle_id]))
            stub.Wait(push_handle) # wait for the push to complete before next iteration
            
        final_pull = stub.Pull(ps_pb2.PullRequest(**access))
        final_weights = np.array(stub.Wait(final_pull).weights)
        distance = np.linalg.norm(final_weights - w_star[keys])
        rmse = distance / np.sqrt(len(keys))
        print("Distance to target convergence:", distance)
        print("Average convergence error (RMSE):", rmse)
 
if __name__ == "__main__":
    run()
 