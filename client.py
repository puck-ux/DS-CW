import argparse
import numpy as np

from ps_client import PSClient
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
        access = {"start": start, "length": length}
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
        access = {"keys": keys}
    
    # channel to connect to the server - can be made remote
    with PSClient("localhost:8000") as ps:
        for _ in range(STEPS):
            pull_handle = ps.pull(**access)
            w = ps.wait(pull_handle)
            grads = w - w_star[keys]
            push_handle = ps.push(grads, deps=[pull_handle])
            ps.wait(push_handle)

        final_pull = ps.pull(**access)
        final_weights = ps.wait(final_pull)

        distance = np.linalg.norm(final_weights - w_star[keys])
        rmse = distance / np.sqrt(len(keys))

        print("Distance to target convergence:", distance)
        print("Average convergence error (RMSE):", rmse)

 
if __name__ == "__main__":
    run()
 