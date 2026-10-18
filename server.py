"""Parameter server process. Run: python server.py"""
import threading
from concurrent import futures

import grpc
import numpy as np

import ps_pb2
import ps_pb2_grpc

from config import NUM_KEYS, LR, PORT


class ParameterServerServicer(ps_pb2_grpc.ParameterServerServicer):
    def __init__(self):
        self.w = np.zeros(NUM_KEYS)    # the model: one weight per key
        self.lock = threading.Lock()   # one request changes the weights at a time
 
    def Push(self, request, context):
        keys = np.array(request.keys, dtype=np.int64)
        grads = np.array(request.grads)
        with self.lock:
            np.add.at(self.w, keys, -LR * grads)   # w = w - lr * g
        return ps_pb2.PushResponse()
 
    def Pull(self, request, context):
        keys = np.array(request.keys, dtype=np.int64)
        with self.lock:
            weights = self.w[keys].tolist()
        return ps_pb2.PullResponse(weights=weights)


# function to start the server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ps_pb2_grpc.add_ParameterServerServicer_to_server(ParameterServerServicer(), server)
    server.add_insecure_port('[::]:8000')
    print("Server starts on port 8000")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()