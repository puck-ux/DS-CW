"""Parameter server process. Run: python server.py"""
import threading
from concurrent import futures

import grpc
import numpy as np

import ps_pb2
import ps_pb2_grpc

from config import NUM_KEYS, LR

class ParameterServerServicer(ps_pb2_grpc.ParameterServerServicer):
    def __init__(self):
        self.w = np.zeros(NUM_KEYS)    # the model: one weight per key
        self.lock = threading.Lock()   # one request changes the weights at a time
        
        # Part A1: Asynchronous Task Management
        self.task_handle_counter = 0
        self.tasks = {} # Maps handle_id (int) -> {"type": "pull"/"push", "data": weights_list or None}
        
        
    def _create_task_handle(self, task_type, data=None, keys=None):
        self.task_handle_counter += 1
        next_task_handle_id = self.task_handle_counter
        self.tasks[next_task_handle_id] = {"type": task_type, "data": data, "keys": keys}
        return ps_pb2.TaskHandle(handle_id=next_task_handle_id)
    
 
    def Push(self, request, context):
        if not request.deps:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,"Push requires a dependency on a Pull task")
            
        pull_handle_id = request.deps[0]
        pull_task = self.tasks.get(pull_handle_id)
        
        if not pull_task or pull_task["type"] != "pull":
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,"Invalid Pull dependency")
            
        keys = pull_task["keys"]
        grads = np.array(request.grads)
        
        if len(keys) != len(grads):
            context.abort( grpc.StatusCode.INVALID_ARGUMENT,"Gradient count does not match the pulled keys")
        
        with self.lock:
            np.add.at(self.w, keys, -LR * grads)   # w = w - lr * g
            task_handle = self._create_task_handle(task_type="push")
            
        return task_handle
    
 
    def Pull(self, request, context):
        if request.sparse_keys:
            keys = np.array(request.sparse_keys, dtype=np.int64)
        else:
            start = request.range.start_idx
            length = request.range.length
            keys = np.arange(start, start + length, dtype=np.int64)
            
        with self.lock:
            weights_snapshot = self.w[keys].tolist()
            task_handle = self._create_task_handle(task_type="pull", data=weights_snapshot, keys=keys.copy())
            
        return task_handle
    
    
    def Wait(self, request, context):
        handle_id = request.handle_id
        if handle_id not in self.tasks: return ps_pb2.WaitResponse()
        task_info = self.tasks[handle_id]
        returned_weights = []
        if task_info["type"] == "pull": returned_weights = task_info["data"] # Return weights only for pull tasks
        return ps_pb2.WaitResponse(weights=returned_weights)
    

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