"""Parameter server process. Run: python server.py"""
import threading
from concurrent import futures

import grpc
import numpy as np

import ps_pb2
import ps_pb2_grpc

from config import DEFAULT_PORT, NUM_KEYS, LR, SERVER_THREADS

class ParameterServerServicer(ps_pb2_grpc.ParameterServerServicer):
    def __init__(self):
        self.w = np.zeros(NUM_KEYS)    # the model: one weight per key
        self.lock = threading.Lock()   # one request changes the weights at a time
        
        # Part A1: Asynchronous Task Management
        self.task_handle_counter = 0
        self.tasks = {}  # # Maps handle IDs to task information (type, data, keys, done, error)
        self.task_executor = futures.ThreadPoolExecutor(max_workers=SERVER_THREADS) # Executes submitted tasks in background threads
        
        
    def _create_task_handle(self, task_type, keys=None):
        with self.lock:
            self.task_handle_counter += 1
            next_task_handle_id = self.task_handle_counter
            self.tasks[next_task_handle_id] = {
                "type": task_type,
                "data": None,
                "keys": keys,
                "done": threading.Event(),
                "error": None
            }
        return ps_pb2.TaskHandle(handle_id=next_task_handle_id)
    
    def _finish_task(self, handle_id, data=None, error=None):
        with self.lock:
            task = self.tasks[handle_id]
            task["data"] = data
            task["error"] = error
            task["done"].set()
    
    def _execute_pull(self, handle_id, dependency_ids):
        try:
            for dep_id in dependency_ids: # Wait for every declared dependency to complete
                with self.lock: dependency_task = self.tasks[dep_id]
                dependency_task["done"].wait()
                with self.lock: dependency_error = dependency_task["error"]
                if dependency_error:
                    self._finish_task(handle_id, error=dependency_error)
                    return

            with self.lock: # Retrieve weights after dependencies have completed
                task = self.tasks[handle_id]
                weights = self.w[task["keys"]].tolist()
            self._finish_task(handle_id, data=weights)

        except Exception as exc:
            self._finish_task(handle_id,error=(grpc.StatusCode.INTERNAL, str(exc)))
            
            
    def _execute_push(self, handle_id, dependency_ids, pull_handle_id, grads):
        try:
            for dep_id in dependency_ids: # Wait for every declared dependency to complete
                with self.lock: dependency_task = self.tasks[dep_id]
                dependency_task["done"].wait()
                with self.lock: dependency_error = dependency_task["error"]
                if dependency_error:
                    self._finish_task(handle_id, error=dependency_error)
                    return
            with self.lock: # Retrieve keys from the Pull dependency
                pull_task = self.tasks[pull_handle_id]
                keys = pull_task["keys"].copy()
                np.add.at(self.w, keys, -LR * grads)
            self._finish_task(handle_id)

        except Exception as exc:
            self._finish_task(handle_id,error=(grpc.StatusCode.INTERNAL, str(exc)))
    
 
    def Push(self, request, context):
        if not request.deps: context.abort(grpc.StatusCode.INVALID_ARGUMENT,"Push requires at least one dependency")
        if len(set(request.deps)) != len(request.deps): context.abort(grpc.StatusCode.INVALID_ARGUMENT,"Duplicate dependency handles are not allowed")
        with self.lock: dependency_tasks = [self.tasks.get(dep) for dep in request.deps]
        if any(task is None for task in dependency_tasks):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,"Invalid dependency handle")
        pull_dependencies = [
            dep for dep, task in zip(request.deps, dependency_tasks)
            if task["type"] == "pull"
        ]
        if len(pull_dependencies) != 1:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,"Push requires exactly one Pull dependency to identify its keys")
        pull_handle_id = pull_dependencies[0]
        pull_task = next(
            task for dep, task in zip(request.deps, dependency_tasks)
            if dep == pull_handle_id
        )
        if len(request.grads) != len(pull_task["keys"]): context.abort(grpc.StatusCode.INVALID_ARGUMENT,"Gradient count does not match the pulled keys")
        grads = np.array(request.grads, dtype=np.float64)
        task_handle = self._create_task_handle("push")
        self.task_executor.submit(
            self._execute_push,
            task_handle.handle_id,
            list(request.deps),
            pull_handle_id,
            grads
        )
        return task_handle
    
 
    def Pull(self, request, context):
        # Validate dependencies
        if len(set(request.deps)) != len(request.deps):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,"Duplicate dependency handles are not allowed")
        with self.lock: dependency_tasks = [self.tasks.get(dep) for dep in request.deps]
        if any(task is None for task in dependency_tasks):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,"Invalid dependency handle")
        
        # Handle sparse keys or range selection
        if request.sparse_keys:
            if min(request.sparse_keys) < 0 or max(request.sparse_keys) >= NUM_KEYS:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT,f"Keys must be between 0 and {NUM_KEYS - 1}")
            keys = np.array(request.sparse_keys, dtype=np.int64)
        elif request.HasField("range"):
            start = request.range.start_idx
            length = request.range.length
            if start > NUM_KEYS or start + length > NUM_KEYS:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT,f"Invalid range [{start}, {start + length}) for {NUM_KEYS} keys")
            keys = np.arange(start, start + length, dtype=np.int64)
        else:
            keys = np.array([], dtype=np.int64) # No range or sparse keys: empty selection
        task_handle = self._create_task_handle("pull", keys.copy())
        self.task_executor.submit(
            self._execute_pull,
            task_handle.handle_id,
            list(request.deps)
        )
        return task_handle
    
    
    def Wait(self, request, context):
        handle_id = request.handle_id
        with self.lock: task_info = self.tasks.get(handle_id)
        if task_info is None: context.abort(grpc.StatusCode.NOT_FOUND,f"Unknown task handle {handle_id}")
        task_info["done"].wait() # Block until the background task finishes        
        with self.lock:
            error = task_info["error"]
            returned_weights = (task_info["data"] if task_info["type"] == "pull" else []) # Return weights only for pull tasks
        if error:
            status, message = error
            context.abort(status, message)
        return ps_pb2.WaitResponse(weights=returned_weights)
    

# function to start the server
def serve(port=DEFAULT_PORT):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=SERVER_THREADS))
    ps_pb2_grpc.add_ParameterServerServicer_to_server(ParameterServerServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"Server starts on port {port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()