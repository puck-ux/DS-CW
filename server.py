"""Core parameter server implementation."""
import threading
from concurrent import futures

import numpy as np

from config import NUM_KEYS, LR, SERVER_THREADS

class ParameterServer:
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
        return next_task_handle_id
    
    def _finish_task(self, handle_id, data=None, error=None):
        with self.lock:
            task = self.tasks[handle_id]
            task["data"] = data
            task["error"] = error
            task["done"].set()
            
    def _validate_dependencies(self, dependency_ids):
        if len(set(dependency_ids)) != len(dependency_ids): raise ValueError("Duplicate dependency handles are not allowed")
        with self.lock:
            dependency_tasks = [self.tasks.get(dep_id) for dep_id in dependency_ids]
        if any(task is None for task in dependency_tasks): raise ValueError("Invalid dependency handle")
        return dependency_tasks
    
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
            self._finish_task(handle_id, error=exc)
            
            
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
            self._finish_task(handle_id, error=exc)
            
    
    def submit_pull(self, keys=None, start=None, length=None, deps=None):
        dependency_ids = list(deps or [])
        self._validate_dependencies(dependency_ids)

        if keys is not None and (start is not None or length is not None):
            raise ValueError("Provide either sparse keys or a range")

        if keys is not None:
            key_values = list(keys)
            if any(int(k) != k or k < 0 or k >= NUM_KEYS for k in key_values):
                raise ValueError(f"Keys must be between 0 and {NUM_KEYS - 1}")
            selected_keys = np.array(key_values, dtype=np.int64)

        elif start is not None and length is not None:
            if start < 0 or length < 0 or start + length > NUM_KEYS:
                raise ValueError(f"Invalid range [{start}, {start + length}) " f"for {NUM_KEYS} keys")
            selected_keys = np.arange(start, start + length, dtype=np.int64)

        elif start is None and length is None:
            selected_keys = np.array([], dtype=np.int64)

        else:
            raise ValueError("Both start and length must be provided")

        handle_id = self._create_task_handle("pull", selected_keys.copy())
        self.task_executor.submit(
            self._execute_pull,
            handle_id,
            dependency_ids
        )
        return handle_id
    
    
    def submit_push(self, grads, deps):
        dependency_ids = list(deps)
        if not dependency_ids: raise ValueError("Push requires at least one dependency")
        dependency_tasks = self._validate_dependencies(dependency_ids)
        
        pull_dependencies = [
            dep_id
            for dep_id, task in zip(dependency_ids, dependency_tasks)
            if task["type"] == "pull"
        ]
        if len(pull_dependencies) != 1: raise ValueError("Push requires exactly one Pull dependency " "to identify its keys")
        pull_handle_id = pull_dependencies[0]
        grads = np.asarray(grads, dtype=np.float64)
        if grads.ndim != 1: raise ValueError("Gradients must be a one-dimensional array")
        pull_task = next(
            task for dep_id, task in zip(dependency_ids, dependency_tasks)
            if dep_id == pull_handle_id
        )
        if len(grads) != len(pull_task["keys"]): raise ValueError("Gradient count does not match the pulled keys")
        
        handle_id = self._create_task_handle("push")
        self.task_executor.submit(
            self._execute_push,
            handle_id,
            dependency_ids,
            pull_handle_id,
            grads
        )
        return handle_id
    
    
    def wait_task(self, handle_id):
        with self.lock: task_info = self.tasks.get(handle_id)
        if task_info is None: raise KeyError(f"Unknown task handle {handle_id}")
        task_info["done"].wait()
        with self.lock:
            error = task_info["error"]
            data = task_info["data"]
            task_type = task_info["type"]
        if error: raise RuntimeError(f"Task {handle_id} failed: {error}") from error
        return data if task_type == "pull" else None # Pull tasks return weights; Push tasks return None.