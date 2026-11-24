import grpc
import numpy as np
import ps_pb2
import ps_pb2_grpc

class PSClient:
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address)
        self.stub = ps_pb2_grpc.ParameterServerStub(self.channel)

    def pull(self, keys=None, start=None, length=None, deps=None):
        dep_ids = [handle.handle_id for handle in deps] if deps else []
        if keys is not None:
            request = ps_pb2.PullRequest(sparse_keys=np.asarray(keys).tolist(), deps=dep_ids)
        elif start is not None and length is not None:
            request = ps_pb2.PullRequest(range=ps_pb2.KeyRange(start_idx=start, length=length), deps=dep_ids)
        else:
            raise ValueError("Provide either keys or start and length")
        return self.stub.Pull(request)

    def push(self, grads, deps):
        dep_ids = [
            h.handle_id if hasattr(h, "handle_id") else h
            for h in deps
        ]
        request = ps_pb2.PushRequest(grads=np.asarray(grads).tolist(),deps=dep_ids)
        return self.stub.Push(request)

    def wait(self, handle):
        response = self.stub.Wait(handle)
        return np.array(response.weights)

    def close(self):
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
