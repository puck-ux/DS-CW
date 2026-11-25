"""gRPC interface for the parameter server. Run: python ps_server.py"""

from concurrent import futures

import grpc

import ps_pb2
import ps_pb2_grpc

from config import DEFAULT_PORT, SERVER_THREADS
from server import ParameterServer


class ParameterServerServicer(ps_pb2_grpc.ParameterServerServicer):
    def __init__(self):
        self.parameter_server = ParameterServer()


    def Push(self, request, context):
        try:
            handle_id = self.parameter_server.submit_push(request.grads,request.deps)
            return ps_pb2.TaskHandle(handle_id=handle_id)
        
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))


    def Pull(self, request, context):
        try:
            if request.sparse_keys:
                handle_id = self.parameter_server.submit_pull(keys=request.sparse_keys,deps=request.deps)
            elif request.HasField("range"):
                handle_id = self.parameter_server.submit_pull(
                    start=request.range.start_idx,
                    length=request.range.length,
                    deps=request.deps
                )
            else:
                handle_id = self.parameter_server.submit_pull(deps=request.deps)
            return ps_pb2.TaskHandle(handle_id=handle_id)

        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))


    def Wait(self, request, context):
        try:
            weights = self.parameter_server.wait_task(request.handle_id)
            return ps_pb2.WaitResponse(weights=weights if weights is not None else [])
        except KeyError as exc: context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except RuntimeError as exc: context.abort(grpc.StatusCode.INTERNAL, str(exc))


def serve(port=DEFAULT_PORT):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=SERVER_THREADS))
    ps_pb2_grpc.add_ParameterServerServicer_to_server(ParameterServerServicer(),server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"Server starts on port {port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
