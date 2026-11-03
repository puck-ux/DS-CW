"""A4 single-node tests for the parameter server.

Run from the project folder with:  python -m unittest test_server -v

Each test starts a fresh in-process gRPC server on a free port, so state never
leaks between tests and nothing needs to be running in another terminal.
"""
import threading
import unittest
from concurrent import futures

import grpc
import numpy as np

import ps_pb2
import ps_pb2_grpc
from config import NUM_KEYS, LR
from server import ParameterServerServicer


class PSTestCase(unittest.TestCase):
    def setUp(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        ps_pb2_grpc.add_ParameterServerServicer_to_server(ParameterServerServicer(), self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")  # port 0 = pick a free one
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        self.stub = ps_pb2_grpc.ParameterServerStub(self.channel)

    def tearDown(self):
        self.channel.close()
        self.server.stop(0)

    # ---- helpers -------------------------------------------------------
    def pull(self, stub=None, **access):
        """Pull then Wait. Returns (handle, weights as numpy array)."""
        stub = stub or self.stub
        handle = stub.Pull(ps_pb2.PullRequest(**access))
        return handle, np.array(stub.Wait(handle).weights)

    def pull_range(self, start, length):
        return self.pull(range=ps_pb2.KeyRange(start_idx=start, length=length))

    def pull_keys(self, keys, stub=None):
        return self.pull(stub=stub, sparse_keys=list(keys))

    def push(self, grads, dep_handle, stub=None):
        stub = stub or self.stub
        handle = stub.Push(ps_pb2.PushRequest(grads=list(grads), deps=[dep_handle.handle_id]))
        stub.Wait(handle)
        return handle

    def all_weights(self):
        return self.pull_range(0, NUM_KEYS)[1]

    def assert_rpc_error(self, code, fn, *args, **kwargs):
        with self.assertRaises(grpc.RpcError) as cm:
            fn(*args, **kwargs)
        self.assertEqual(cm.exception.code(), code)


class TestConcurrentAggregation(PSTestCase):
    def run_threads(self, target, n_threads):
        errors = []

        def wrapped(i):
            try:
                target(i)
            except Exception as e:  # surface failures from worker threads
                errors.append(e)

        threads = [threading.Thread(target=wrapped, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_concurrent_pushes_to_same_keys_are_all_applied(self):
        n_threads, pushes_each, keys = 8, 25, range(10)

        def worker(i):
            for _ in range(pushes_each):
                handle, _ = self.pull_keys(keys)
                self.push(np.full(10, float(i + 1)), handle)

        self.run_threads(worker, n_threads)

        total_grad = sum(i + 1 for i in range(n_threads)) * pushes_each  # 36 * 25
        expected = np.zeros(NUM_KEYS)
        expected[:10] = -LR * total_grad
        np.testing.assert_allclose(self.all_weights(), expected)

    def test_concurrent_pushes_to_overlapping_keys_match_offline_sum(self):
        n_threads, pushes_each = 8, 20
        expected = np.zeros(NUM_KEYS)
        for i in range(n_threads):  # thread i owns keys [i, i+10), overlapping neighbours
            expected[i:i + 10] += -LR * (i + 1) * pushes_each

        def worker(i):
            keys = list(range(i, i + 10))
            for _ in range(pushes_each):
                handle, _ = self.pull_keys(keys)
                self.push(np.full(10, float(i + 1)), handle)

        self.run_threads(worker, n_threads)
        np.testing.assert_allclose(self.all_weights(), expected)

    def test_task_handle_ids_are_unique_under_concurrency(self):
        ids, lock = [], threading.Lock()

        def worker(_):
            for _ in range(50):
                handle = self.stub.Pull(ps_pb2.PullRequest(sparse_keys=[0]))
                with lock:
                    ids.append(handle.handle_id)

        self.run_threads(worker, 8)
        self.assertEqual(len(ids), len(set(ids)))

    def test_duplicate_keys_in_one_push_are_summed(self):
        handle, _ = self.pull_keys([3, 3])
        self.push([1.0, 2.0], handle)
        self.assertAlmostEqual(self.all_weights()[3], -LR * 3.0)


class TestRangesAndKeyLists(PSTestCase):
    def test_range_and_sparse_pull_return_the_same_values(self):
        handle, _ = self.pull_range(0, NUM_KEYS)
        self.push(np.arange(NUM_KEYS, dtype=float), handle)  # make weights distinct

        _, by_range = self.pull_range(10, 10)
        _, by_keys = self.pull_keys(range(10, 20))
        np.testing.assert_array_equal(by_range, by_keys)
        self.assertEqual(len(by_range), 10)

    def test_push_through_range_pull_equals_push_through_key_pull(self):
        grads = np.linspace(1.0, 2.0, 10)
        range_handle, _ = self.pull_range(0, 10)
        self.push(grads, range_handle)
        keys_handle, _ = self.pull_keys(range(10, 20))
        self.push(grads, keys_handle)

        w = self.all_weights()
        np.testing.assert_allclose(w[0:10], w[10:20])
        np.testing.assert_allclose(w[0:10], -LR * grads)
        np.testing.assert_array_equal(w[20:], np.zeros(NUM_KEYS - 20))

    def test_sparse_pull_respects_key_order(self):
        handle, _ = self.pull_range(0, NUM_KEYS)
        self.push(np.arange(NUM_KEYS, dtype=float), handle)
        _, w = self.pull_keys([7, 2, 5])
        np.testing.assert_allclose(w, -LR * np.array([7.0, 2.0, 5.0]))

    def test_empty_pull_and_empty_push_change_nothing(self):
        handle, w = self.pull()  # neither range nor keys
        self.assertEqual(len(w), 0)
        self.push([], handle)
        np.testing.assert_array_equal(self.all_weights(), np.zeros(NUM_KEYS))

    def test_pull_returns_snapshot_from_pull_time(self):
        old_handle, _ = self.pull_keys([0])
        self.push([5.0], old_handle)
        # Wait on the old handle again: it must still show the value at pull time (0.0)
        self.assertEqual(list(self.stub.Wait(old_handle).weights), [0.0])
        self.assertAlmostEqual(self.pull_keys([0])[1][0], -LR * 5.0)


class TestDependencies(PSTestCase):
    def test_push_without_dependency_is_rejected(self):
        self.assert_rpc_error(grpc.StatusCode.INVALID_ARGUMENT,
                              self.stub.Push, ps_pb2.PushRequest(grads=[1.0]))

    def test_push_with_unknown_dependency_is_rejected(self):
        self.assert_rpc_error(grpc.StatusCode.INVALID_ARGUMENT,
                              self.stub.Push, ps_pb2.PushRequest(grads=[1.0], deps=[9999]))

    def test_push_cannot_depend_on_a_push(self):
        pull_handle, _ = self.pull_keys([0])
        push_handle = self.push([1.0], pull_handle)
        self.assert_rpc_error(grpc.StatusCode.INVALID_ARGUMENT,
                              self.stub.Push, ps_pb2.PushRequest(grads=[1.0], deps=[push_handle.handle_id]))

    def test_rejected_push_has_no_side_effects(self):
        handle, _ = self.pull_keys([0, 1])
        self.assert_rpc_error(grpc.StatusCode.INVALID_ARGUMENT,
                              self.stub.Push, ps_pb2.PushRequest(grads=[1.0], deps=[handle.handle_id]))  # wrong length
        self.assert_rpc_error(grpc.StatusCode.INVALID_ARGUMENT,
                              self.stub.Push, ps_pb2.PushRequest(grads=[1.0, 1.0], deps=[12345]))       # bad dep
        np.testing.assert_array_equal(self.all_weights(), np.zeros(NUM_KEYS))

    def test_push_only_touches_the_keys_of_its_dependency(self):
        handle, _ = self.pull_keys([5, 6])
        self.push([1.0, 1.0], handle)
        expected = np.zeros(NUM_KEYS)
        expected[[5, 6]] = -LR
        np.testing.assert_allclose(self.all_weights(), expected)

    def test_one_pull_can_feed_several_pushes(self):
        handle, _ = self.pull_keys([0])
        self.push([1.0], handle)
        self.push([1.0], handle)
        self.assertAlmostEqual(self.all_weights()[0], -2 * LR)


class TestErrorHandling(PSTestCase):
    """Contract checks for A2 ('meaningful errors'). Failures here mean the server
    still leaks a raw exception (status UNKNOWN) instead of a clear status code."""

    def test_out_of_range_sparse_key_is_invalid_argument(self):
        self.assert_rpc_error(grpc.StatusCode.INVALID_ARGUMENT,
                              self.stub.Pull, ps_pb2.PullRequest(sparse_keys=[NUM_KEYS]))

    def test_out_of_range_range_is_invalid_argument(self):
        request = ps_pb2.PullRequest(range=ps_pb2.KeyRange(start_idx=NUM_KEYS - 1, length=5))
        self.assert_rpc_error(grpc.StatusCode.INVALID_ARGUMENT, self.stub.Pull, request)

    def test_wait_on_unknown_handle_is_not_found(self):
        self.assert_rpc_error(grpc.StatusCode.NOT_FOUND,
                              self.stub.Wait, ps_pb2.TaskHandle(handle_id=424242))


if __name__ == "__main__":
    unittest.main(verbosity=2)
