import threading
import time

from ps_client import PSClient

def client_worker(client_id):
    with PSClient("localhost:8000") as client:
        print(f"Client {client_id}: submitting Pull")

        pull = client.pull(start=0, length=5)
        client.wait(pull)

        print(f"Client {client_id}: submitting Push")

        grads = [1.0] * 5
        push = client.push(grads, deps=[pull])
        client.wait(push)

        print(f"Client {client_id}: Push completed")

threads = []

for i in range(3):
    thread = threading.Thread(target=client_worker, args=(i,))
    threads.append(thread)

start = time.perf_counter()

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
    

# Check the final weights after all clients have finished.
with PSClient("localhost:8000") as client:
    pull = client.pull(start=0, length=5)
    final_weights = client.wait(pull)

print("Final weights:", final_weights)
print(f"Total elapsed time: {time.perf_counter() - start:.3f}s")