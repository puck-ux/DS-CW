# Model & Correctness Oracle
NUM_KEYS = 100
# Gradients are summed across m workers, so the correctness oracle converges
# iff 0 < LR * m < 2 (e.g. LR = 0.05 supports up to m = 39 concurrent workers).
LR = 0.05
STEPS = 50
SEED = 0  # seed for the hidden target vector

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_ADDRESS = "localhost:8000"
SERVER_THREADS = 64