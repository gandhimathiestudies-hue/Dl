import numpy as np

def step(x):
    return 1 if x >= 0 else 0

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

for x1, x2 in X:
    y = step(x1 + x2 - 0.5)
    print(f"{x1} OR {x2} = {y}")