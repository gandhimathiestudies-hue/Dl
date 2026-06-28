import numpy as np

def step(x):
    return 1 if x >= 0 else 0

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

for x1, x2 in X:
    h1 = step(x1 + x2 - 0.5)   
    h2 = step(x1 + x2 - 1.5) 
    y = step(h1 - 2*h2 - 0.5)

    print(f"{x1} {x2} -> {y}")