import numpy as np
import matplotlib.pyplot as plt

# Input data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# OR gate outputs
y = np.array([0, 1, 1, 1])

# Perceptron parameters
w1 = 1
w2 = 1
b = -0.5

# Plot data points
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red', s=100, label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='blue', s=100, label='Class 1' if i == 1 else "")

# Decision boundary: w1*x1 + w2*x2 + b = 0
x = np.linspace(-0.2, 1.2, 100)
y_boundary = (-w1 * x - b) / w2

plt.plot(x, y_boundary, 'k--', label='Decision Boundary')

# Labels
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Boundary for OR Gate")
plt.grid(True)
plt.legend()

plt.show()
