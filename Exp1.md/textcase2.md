import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)
        self.lr, self.epochs = lr, epochs
    def activation(self, x): return 1 if x >= 0 else 0
    def predict(self, x):
        return self.activation(np.dot(x, self.weights[1:]) + self.weights[0])
    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                error = target - self.predict(xi)
                self.weights[1:] += self.lr * error * xi
                self.weights[0] += self.lr * error

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
p = Perceptron(2)
p.train(X,y)
print("XOR predictions:")
for xi in X: print(f"{xi.tolist()} => {p.predict(xi)}")
 output:
 Perceptron Predictions on XOR:
Input: [0, 0] => Output: 1
Input: [0, 1] => Output: 1
Input: [1, 0] => Output: 0
Input: [1, 1] => Output: 0
