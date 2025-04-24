import numpy as np
import torch

from collections import OrderedDict
from .belady import Belady


class SVM:
    def __init__(self, M):
        self.w = np.zeros(M)
        self.b = 0.0

    def svm_forward(self, X, y):
        """
        X: (N, M)
        y: (N,)
        w: (M,)
        b: scalar
        返回: loss, f
        """
        f = X @ self.w + self.b  # (N,)
        margins = 1 - y * f
        losses = np.maximum(0, margins)
        loss = np.mean(losses)
        return loss, f

    def svm_backward(self, X, y, f):
        """
        X: (N, M)
        y: (N,)
        f: (N,)
        返回: grad_w: (M,), grad_b: 标量
        """
        _, M = X.shape
        mask = (1 - y * f) > 0  # 违背margin的样本
        # w的梯度
        grad_w = (
            -np.mean(y[mask, None] * X[mask], axis=0) if np.any(mask) else np.zeros(M)
        )
        # b的梯度
        grad_b = -np.mean(y[mask]) if np.any(mask) else 0.0
        return grad_w, grad_b

    def svm_predict(self, X):
        preds = np.sign(X @ self.w + self.b)
        preds[preds == 0] = 1
        return preds.reshape(-1, 1)

class SVM_Cache:
    def __init__(self, cache_size, k=5, M=16):
        self.cache_size = cache_size
        self.k = k
        self.M = M
        self.isvm = None

    def train(self, requests):
        X = [i.obj_id for i in requests]
        N = len(X)
        belady = Belady(self.cache_size)
        belady.initial(requests)
        belady.resize(N)
        X = self.preprocess(X)
        Y = np.array([1 if i else -1 for i in belady.result])
        lr = 0.01
        epochs = 2000
        last_b = -1.0

        self.isvm = SVM(self.M)

        for epoch in range(epochs):
            loss, f = self.isvm.svm_forward(X, Y)
            grad_w, grad_b = self.isvm.svm_backward(X, Y, f)
            self.isvm.w -= lr * grad_w
            self.isvm.b -= lr * grad_b

            print(f"Epoch {epoch}, Loss: {loss:.3f}, b: {self.isvm.b:.4f}")
            if (epoch % 100 == 0):
                if (abs((self.isvm.b - last_b) / last_b) < 0.01):
                    break
                last_b = self.isvm.b

    def predict(self, X):
        X = self.preprocess([i.obj_id for i in X])
        return [False if i < 0 else True for i in self.isvm.svm_predict(X)]

    def preprocess(self, X):
        lru_pc = OrderedDict()
        X_processed = []

        for i in range(len(X)):
            x_i = X[i] & 0xF  # 只保留最后4个bit (16种可能)
            # LRU操作：如果已存在先移除，插入到末尾
            if x_i in lru_pc:
                lru_pc.pop(x_i)
            lru_pc[x_i] = True
            # 如果超过5个，移除最早插入的
            if len(lru_pc) > self.k:
                lru_pc.popitem(last=False)

            # 构建16维向量
            vec = np.zeros(self.M, dtype=int)
            for key in lru_pc.keys():
                vec[key] = 1
            X_processed.append(vec)

        X_processed = np.array(X_processed)
        return X_processed
    
class ISVM:
    def __init__(self, M=16, threshold=60, upper_bound=100):
        self.weight = np.zeros(M)
        self.threshold = threshold
        self.upper_bound = upper_bound

    def svm_update(self, X, y):
        # X: [N, M], 元素0或1
        # y: [N], 元素-1或1
        for i in range(X.shape[0]):
            xi = X[i]
            yi = y[i]
            if yi == 1:
                score = np.dot(self.weight, xi)
                if score < self.upper_bound:
                    self.weight += xi

    def svm_predict(self, x):
        # X: [N]
        score = np.dot(self.weight, x)
        if score > self.threshold:
            return False
        else:
            return True
    
class ISVM_Cache:
    def __init__(self, cache_size, k=5, N=32, M=16, threhold=60, upper_bound=100):
        self.cache_size = cache_size
        self.k = k
        self.N = N
        self.M = M
        self.threhold = threhold
        self.upper_bound = upper_bound
        self.isvm_table = [ISVM(M,threhold, upper_bound) for _ in range(N)]

    def train(self, requests):
        X = [i.obj_id for i in requests]
        belady = Belady(self.cache_size)
        belady.initial(requests)
        belady.resize(len(X))
        Y = np.array([-1 if i else 1 for i in belady.result])
        X_hash, Y_hash = self.preprocess(X,Y)

        for i in range(self.N):
            self.isvm_table[i].svm_update(X_hash[i], Y_hash[i])

    def predict(self, X):
        X = [i.obj_id for i in X]
        X_processed = self.preprocess_X(X)

        Y = []
        for i in range(len(X)):
            Y.append(self.isvm_table[X[i] % self.N].svm_predict(X_processed[i]))
        
        return Y

    def preprocess_X(self, X):
        lru_pc = OrderedDict()
        X_processed = []

        for i in range(len(X)):
            x_i = X[i] & 0xF  # 只保留最后4个bit (16种可能)
            # LRU操作：如果已存在先移除，插入到末尾
            if x_i in lru_pc:
                lru_pc.pop(x_i)
            lru_pc[x_i] = True
            # 如果超过5个，移除最早插入的
            if len(lru_pc) > self.k:
                lru_pc.popitem(last=False)

            # 构建16维向量
            vec = np.zeros(self.M, dtype=int)
            for key in lru_pc.keys():
                vec[key] = 1
            X_processed.append(vec)

        X_processed = np.array(X_processed)
        return X_processed

    def preprocess(self, X, Y):
        lru_pc = OrderedDict()
        X_hash = []
        Y_hash = []

        for i in range(self.N):
            X_hash.append([])
            Y_hash.append([])

        for i in range(len(X)):
            x_i = X[i] & 0xF  # 只保留最后4个bit (16种可能)
            # LRU操作：如果已存在先移除，插入到末尾
            if x_i in lru_pc:
                lru_pc.pop(x_i)
            lru_pc[x_i] = True
            # 如果超过5个，移除最早插入的
            if len(lru_pc) > self.k:
                lru_pc.popitem(last=False)

            # 构建16维向量
            vec = np.zeros(self.M, dtype=int)
            for key in lru_pc.keys():
                vec[key] = 1
            X_hash[X[i] % self.N].append(vec)
            Y_hash[X[i] % self.N].append(Y[i])

        for i in range(self.N):
            X_hash[i] = np.array(X_hash[i])
            Y_hash[i] = np.array(Y_hash[i])
        return X_hash, Y_hash


if __name__ == "__main__":
    N, M = 10, 5
    X = np.random.randint(0, 2, size=(N, M))
    y = np.random.choice([-1, 1], size=N)

    lr = 1.0
    epochs = 10

    isvm = SVM(M)

    for epoch in range(epochs):
        loss, f = isvm.svm_forward(X, y)
        grad_w, grad_b = isvm.svm_backward(X, y, f)
        isvm.w -= lr * grad_w
        isvm.b -= lr * grad_b
        print(f"Epoch {epoch}, Loss: {loss:.3f}, b: {isvm.b:.4f}")
