import numpy as np
import torch
import copy

from collections import OrderedDict
from .belady import Belady


class SVM:
    def __init__(self, M):
        self.w = np.zeros(M)
        self.b = 0.0
        self.lr = 0.01

    def svm_forward(self, x, y):
        """
        x: (M,)
        y: 标量 (+1或-1)
        返回: loss, f
        """
        f = np.dot(x, self.w) + self.b   # 标量
        margin = 1 - y * f               # 标量
        loss = max(0.0, margin)
        return loss, f
    
    def svm_backward(self, x, y, f):
        """
        x: (M,)
        y: 标量
        f: 标量
        返回: grad_w: (M,), grad_b: 标量
        """
        if 1 - y * f > 0:
            grad_w = -y * x
            grad_b = -y
        else:
            grad_w = np.zeros_like(self.w)
            grad_b = 0.0
        return grad_w, grad_b

    def update_one(self, x, y):
        _, f = self.svm_forward(x, y)
        grad_w, grad_b = self.svm_backward(x, y, f)
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b

    def svm_predict(self, x):
        """
        x: (M,)
        返回: f = x @ w + b，float类型
        """
        return float(np.dot(x, self.w) + self.b)

class SVM_Cache:
    def __init__(self, cache_size, k=5, M=16):
        self.cache_size = cache_size
        self.k = k
        self.M = M
        self.svm_table = {}

    def predict_online(self, X):
        """
        X_all: list, 长度N，请求（带obj_id）
        Y_true: list or np array, 长度N，真实标签（如belady结果[-1/1]）
        return: list, 预测结果
        """
        N = len(X)
        belady = Belady(self.cache_size)
        belady.initial(X)
        belady.resize(N)
        X_processed = self.preprocess([i.obj_id for i in X])
        Y = np.array([1 if i else -1 for i in belady.result])

        Y_pred = []
        total = len(X)
        for i in range(0, total):
            # 预测
            xi = X[i].obj_id
            if xi not in self.svm_table:
                self.svm_table[xi] = SVM(self.M)
            Y_pred.append(int(self.svm_table[xi].svm_predict(X_processed[i]) * 1000))
            # 按实际label增量训练
            self.svm_table[xi].update_one(X_processed[i], Y[i])

        return Y_pred

    def preprocess(self, X):
        lru_pc = OrderedDict()
        X_processed = []

        for i in range(len(X)):
            x_i = int(X[i] / 4096) & (self.M - 1)  # 只保留最后4个bit (16种可能)
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
    def __init__(self, upper_bound=100):
        self.weight = {}
        self.upper_bound = upper_bound

    def svm_update_one(self, x, y):
        score = 0
        for i in x:
            if i not in self.weight:
                self.weight[i] = 0
            score += self.weight[i]
        if score < self.upper_bound and score > -self.upper_bound:
            for i in x:
                self.weight[i] -= y

    def svm_predict(self, x):
        # X: [N]
        score = 0
        for i in x:
            if i not in self.weight:
                self.weight[i] = 0
            score += self.weight[i]
        return score
        
    
class ISVM_Cache:
    def __init__(self, cache_size, k=5, upper_bound=100):
        self.cache_size = cache_size
        self.k = k
        self.upper_bound = upper_bound
        self.isvm_table = {}
    
    def predict_online(self, X):
        """
        X_all: list, 长度N，请求（带obj_id）
        Y_true: list or np array, 长度N，真实标签（如belady结果[-1/1]）
        batch_size: int, e.g. 100
        return: list, 预测结果
        """
        Y_pred = []
        total = len(X)
        belady = Belady(self.cache_size)
        belady.initial(X)
        belady.resize(len(X))
        X1 = [i.obj_id for i in X]
        X_processed = self.preprocess_X(X1)
        Y_processed = [-1 if i else 1 for i in belady.result]
        for i in range(0, total):
            # 预测
            if X1[i] not in self.isvm_table:
                self.isvm_table[X1[i]] = ISVM(self.upper_bound)
            Y_pred.append(self.isvm_table[X1[i]].svm_predict(X_processed[i]))
            # 按实际label增量训练
            self.isvm_table[X1[i]].svm_update_one(X_processed[i], Y_processed[i])
        return Y_pred
    
    def preprocess_X(self, X):
        lru_pc = OrderedDict()
        X_processed = []

        for i in range(len(X)):
            x_i = X[i]
            vec = list(lru_pc.keys())
            X_processed.append(copy.deepcopy(vec))

            # LRU操作：如果已存在先移除，插入到末尾
            if x_i in lru_pc:
                lru_pc.pop(x_i)
            lru_pc[x_i] = True
            # 如果超过5个，移除最早插入的
            if len(lru_pc) > self.k:
                lru_pc.popitem(last=False)

        return X_processed


class ISVM2:
    def __init__(self, M=16, upper_bound=100):
        self.weight = np.zeros(M)
        self.upper_bound = upper_bound

    def svm_update_one(self, x, y):
        score = np.dot(self.weight, x)
        if score < self.upper_bound and score > -self.upper_bound:
            self.weight -= x * y

    def svm_predict(self, x):
        # X: [N]
        return int(np.dot(self.weight, x))
        
    
class ISVM_Cache2:
    def __init__(self, cache_size, k=5, M=128, upper_bound=100):
        self.cache_size = cache_size
        self.k = k
        self.M = M
        self.upper_bound = upper_bound
        self.isvm_table = {}
    
    def predict_online(self, X):
        """
        X_all: list, 长度N，请求（带obj_id）
        Y_true: list or np array, 长度N，真实标签（如belady结果[-1/1]）
        batch_size: int, e.g. 100
        return: list, 预测结果
        """
        Y_pred = []
        total = len(X)
        belady = Belady(self.cache_size)
        belady.initial(X)
        belady.resize(len(X))
        X1 = [i.obj_id for i in X]
        X_processed = self.preprocess_X(X1)
        Y_processed = [-1 if i else 1 for i in belady.result]
        for i in range(0, total):

            # 预测
            if X1[i] not in self.isvm_table:
                self.isvm_table[X1[i]] = ISVM2(self.M, self.upper_bound)
            Y_pred.append(self.isvm_table[X1[i]].svm_predict(X_processed[i]))
            # 按实际label增量训练
            self.isvm_table[X1[i]].svm_update_one(X_processed[i], Y_processed[i])
        return Y_pred
    
    def preprocess_X(self, X):
        lru_pc = OrderedDict()
        X_processed = []

        for i in range(len(X)):
            x_i = int(X[i] / 4096) & (self.M -1)  # 只保留最后4个bit (16种可能)
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

class Perceptron:
    def __init__(self, M):
        self.w = np.zeros(M)
        self.b = 0.0

    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                if yi * (xi @ self.w + self.b) <= 0:
                    self.w += yi * xi
                    self.b += yi

    def predict(self, X):
        return np.sign(X @ self.w + self.b).reshape(-1, 1)


class Perceptron_Cache:
    def __init__(self, cache_size,  k=5,M=16):
        self.cache_size = cache_size
        self.M = M
        self.k = k
        self.perceptron_table = {}

    def train(self, requests):
        X = [i.obj_id for i in requests]
        N = len(X)
        belady = Belady(self.cache_size)
        belady.initial(requests)
        belady.resize(N)
        X = self.preprocess(X)
        Y = np.array([1 if i else -1 for i in belady.result])
        epochs = 200
        self.perceptron.fit(X, Y, epochs)

    def predict(self, X):
        X = self.preprocess([i.obj_id for i in X])
        return [False if i < 0 else True for i in self.perceptron.predict(X)]

    def preprocess(self, X):
        lru_pc = OrderedDict()
        X_processed = []

        for i in range(len(X)):
            x_i = int(X[i] / 4096) & (self.M - 1)  # 只保留最后4个bit (16种可能)
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
    
class MultiVariableLinearClassifier:
    def fit(self, X, y):
        # 直接用贝叶斯推理里的概率差“赋权”，也支持直接用最小二乘
        freq_1 = (X[y==1].mean(axis=0))
        freq_m1 = (X[y==-1].mean(axis=0))
        self.w = freq_1 - freq_m1
        # 或者直接最小二乘法
        # self.w, *rest = np.linalg.lstsq(X, y, rcond=None)[0:2]
        self.b = 0  # 可以省略b，或设为类均值偏置

    def predict(self, X):
        score = X @ self.w + self.b
        pred = np.where(score > 0, 1, -1)
        return pred.reshape(-1,1)
    
class MultiVar_Cache:
    def __init__(self, cache_size,  k=5,M=16):
        self.cache_size = cache_size
        self.k = k
        self.M = M
        self.perceptron = MultiVariableLinearClassifier()

    def train(self, requests):
        X = [i.obj_id for i in requests]
        N = len(X)
        belady = Belady(self.cache_size)
        belady.initial(requests)
        belady.resize(N)
        X = self.preprocess(X)
        Y = np.array([1 if i else -1 for i in belady.result])
        self.perceptron.fit(X, Y)

    def predict(self, X):
        X = self.preprocess([i.obj_id for i in X])
        return [False if i < 0 else True for i in self.perceptron.predict(X)]

    def preprocess(self, X):
        lru_pc = OrderedDict()
        X_processed = []

        for i in range(len(X)):
            x_i = int(X[i] / 4096) & (self.M - 1)  # 只保留最后4个bit (16种可能)
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

class NaiveBayes:
    def fit(self, X, y):
        '''
        X: (N, M), 0/1
        y: (N,), -1/1
        '''
        X = np.asarray(X, dtype=np.uint8)
        y = np.asarray(y)
        self.M = X.shape[1]

        mask_pos = (y == 1)
        mask_neg = (y == -1)
        X_pos = X[mask_pos]
        X_neg = X[mask_neg]

        n_pos = np.sum(mask_pos)
        n_neg = np.sum(mask_neg)

        # 条件概率，若正负类全为0，避免除零出现0/0（此时概率无意义，但我们强行存1）
        self.P_x1_y1 = np.sum(X_pos, axis=0) / n_pos if n_pos > 0 else np.ones(self.M)
        self.P_x0_y1 = 1 - self.P_x1_y1
        self.P_x1_y_1 = np.sum(X_neg, axis=0) / n_neg if n_neg > 0 else np.ones(self.M)
        self.P_x0_y_1 = 1 - self.P_x1_y_1

        self.P_y1 = n_pos / len(y) if len(y) > 0 else 0.5
        self.P_y_1 = n_neg / len(y) if len(y) > 0 else 0.5

        # LUT建表
        self.lut = np.empty(2**self.M, dtype=np.int8)
        for i in range(2 ** self.M):
            x = np.array([(i >> j) & 1 for j in range(self.M)][::-1], dtype=np.uint8)
            # 正类概率
            prob1 = self.P_y1 * np.prod(np.where(x==1, self.P_x1_y1, self.P_x0_y1))
            # 负类概率
            prob2 = self.P_y_1 * np.prod(np.where(x==1, self.P_x1_y_1, 1-self.P_x1_y_1))
            # 哪个概率大
            self.lut[i] = 1 if prob1 >= prob2 else -1

        import matplotlib.pyplot as plt
        ind = np.arange(self.M)
        width = 0.35
        plt.bar(ind, self.P_x1_y1, width, label="P(x=1|y=1)")
        plt.bar(ind+width, self.P_x1_y_1, width, label="P(x=1|y=-1)")
        plt.xlabel("feature id")
        plt.ylabel("probability")
        plt.legend()
        plt.savefig("naivebayes.png")

    def predict(self, X):
        X = np.asarray(X, dtype=np.uint8)
        idxs = np.packbits(X, axis=1, bitorder="little")[:, 0]
        return self.lut[idxs]

class NaiveBayes_Cache:
    def __init__(self, cache_size,  k=5,M=16):
        self.cache_size = cache_size
        self.k = k
        self.M = M
        self.naivebayes = NaiveBayes()

    def train(self, requests):
        X = [i.obj_id for i in requests]
        N = len(X)
        belady = Belady(self.cache_size)
        belady.initial(requests)
        belady.resize(N)
        X = self.preprocess(X)
        Y = np.array([1 if i else -1 for i in belady.result])
        self.naivebayes.fit(X, Y)

    def predict(self, X):
        X = self.preprocess([i.obj_id for i in X])
        return [False if i < 0 else True for i in self.naivebayes.predict(X)]
    
    def predict_online(self, X, batch_size=100):
        """
        X_all: list, 长度N，请求（带obj_id）
        Y_true: list or np array, 长度N，真实标签（如belady结果[-1/1]）
        batch_size: int, e.g. 100
        return: list, 预测结果
        """
        Y_pred = []
        total = len(X)
        for start in range(0, total, batch_size):
            end = min(start+batch_size, total)
            X_batch = X[start:end]

            # 预测
            Y_batch_pred = self.predict(X_batch)
            Y_pred.extend(Y_batch_pred)
            # 按实际label增量训练
            self.train(X_batch)
        return Y_pred

    def preprocess(self, X):
        lru_pc = OrderedDict()
        X_processed = []

        for i in range(len(X)):
            x_i = int(X[i] / 4096) & (self.M - 1)  # 只保留最后4个bit (16种可能)
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
