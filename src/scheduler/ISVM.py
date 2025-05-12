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
            Y_pred.append(self.isvm_table[X1[i]].svm_predict(X_processed[i]) // 50)
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

    def fit_one(self, x, y):
        """
        x: (M,)
        y: 标量（+1或-1）
        单样本感知机在线更新
        """
        if y * (np.dot(x, self.w) + self.b) <= 0:
            self.w += y * x
            self.b += y

    def predict_one(self, x):
        """
        x: (M,)
        输出预测标签（+1或-1），如等于0返回1
        """
        return np.dot(x, self.w) + self.b
        

class Perceptron_Cache:
    def __init__(self, cache_size,  k=5,M=16):
        self.cache_size = cache_size
        self.M = M
        self.k = k
        self.perceptron_table = {}

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
            if xi not in self.perceptron_table:
                self.perceptron_table[xi] = Perceptron(self.M)
            Y_pred.append(int(self.perceptron_table[xi].predict_one(X_processed[i])))
            # 按实际label增量训练
            self.perceptron_table[xi].fit_one(X_processed[i], Y[i])

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
    
class MultiVariableLinearClassifier:
    def __init__(self, M):
        self.M = M
        self.sum_1 = np.zeros(M)
        self.sum_m1 = np.zeros(M)
        self.count_1 = 0
        self.count_m1 = 0
        self.w = np.zeros(M)

    def fit_one(self, x, y):
        """
        x: (M,)
        y: 1 或 -1
        """
        if y == 1:
            self.sum_1 += x
            self.count_1 += 1
        elif y == -1:
            self.sum_m1 += x
            self.count_m1 += 1
        # 更新参数（当前均值差）：
        if self.count_1 > 0 and self.count_m1 > 0:
            freq_1 = self.sum_1 / self.count_1
            freq_m1 = self.sum_m1 / self.count_m1
            self.w = freq_1 - freq_m1

    def predict_one(self, x):
        score = np.dot(x, self.w)
        return score
    
class MultiVar_Cache:
    def __init__(self, cache_size,  k=5,M=16):
        self.cache_size = cache_size
        self.k = k
        self.M = M
        self.multivar_table = {}

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
            if xi not in self.multivar_table:
                self.multivar_table[xi] = MultiVariableLinearClassifier(self.M)
            Y_pred.append(int(self.multivar_table[xi].predict_one(X_processed[i])))
            # 按实际label增量训练
            self.multivar_table[xi].fit_one(X_processed[i], Y[i])

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

class NaiveBayes:
    def __init__(self, M):
        self.M = M
        # 初始化计数器
        self.count_y1 = 0         # y=1的样本数
        self.count_yneg1 = 0      # y=-1的样本数
        self.count_x_y1 = np.zeros(M, dtype=np.int32)     # x=1且y=1的次数
        self.count_x_yneg1 = np.zeros(M, dtype=np.int32)  # x=1且y=-1的次数

    def fit_one(self, x, y):
        """
        x: (M,), 0/1
        y: -1 或 1
        """
        if y==1:
            self.count_y1 += 1
            self.count_x_y1 += x
        elif y==-1:
            self.count_yneg1 += 1
            self.count_x_yneg1 += x

    def predict_one(self, x, laplace=1e-6):
        """
        x: (M,), 0/1
        laplace: 拉普拉斯平滑，为避免分母为0，默认为1e-6（或1)
        return: 1或-1
        """
        # P(y=1)
        total = self.count_y1 + self.count_yneg1
        if total == 0:         # 没有数据
            return 1  # 或任选
        P_y1 = self.count_y1 / total
        P_yneg1 = self.count_yneg1 / total

        # 条件概率
        # 拉普拉斯平滑: +laplace, 分母也+2*laplace
        P_x1_y1 = (self.count_x_y1 + laplace) / (self.count_y1 + 2 * laplace)
        P_x0_y1 = 1 - P_x1_y1
        P_x1_yneg1 = (self.count_x_yneg1 + laplace) / (self.count_yneg1 + 2 * laplace)
        P_x0_yneg1 = 1 - P_x1_yneg1

        prob1 = P_y1
        probneg1 = P_yneg1
        # 贝叶斯公式 连乘
        for j in range(self.M):
            if x[j]:
                prob1 *= P_x1_y1[j]
                probneg1 *= P_x1_yneg1[j]
            else:
                prob1 *= P_x0_y1[j]
                probneg1 *= P_x0_yneg1[j]
        total = prob1 + probneg1
        prob1_norm = prob1 / total
        probneg1_norm = probneg1 / total
        return prob1_norm - probneg1_norm
    
class NaiveBayes_Cache:
    def __init__(self, cache_size,  k=5,M=16):
        self.cache_size = cache_size
        self.k = k
        self.M = M
        self.naivebayes_table = {}

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
            if xi not in self.naivebayes_table:
                self.naivebayes_table[xi] = NaiveBayes(self.M)
            Y_pred.append(int(self.naivebayes_table[xi].predict_one(X_processed[i])))
            # 按实际label增量训练
            self.naivebayes_table[xi].fit_one(X_processed[i], Y[i])

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
