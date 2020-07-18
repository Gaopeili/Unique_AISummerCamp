import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from validation import *

class SVM_by_SMO():

    def __init__(self, max_iter=200, C=1):
        # 最多迭代数
        self.max_iter = max_iter
        # w
        self.w = 0
        # b
        self.b = 0
        # 松弛变量
        self.C = C
        # 精度(xi为其希腊字母)
        self.xi = 0.01

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        # 样本个数和X的维度
        self.m, self.n = X_train.shape
        # X
        self.X = X_train
        # Y
        self.Y = Y_train
        # alpha
        self.alpha = np.zeros(self.m)
        #  error_i = w·x_i+b-y_i
        self.E = np.array([self.g(i) - self.Y[i] for i in range(self.m)])

        for _iter in range(self.max_iter):

            # 非边界alpha
            non_bound_alpha = [
                i for i in range(
                    self.m) if 0 < self.alpha[i] < self.C]
            # 边界alpha
            bound_alpha = [
                i for i in range(
                    self.m) if i not in non_bound_alpha]

            # 先非边界alpha后边界alpha
            non_bound_alpha.extend(bound_alpha)

            # 选取的两个变量的下标
            i1 = i2 = None

            for i in non_bound_alpha:
                # 满足KKT条件则跳出
                if self.KKT(i):
                    continue

                # 第一个优化alpha下标
                i1 = i
                # 使第二个优化变量优化步长最大

                if self.E[i1] < 0:
                    i2 = self.E.argmax()
                else:
                    i2 = self.E.argmin()
                break

            # 待优化变量
            E1_old = self.E[i1]
            E2_old = self.E[i2]
            alpha1_old = self.alpha[i1]
            alpha2_old = self.alpha[i2]
            Y1 = self.Y[i1]
            Y2 = self.Y[i2]

            # eta(希腊字母) = K11-2K12+K22
            eta = self.kernel(i1, i1) - 2 * self.kernel(i1,
                                                        i2) + self.kernel(i2, i2)

            # eta小于等于0则重新找优化变量
            if eta <= 0:
                continue

            # 找到L,H
            if Y1 == Y2:
                L = max(0, alpha1_old + alpha2_old - self.C)
                H = min(self.C, alpha1_old + alpha2_old)
            else:
                L = max(0, alpha2_old - alpha1_old)
                H = min(self.C, alpha2_old - alpha1_old + self.C)

            # 更新alpha2
            alpha2_new_unclipped = alpha2_old + Y2 * (E1_old - E2_old) / eta
            alpha2_new = self.update_alpha2(alpha2_new_unclipped, L, H)

            # 更新alpha1
            alpha1_new = alpha1_old + Y1 * Y2 * (alpha2_old - alpha2_new)

            # 更新b
            b1_new = self.b - E1_old - Y1 * self.kernel(i1, i1) * (alpha1_new - alpha1_old) - Y2 * self.kernel(i1,
                                                                                                               i2) * (
                alpha2_new - alpha2_old)
            b2_new = self.b - E2_old - Y1 * self.kernel(i1, i2) * (alpha1_new - alpha1_old) - Y2 * self.kernel(i2,
                                                                                                               i2) * (
                alpha2_new - alpha2_old)

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            # 更新E
            self.E = self.E + Y1 * (alpha1_new - alpha1_old) * np.array([self.kernel(i1, j) for j in range(self.m)]) + \
                Y2 * (alpha2_new - alpha2_old) * np.array([self.kernel(i2, j) for j in range(self.m)]) + np.array(
                [b_new - self.b for _ in range(self.m)])

            # 修改到原变量中
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

        # 获取w
        self.w = np.dot(self.alpha, self.Y.reshape(-1, 1) * self.X)

    # 更新alpha2
    def update_alpha2(self, alpha2_new_unclipped, L, H):
        # 越界则取边界
        if alpha2_new_unclipped < L:
            return L
        elif alpha2_new_unclipped > H:
            return H
        # 没有越界则不变
        else:
            return alpha2_new_unclipped

    # KKT条件
    def KKT(self, i):
        if self.alpha[i] == 0:
            return self.Y[i] * self.g(i) >= 1 - self.xi
        elif self.alpha[i] == self.C:
            return self.Y[i] * self.g(i) <= 1 + self.xi
        else:
            return 1 - self.xi <= self.Y[i] * self.g(i) <= 1 + self.xi

    # g(i) = w·x+b = SUM(a_j*y_j*K_ij)
    def g(self, i):
        sum = 0
        for j in range(self.m):
            sum += self.alpha[j] * self.Y[j] * self.kernel(i, j)
        return sum + self.b

    # 核函数
    def kernel(self, i, j):
        return np.dot(self.X[i], self.X[j])

    def predict(self, X):
        return np.sign(np.dot(self.w, X) + self.b)

    def score(self, X_test, Y_test):
        right_num = 0
        for i in range(len(X_test)):
            y = self.predict(X_test[i])
            if y == Y_test[i]:
                right_num += 1
        return right_num / len(X_test)


class MultiSVM():

    def __init__(self):
        self.W = None

    def gradients(self, X: np.ndarray, Y: np.ndarray, reg=0.05):
        """
        :param X: (N,D)
        :param Y: (N)
        :return: 损失值，梯度
        """
        """
        Loss function -- Hinge Loss
        L = max(0,S_ij-S_y_i+1) + 0.5*reg*W.T*W for i in range(N) j in range(C) but j!= y_i
        其中S_ij = X_i * W_j 表示X_i错误分类时的分数
        S_y_i = X_j * W_y_i 表示X_i正确分类时的分数
        对W求导数:
        L_i 对 W_j求导数:
            X_i
        L_i 对 W_y_i求导数:
            -count(S_ij-S_y_i+1>0)*X_i

        L 对W_j求导数:
            分类错误或者确性度不足1时的X_i for i in range(N)
        L 对w_yi求导数:
            分类错误或者确信度不足1的X_i的类别和*X_i for i in range(N)
        """
        # N
        N = len(Y)
        # 对每一类别的得分(N,C)
        scores = X.dot(self.W)
        # (N,C)
        margin = scores - scores[np.arange(N), Y].reshape(N, -1) + 1
        # 小于等于0的部分设置成0
        margin = np.maximum(0, margin)
        # 正确的类设置成0
        margin[np.arange(N),Y] = 0
        # 损失值
        Loss = np.sum(margin)/N+0.5*reg*np.sum(self.W**2)
        # 大于0的部分设置成1为了方便计算分类错误或者确信度不足1的X_i的类别个数
        margin[margin > 0] = 1
        # 计算分类错误或者确信度不足1的X_i的类别个数
        wrong_sum = np.sum(margin, axis=1)
        # 将正确类设置成分类错误或者确信度不足1的X_i的类别个数
        margin[np.arange(N), Y] = -wrong_sum
        # 梯度
        gradient = np.dot(X.T, margin)/N + reg*self.W
        return Loss,gradient

    def fit(self, X_train, Y_train, learning_rate=0.1,
            max_iter=1000, batch_size=20, reg=0.05):
        """
        :param X_train: 训练数据 (N,D)
        :param Y_train: 训练数据 (N,)
        :param learning_rate: 学习率 int
        :param max_iter: 最大迭代数  int
        :param batch_size: 随机梯度个数
        :param W: (D,C)
        :param D: 样本维度
        :param C: 样本类别个数
        :param N: 样本个数
        :return:
        """

        self.N, self.D = X_train.shape
        self.C = np.max(Y_train)+1
        self.W = np.random.random((self.D,self.C))

        Loss_history =  []
        for _ in range(max_iter):

            index = np.random.choice(self.N, batch_size)
            X_batch = X_train[index]
            Y_batch = Y_train[index]

            Loss,gradient = self.gradients(X_batch,Y_batch,reg)

            Loss_history.append(Loss)
            self.W -= learning_rate * gradient

        return Loss_history

    def predict(self, X_test):
        score = np.dot(X_test, self.W)
        label = np.argmax(score)
        return label

    def score(self, X_test, Y_test):
        correct_num = 0
        for i in range(len(X_test)):
            Y_pred = self.predict(X_test[i])
            if Y_pred == Y_test[i]:
                correct_num += 1
        return correct_num / len(Y_test)


iris = load_iris()

X = np.array(iris.data)
Y = np.array(iris.target)
X = np.hstack((X,np.ones(len(X)).reshape(-1,1)))


index = np.random.choice(100, 100)
X_train = X[index[:50]]
Y_train = Y[index[:50]]
X_test = X[index[50:]]
Y_test = Y[index[50:]]

svm = MultiSVM()
Loss_history = svm.fit(X_train, Y_train)
score = svm.score(X_test,Y_test)
print(score)

