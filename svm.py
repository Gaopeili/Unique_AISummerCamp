import numpy as np


class SVM():

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
            non_bound_alpha = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
            # 边界alpha
            bound_alpha = [i for i in range(self.m) if i not in non_bound_alpha]

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

                if self.E[i] < 0:
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
            eta = self.kernel(i1, i1) - 2 * self.kernel(i1, i2) + self.kernel(i2, i2)

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
