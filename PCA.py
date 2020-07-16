import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


def MyPCA(X: np.ndarray, p: float = 0,
          mode: str = "eig", n_components: int = 2):
    """
    实现PCA降维处理

    :param X: 样本矩阵 (m,n) m为样本个数 n为样本维度数
    :param p: 信息量保存百分比
    :param mode "eig"：特征值分解  "svd"：奇异值分解
    :return: 返回降维之后的样本矩阵
    """

    def normalize(X: np.ndarray):
        """
        样本规范化预处理

        :param X:
        :return: 返回每一维度的均值为0，方差为1的样本矩阵
        """

        # 每个维度的平均值
        means = np.mean(X, axis=0)
        # 每个维度的标准差
        std = np.std(X, axis=0)

        # 返回规范化后的样本矩阵
        return (X - means)  # /std

    def cal_sigma(X: np.ndarray):
        "计算并返回协方差矩阵"

        return X.T.dot(X) / (len(X) - 1)

    def cal_eig(sigma: np.ndarray):
        "计算并返回降序的特征值以及特征向量"

        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        # 降序排列索引值
        index = eigenvalues.argsort()[::-1]

        # 降序排列的特征值以及特征向量
        return eigenvalues[index], eigenvectors[index]

    def cal_svd(X_new):
        "计算并返回奇异值分解的特征值以及对应的特征向量"

        # 奇异值分解，注意svd_vector2返回的是列向量的转置
        svd_vector1, svd_values, svd_vector2 = np.linalg.svd(X_new)

        # 转置
        svd_vector2 = svd_vector2.T

        # 降序索引
        index = svd_values.argsort()[::-1]

        # 返回
        return svd_values[index], svd_vector2[index]

    def fun(X: np.ndarray):

        # 规范化
        x = normalize(X)

        eigenvalues, eigenvectors = None, None

        if mode == "eig":

            # 协方差矩阵
            sigma = cal_sigma(x)
            # 特征值 & 特征向量
            eigenvalues, eigenvectors = cal_eig(sigma)

        elif mode == "svd":

            # 奇异值分解的新x
            x_new = x / (len(x) - 1) ** 0.5
            # 特征值 & 特征向量
            eigenvalues, eigenvectors = cal_svd(x_new)

        # 记录累加概率
        P = 0
        # 特征求和作为分母
        eig_val_sum = np.sum(eigenvalues)

        # 如果输入了p则按照p计算
        if p != 0:

            for k in range(len(eigenvalues)):
                P += eigenvalues[k] / eig_val_sum
                if P >= p:
                    # 列向量是特征向量
                    eigenvectors = eigenvectors[:, :k]
                    break

        # 否则按照n_components计算
        else:

            eigenvectors = eigenvectors[:, :n_components]

        # 返回降维结果
        return x.dot(eigenvectors)

    return fun(X)


iris = load_iris()
X = np.array(iris.data)
Y = np.array(iris.target)

array = np.array(X)

result = MyPCA(array, mode="eig")
plt.subplot(311)
plt.scatter(result[:50, 0], result[:50, 1], label="0")
plt.scatter(result[50:100, 0], result[50:100, 1], label="1")
plt.scatter(result[100:, 0], result[100:, 1], label="2")
plt.title("PCA of my own——eigen")
plt.legend()

result = MyPCA(array, mode="svd")
plt.subplot(312)
plt.scatter(result[:50, 0], result[:50, 1], label="0")
plt.scatter(result[50:100, 0], result[50:100, 1], label="1")
plt.scatter(result[100:, 0], result[100:, 1], label="2")
plt.title("PCA of my own——svd")
plt.legend()

plt.subplot(313)
clf = PCA(n_components=2)
result1 = clf.fit_transform(X)
plt.scatter(result1[:50, 0], result[:50, 1], label="0")
plt.scatter(result1[50:100, 0], result[50:100, 1], label="1")
plt.scatter(result1[100:, 0], result[100:, 1], label="2")
plt.title("PCA of sk-learn")
plt.legend()
plt.show()
