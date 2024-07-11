import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 生成两个类别的数据
def generate_binary_data(num_samples=100, noise=0.1):
    np.random.seed(42)
    
    # 生成类别1的数据，平均分布在半径为1的圆周上
    theta_1 = np.linspace(0, np.pi, num_samples // 2)
    class_1_x = np.cos(theta_1) + np.random.normal(0, noise, len(theta_1))
    class_1_y = np.sin(theta_1) + np.random.normal(0, noise, len(theta_1))
    
    # 生成类别2的数据，平均分布在半径为2的圆周上
    theta_2 = np.linspace(0, np.pi, num_samples // 2)
    class_2_x = 2 * np.cos(theta_2) + np.random.normal(0, noise, len(theta_2))
    class_2_y = 2 * np.sin(theta_2) + np.random.normal(0, noise, len(theta_2))
    
    # 合并数据和标签
    X = np.concatenate([np.column_stack((class_1_x, class_1_y)),
                        np.column_stack((class_2_x, class_2_y))])
    y = np.concatenate([np.zeros(len(theta_1)), np.ones(len(theta_2))])
    
    return X, y

# 生成二分类数据集
X, y = generate_binary_data(num_samples=400, noise=0.5)

# 训练支持向量机模型 (RBF kernel)
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X, y)

# 绘制数据集和决策边界
def plot_decision_boundary(model, X, y, save_path=None):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', marker='o')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', marker='x')

    
    # 绘制决策边界
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 生成网格点
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    
    # 获取决策边界
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
    
    plt.legend()

    
    # 移除 x 轴和 y 轴的刻度
    plt.xticks([])
    plt.yticks([])
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=1500, bbox_inches='tight')
    
    # 保存决策边界数据到文件
    if save_path:
        np.savez(save_path.replace('.png', '_data.npz'), xx=xx, yy=yy, Z=Z)

# 绘制数据集和决策边界，并保存图像和决策边界数据到文件
plot_decision_boundary(svm_model, X, y, save_path='decision_boundary_plot.png')
