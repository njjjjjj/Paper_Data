import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 模拟四个不同算法的 Pareto 前沿解
np.random.seed(42)
# 算法 1 的 Pareto 前沿解
pareto_front_1 = np.random.rand(20, 3)
# 算法 2 的 Pareto 前沿解
pareto_front_2 = np.random.rand(20, 3) + 0.2
# 算法 3 的 Pareto 前沿解
pareto_front_3 = np.random.rand(20, 3) - 0.2
# 算法 4 的 Pareto 前沿解
pareto_front_4 = np.random.rand(20, 3) + 0.4

# 创建三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

'''
#1b1bf5
#629f57
#76605f
#eb5bf7
'''
# 定义更适合学术论文的颜色
colors = ['#1b1bf5', '#629f57', '#76605f', '#eb5bf7']
markers = ['o', 'o', 'o', 'o']
algorithms = ['Algorithm 1', 'Algorithm 2', 'Algorithm 3', 'Algorithm 4']
frontiers = [pareto_front_1, pareto_front_2, pareto_front_3, pareto_front_4]

# 绘制不同算法的 Pareto 前沿解
for i in range(4):
    ax.scatter(frontiers[i][:, 0], frontiers[i][:, 1], frontiers[i][:, 2], c=colors[i], marker=markers[i],
               label=algorithms[i])

# 设置坐标轴标签和标题
ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')
ax.set_title('Comparison of 3D Pareto Frontiers of Four Algorithms')

# 显示图例
ax.legend()

# 显示图形
plt.show()
