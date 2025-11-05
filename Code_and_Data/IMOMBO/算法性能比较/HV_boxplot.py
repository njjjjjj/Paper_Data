import matplotlib.pyplot as plt
import numpy as np

# 设置学术论文样式
plt.style.use('seaborn-v0_8-whitegrid')  # 使用白色网格背景
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 使用学术论文常用字体
    'font.size': 12,                   # 正文字号
    'axes.labelsize': 14,              # 坐标轴标签字号
    'axes.titlesize': 16,              # 标题字号
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300                  # 输出分辨率
})

# 生成示例数据（4个组，每组10个数据点）
###P24###
data = [[0 for i in range(10)] for i in range(4)]
#P24
# data[0]=[1498.13,1486.99,1505.41,1474.35,1441.66,1489.15,1498.99,1468.82,1472.77,1480.55]
# data[1]=[997.58,1040.28,1027.74,999.34,1015.37,1029.21,1038.76,1007.99,1028.24,1014.33]
# data[2]=[1163.35,1093.04,1177.82,1018.92,1065.95,954.68,979.12,1028.70,1007.18,1025.41]
# data[3]=[1434.73,1421.61,1355.08,1443.64,1424.99,1474.05,1464.02,1471.99,1446.33,1438.99]

data[0]=[1516.82,1492.56,1511.00,1469.62,1458.70,1491.29,1487.62,1486.76,1462.45,1473.20]
data[1]=[1451.41,1467.29,1464.95,1409.92,1472.27,1447.48,1434.72,1429.70,1475.10,1436.43]
data[2]=[1431.23,1405.52,1347.50,1401.31,1441.10,1432.95,1462.28,1397.90,1406.07,1459.74]
data[3]=[1408.19,1427.53,1457.97,1418.96,1480.15,1423.68,1434.35,1451.12,1474.42,1430.40]

#P65
# data[0]=[5805.22,5605.89,5739.72,5873.30,5824.37,5910.97,5803.16,5725.61,5604.87,5730.25]
# data[1]=[5355.46,5168.27,5253.53,5439.56,5609.43,5516.65,5522.54,5051.11,5211.78,5184.55]
# data[2]=[5398.60,5640.44,5617.78,5761.70,5699.74,5553.99,5548.93,5345.88,5550.60,5552.63]
# data[3]=[5675.24,5699.15,5478.59,5488.82,5683.37,5537.38,5568.01,5433.86,5649.23,5660.15]
# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))  # 设置合适的长宽比

# 绘制箱线图
box = ax.boxplot(
    data,
    patch_artist=True,          # 启用填充颜色
    labels=['IMOMBO','MOABC','MOMBO','NSGA-II'],  # 设置组标签
    widths=0.6,                # 控制箱体宽度
    showmeans=True,             # 显示均值标记
    meanprops={'marker': 'D',   # 均值点样式
               'markerfacecolor': 'gold',
               'markersize': 8},
    medianprops={'color': 'red', 'linewidth': 2},  # 中位数线样式
    boxprops={'facecolor': 'lightblue', 'color': 'darkblue'},  # 箱体样式
    whiskerprops={'color': 'darkblue', 'linewidth': 2},
    capprops={'color': 'darkblue', 'linewidth': 2}
)

# 添加辅助元素
# ax.set_title('Distribution Comparison Across Groups')  # 标题
ax.set_ylabel('HV')            # Y轴标签
ax.set_xlabel('(a) P24')
# ax.set_xlabel('(b) P65')                 # X轴标签

# 设置Y轴范围（根据数据自动调整）
ax.set_ylim(bottom=np.min(data)-0.5, top=np.max(data)+0.5)

# 优化布局并保存
plt.tight_layout()
plt.savefig('P24_boxplot.png', dpi=300, bbox_inches='tight')  # 保存高分辨率图片
plt.show()