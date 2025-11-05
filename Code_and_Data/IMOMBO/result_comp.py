from IMOMBO import *
from NSGAII import *
from MOMBO import *
from MOABC import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.ticker import ScalarFormatter

plt.rcParams['font.family'] = 'Times New Roman'
def plot_pareto_front_academic(domfit, title):
    x = []
    y = []
    z = []
    for sub_domfit in domfit:
        x.append([row[0] for row in sub_domfit])
        y.append([row[1] for row in sub_domfit])
        z.append([row[2] for row in sub_domfit])

    # 创建高分辨率图形，使用论文常用的 figsize
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置标记和颜色，以便更好地区分不同的数据集
    # markers = ['o', 's', '^']  # 圆形、方形、三角形
    # colors = ['#1f77b4', '#fe7e0e', '#2ba02c']  # 学术论文配色方案（ColorBlind友好）
    # labels = ['Ratio=0.0', 'Ratio=0.6', 'Ratio=0.8']

    #########比较不同人机协作比例########
    labels = ['Ratio=0.0', 'Ratio=0.6','Ratio=0.8']
    colors = ['#1b1bf5', '#629f57', '#eb5bf7']
    markers = ['o', 'o', 'o']

    #########比较算法性能##########
    # labels = ['IMOMBO', 'MOABC', 'MOMBO', 'NSGA-II']
    # colors = ['#1b1bf5', '#629f57', '#76605f', '#eb5bf7']
    # markers = ['o', 'o', 'o', 'o']

    # 确保图例透明度一致的关键：创建单独的散点对象用于图例
    legend_handles = []

    # 绘制数据点并设置样式
    for i in range(len(x)):
        # 绘制实际数据点（可以设置透明度）
        scatter = ax.scatter(
            x[i], y[i], z[i],
            marker=markers[i],
            s=70,  # 点的大小
            color=colors[i],
            edgecolors='white',  # 点的边缘颜色
            linewidth=0.7,  # 点的边缘线宽
            alpha=0.9  # 这里的透明度不会影响图例
        )

        # 创建一个完全不透明的点用于图例
        legend_handle = plt.Line2D(
            [0], [0],
            marker=markers[i],
            color='w',  # 白色线条（不可见）
            markerfacecolor=colors[i],  # 标记的填充颜色
            markeredgecolor='white',  # 标记的边缘颜色
            markeredgewidth=0.7,  # 标记的边缘宽度
            markersize=10,  # 标记大小
            linestyle='None',  # 无线条
            label=labels[i]
        )

        legend_handles.append(legend_handle)

    # 设置坐标轴标签，使用LaTeX风格的数学格式
    ax.set_xlabel('Cycle time', fontsize=12, labelpad=10)
    ax.set_ylabel('Ergonomic risk', fontsize=12, labelpad=10)
    ax.set_zlabel('Energy consumption', fontsize=12, labelpad=10)

    # 设置标题
    ax.set_title(title, fontsize=14, pad=15)

    # 设置网格
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # 调整轴刻度格式
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)  # 不使用科学计数法
        axis.set_major_formatter(formatter)

    # 调整视角以更好地展示三维结构
    ax.view_init(elev=30, azim=45)

    # 使用自定义的图例句柄，确保图例颜色一致
    ax.legend(
        handles=legend_handles,
        loc='best',  # 自动选择最佳位置
        fontsize=11,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        framealpha=0.9
    )

    # 调整布局
    plt.tight_layout()
    # fig.text(0.5, 0.00, "(b) P65", fontsize=12, ha='center')

    # 如果提供保存路径，则保存图片
    save_path = "生成图/case_pareto_front.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图形
    plt.show()

def plot_pareto_front1(domfit, title):
    x = []
    y = []
    z = []
    for sub_domfit in domfit:
        x.append([row[0] for row in sub_domfit])
        y.append([row[1] for row in sub_domfit])
        z.append([row[2] for row in sub_domfit])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    label = ['Ratio=0.0', 'Ratio=0.6','Ratio=0.8']
    colors = ['#1b1bf5', '#629f57', '#eb5bf7']
    markers = ['o', 'o', 'o']
    # labels = ['IMOMBO','MOABC','MOMBO','NSGA-II']
    # colors = ['#1b1bf5', '#629f57', '#76605f', '#eb5bf7']
    # markers = ['o', 'o', 'o', 'o']
    # 1b1bf5
    # 629f57
    # 76605f
    # eb5bf7


    # 绘制 domfit 的散点图
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], label=labels[i],c=colors[i],marker=markers[i])

    ax.set_xlabel('Cycle time')
    ax.set_ylabel('Ergonomic risk')
    ax.set_zlabel('Energy consumption')
    # ax.set_title(title)
    ax.legend()  # 添加图例

    save_path = "生成图/algorithm_pareto_front.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def select_top_crowded_solutions(npop, objnum, top_k=5):
    fit = [p.fitness for p in npop]
    crowding_distances = crowd(fit, len(npop), objnum)

    # 将npop中的解和对应的拥挤度距离组成元组
    solutions_with_distances = list(zip(npop, crowding_distances))
    # 根据拥挤度距离对元组进行降序排序
    sorted_solutions = sorted(solutions_with_distances, key=lambda x: x[1], reverse=True)
    # 选择拥挤度距离前10的解
    top_solutions = [solution for solution, _ in sorted_solutions[:top_k]]
    return top_solutions

if __name__ == '__main__':
    domfit = []
    fit1 = [[32.7, 4.1921, 8.0481],
            [35.9, 2.7758, 19.4774],
            [37.4, 3.0285, 12.3396],
            [38.6, 2.7926, 14.6787],
            [39.5, 2.3250, 17.4641],
            [40.3, 2.7806, 15.7170],
            [41.8, 2.2011, 16.3020],
            [42.4, 1.7589, 28.5772],
            [43.5, 1.1492, 36.7077],
            [44.5, 1.7963, 11.4318],
            [46.6, 0.6999, 45.8902],
            [48.0, 0.8280, 39.6480],
            [50.1, 0.7433, 37.2003],
            [55.9, 0.6442, 39.9937],
            [58.5, 0.8131, 19.3986]]

    fit2 = [[28.2, 4.2179, 17.2938],
            [31.6, 4.4279, 9.6373],
            [33.7, 3.7149, 6.9961],
            [34.2, 3.6989, 8.3421],
            [35.4, 2.8786, 14.4723],
            [37.2, 2.9019, 13.9710],
            [38.9, 2.4171, 16.5446],
            [40.0, 1.5867, 32.2621],
            [40.8, 2.6609, 9.3150],
            [41.7, 2.7711, 7.8084],
            [41.9, 1.5774, 30.559],
            [43.5, 1.2218, 29.8794],
            [51.8, 0.9176, 31.7310],
            [51.6, 0.7928, 26.0868],
            [56.7, 0.6986, 37.3455]
            ]

    fit3 = [[25.7, 4.7482, 20.4689],
            [26.7, 4.9164, 15.4680],
            [28.0, 4.2432, 17.5909],
            [28.5, 4.3266, 14.8083],
            [30.6, 3.3705, 16.9677],
            [31.4, 3.8618, 14.2385],
            [32.3, 3.9560, 11.9834],
            [33.1, 3.3760, 6.9420],
            [33.6, 3.6699, 13.1040],
            [38.4, 2.5748, 14.1471],
            [40.8, 2.1624, 28.1784],
            [42.8, 1.7321, 21.4151],
            [45.9, 1.9292, 16.7004],
            [52.5, 0.8135, 37.5555],
            [54.5, 0.7892, 28.8662]
            ]
    domfit.append(fit1)
    domfit.append(fit2)
    domfit.append(fit3)
    plot_pareto_front_academic(domfit, '')
    # 调用 maincode 函数并存储结果
'''
    算法比较
    domfit=[]

    Archivepop, hyper_archive, space_archive, curNAP = maincode()
    # 使用列表推导式存储 Archivepop 中个体的适应度
    # Archivepopn = select_top_crowded_solutions(Archivepop,3)
    domfit.append([archive.fitness for archive in Archivepop])

    #MOABC
    pop_2, hyper_pop_2, space_pop_2 = maincode3()
    # popn_2 = select_top_crowded_solutions(pop_2, 3)
    domfit.append([p.fitness for p in pop_2])

    #MOMBO
    pop_1, hyper_pop_1, space_pop_1 = maincode2()
    # popn_1 = select_top_crowded_solutions(pop_1, 3)
    domfit.append([p.fitness for p in pop_1])

    #NSGA-II
    pop, hyper_pop, space_pop = maincode1()
    # 使用列表推导式存储 pop 中个体的适应度
    # popn = select_top_crowded_solutions(pop,3)
    domfit.append([p.fitness for p in pop])

    plot_pareto_front_academic(domfit,'')
'''


'''

    domfit = []
    fit1 = [
[8.9,3.2841,8.9938],
[10.9,2.9216,7.4128],
[11.1,2.6225,8.8179],
[12.4,3.3182,2.6908],
[13.6,2.6982,3.1780],
[14.2,3.1665,0],
[15.4,2.1422,6.2392],
[16.0,1.0549,14.9609],
[16.9,0.3706,17.7032],
[20.2,0.3121,18.3299]
            ]

    fit2 = [
[10.2,2.9209,9.5397],
[11.6,2.7556,7.3837],
[14.7,2.2472,5.3760],
[21.8,0.1344,18.5287],
[27.4,0.7679,11.3878],
[28.9,0.1919,20.9762],
[32.3,1.3341,0],
[35.6,0.3681,13.6329],
[36.5,0.1344,18.5287],
[37.7,0.1529,20.2893]
    ]

    fit3=[
[10.2,2.8432,9.6603],
[11.4,2.7008,7.8105],
[12.3,2.9034,4.4211],
[13.9,2.3316,10.5764],
[14.8,2.3533,5.4799],
[15.0,1.8220,12.2628],
[16.4,1.0884,14.4493],
[16.8,0.7973,15.3147],
[17.8,0.6940,16.7555],
[19.6,1.7220,9.6653]
    ]
    fit4=[
[10.5,2.4284,10.8228],
[11.1,2.8726,10.0347],
[13.1,2.4789,6.1616],
[13.5,3.2661,0],
[14.4,2.8139,2.6244],
[17.0,0.9512,15.9229],
[18.9,2.0377,4.1328],
[19.1,1.1026,13.4989],
[19.8,2.0377,4.1328],
[21.9,0.2813,18.6657]
    ]

    domfit.append(fit1)
    domfit.append(fit2)
    domfit.append(fit3)
    domfit.append(fit4)

    plot_pareto_front_academic(domfit,'')

    plot_pareto_front_academic(domfit, '')
    # dfit1=[]
    # dfit2=[]
    # for i in range(len(fit2)):
    #     for j in range(len(fit1)):
    #         if dominates(fit2[i], fit1[j]):
    #             dfit1.append(fit2[i])
    #             dfit2.append(fit1[j])
    #
    # print('dfit1',dfit1)
    # print('dfit2',dfit2)

    # for i in range(len(fit1)):
    #     for j in range(len(fit2)):
    #         if dominates(fit1[i], fit2[j]):
    #             dfit1.append(fit1[i])
    #             dfit2.append(fit2[j])
    #
    # print('dfit1',dfit1)
    # print('dfit2',dfit2)
    #
    # for i in range(len(fit2)):
    #     for j in range(len(fit3)):
    #         if dominates(fit2[i], fit3[j]):
    #             dfit2.append(fit2[i])
    #             dfit3.append(fit3[j])
    #
    # print('dfit1',dfit1)
    # print('dfit3',dfit3)
    # for i in range(len(fit)):
    #     domfit.append(fit[i])


    # crowding_distances = [[0 for i in range(60)] for j in range(3)]
    # for i in range(3):
    #     crowding_distances = crowd(fit[i], len(fit[i]), 3)
    # # 创建解和对应拥挤度距离的索引对
    #     solution_distance_pairs = [(i, crowding_distances) for i in range(len(crowding_distances))]
    # # 按拥挤度距离降序排序
    #     sorted_solutions = sorted(solution_distance_pairs, key=lambda x: x[1], reverse=True)
    # # 选择拥挤度距离前10的解的索引
    #     top_indices = [idx for idx, _ in sorted_solutions[:15]]
    # # 选择对应的解
    #     top_solutions = [fit[i][idx] for idx in top_indices]
    #     domfit.append(top_solutions)
    #
    #     print(f"\n拥挤度排名前10的解 (fit[{i}]):")
    #     for i, solution in enumerate(top_solutions):
    #         print(f"{i + 1}. {solution} (拥挤度: {crowding_distances[top_indices[i]]})")
'''