import numpy as np
import matplotlib.pyplot as plt
import matplotlib

'''
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 将非支配解数据导入
data=[[34.099999999999994, 4.218559302285476, 2.3787], [26.099999999999998, 5.177518687198074, 12.9978], [49.099999999999994, 0.9481977621591728, 40.6352], [36.7, 2.5030166148475006, 24.4189], [50.2, 0.795882467305086, 43.9975], [51.800000000000004, 0.6824371672526918, 38.645300000000006], [85.3, 0.11615828394161723, 47.206], [81.1, 0.24632712982770175, 33.996900000000004], [79.29999999999998, 0.3650915153206483, 28.0866], [75.6, 0.7958508635655989, 18.891900000000003], [27.5, 5.257097454428511, 8.7905], [73.1, 0.2682119751904912, 34.6304], [46.4, 2.202871325262614, 4.995], [41.5, 2.9215633169152198, 0], [67.6, 0.34627244920629896, 41.0197], [57.300000000000004, 1.0636737717954485, 22.843800000000005], [66.0, 0.9304004396413563, 17.1459], [71.8, 0.8720099263108058, 7.8195000000000014], [35.5, 3.5829722063529337, 10.5969], [31.499999999999996, 3.721737909893579, 18.2052], [32.99999999999999, 3.9627185575295045, 12.1092], [34.8, 4.649804766085301, 2.0574], [39.800000000000004, 3.129348504735805, 2.7513000000000005], [64.19999999999999, 0.5620391346536147, 36.65370000000001], [39.1, 1.9061420108474416, 25.684], [60.3, 1.198846516280567, 7.371], [43.1, 1.7675722322276128, 23.350099999999998], [61.699999999999996, 1.2315186071449695, 6.7587], [51.599999999999994, 1.3255534018373853, 20.124], [28.499999999999996, 4.9042531564233425, 3.5883000000000003], [53.099999999999994, 2.055616149817644, 0], [27.5, 4.742355908506387, 9.183200000000001], [44.49999999999999, 1.6390383962282644, 16.150799999999997], [38.6, 2.7112438644264465, 13.9976], [73.60000000000001, 0.25074878184756816, 33.02080000000001], [29.4, 3.4885367960775078, 22.2297], [41.7, 2.3287960308201257, 11.259000000000002], [45.099999999999994, 1.4293406797506514, 30.274600000000003], [62.59999999999999, 0.657150850652239, 31.6145], [63.7, 0.4716281975105089, 41.8609], [58.2, 0.9973385476865216, 21.274800000000003], [35.8, 3.8820060199404893, 1.7928], [36.5, 3.0626656423731986, 9.855], [36.0, 3.7593783283762847, 6.075], [80.6, 0.23920213485482317, 35.3999], [47.4, 1.5363474616206834, 20.5476], [32.9, 4.400716854062217, 2.5893], [29.9, 4.071446509179053, 14.1218], [55.699999999999996, 0.8330223811631664, 32.3043], [54.60000000000001, 1.8008726305151592, 21.2451], [42.1, 2.7526371431080134, 2.8134], [33.4, 3.449960283420279, 18.0777], [75.3, 0.4025394042519417, 29.088900000000002], [37.3, 2.444138560225867, 16.3533], [48.7, 1.7302171444147512, 15.255400000000002], [70.3, 0.4937848803236927, 25.9024], [39.699999999999996, 3.328859194410113, 0], [59.19999999999999, 0.7299926572442036, 36.4483], [48.3, 1.961664292390687, 10.8693], [27.0, 4.962394588285242, 13.8492]]

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['周期时间', '工效学风险', '机器能耗'])

# 移除重复的解（如果有）
df = df.drop_duplicates()

# 计算每个目标的归一化值（用于综合评分）
for column in df.columns:
    min_val = df[column].min()
    max_val = df[column].max()
    # 归一化为0-1范围，值越小越好
    df[f"{column}_标准化"] = (df[column] - min_val) / (max_val - min_val)

# 计算归一化后的综合分数（假设三个目标同等重要）
df['综合分数'] = df['周期时间_标准化'] + df['工效学风险_标准化'] + df['机器能耗_标准化']

# 找出按各个目标排序的前10个解
top10_cycle_time = df.sort_values(by='周期时间').head(20)
top10_ergonomics = df.sort_values(by='工效学风险').head(20)
top10_energy = df.sort_values(by='机器能耗').head(20)
top10_overall = df.sort_values(by='综合分数').head(20)

# 创建结果输出
print("按周期时间排序的前10个非支配解：")
print(top10_cycle_time[['周期时间', '工效学风险', '机器能耗']].to_string(index=False))
print("\n按工效学风险排序的前10个非支配解：")
print(top10_ergonomics[['周期时间', '工效学风险', '机器能耗']].to_string(index=False))
print("\n按机器能耗排序的前10个非支配解：")
print(top10_energy[['周期时间', '工效学风险', '机器能耗']].to_string(index=False))
print("\n按综合分数排序的前10个非支配解：")
print(top10_overall[['周期时间', '工效学风险', '机器能耗', '综合分数']].to_string(index=False))

# 创建可视化
fig = plt.figure(figsize=(18, 12), dpi=300)

# 1. 三维散点图
ax1 = fig.add_subplot(221, projection='3d')
scatter = ax1.scatter(df['周期时间'], df['工效学风险'], df['机器能耗'],
                     c=df['综合分数'], cmap='viridis', s=50, alpha=0.8)
ax1.set_xlabel('周期时间', fontsize=12)
ax1.set_ylabel('工效学风险', fontsize=12)
ax1.set_zlabel('机器能耗', fontsize=12)
ax1.set_title('非支配解的三维分布', fontsize=14)
cbar = plt.colorbar(scatter, ax=ax1, pad=0.1)
cbar.set_label('综合分数（越小越好）', fontsize=10)

# 标记三个目标的最优解
min_cycle = df.loc[df['周期时间'].idxmin()]
min_ergo = df.loc[df['工效学风险'].idxmin()]
min_energy = df.loc[df['机器能耗'].idxmin()]
min_overall = df.loc[df['综合分数'].idxmin()]

ax1.scatter([min_cycle['周期时间']], [min_cycle['工效学风险']], [min_cycle['机器能耗']],
           color='red', s=100, marker='*', label='最小周期时间')
ax1.scatter([min_ergo['周期时间']], [min_ergo['工效学风险']], [min_ergo['机器能耗']],
           color='blue', s=100, marker='*', label='最小工效学风险')
ax1.scatter([min_energy['周期时间']], [min_energy['工效学风险']], [min_energy['机器能耗']],
           color='green', s=100, marker='*', label='最小机器能耗')
ax1.scatter([min_overall['周期时间']], [min_overall['工效学风险']], [min_overall['机器能耗']],
           color='black', s=100, marker='*', label='最佳综合解')
ax1.legend(loc='upper left', fontsize=10)

# 2. 二维投影：周期时间 vs 工效学风险
ax2 = fig.add_subplot(222)
scatter2 = ax2.scatter(df['周期时间'], df['工效学风险'], c=df['机器能耗'],
                      cmap='plasma', s=50, alpha=0.7)
ax2.set_xlabel('周期时间', fontsize=12)
ax2.set_ylabel('工效学风险', fontsize=12)
ax2.set_title('周期时间 vs 工效学风险（颜色表示机器能耗）', fontsize=14)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('机器能耗', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)

# 3. 二维投影：周期时间 vs 机器能耗
ax3 = fig.add_subplot(223)
scatter3 = ax3.scatter(df['周期时间'], df['机器能耗'], c=df['工效学风险'],
                      cmap='inferno', s=50, alpha=0.7)
ax3.set_xlabel('周期时间', fontsize=12)
ax3.set_ylabel('机器能耗', fontsize=12)
ax3.set_title('周期时间 vs 机器能耗（颜色表示工效学风险）', fontsize=14)
cbar3 = plt.colorbar(scatter3, ax=ax3)
cbar3.set_label('工效学风险', fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.7)

# 4. 二维投影：工效学风险 vs 机器能耗
ax4 = fig.add_subplot(224)
scatter4 = ax4.scatter(df['工效学风险'], df['机器能耗'], c=df['周期时间'],
                      cmap='cividis', s=50, alpha=0.7)
ax4.set_xlabel('工效学风险', fontsize=12)
ax4.set_ylabel('机器能耗', fontsize=12)
ax4.set_title('工效学风险 vs 机器能耗（颜色表示周期时间）', fontsize=14)
cbar4 = plt.colorbar(scatter4, ax=ax4)
cbar4.set_label('周期时间', fontsize=10)
ax4.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()
plt.suptitle('非支配解的多维分析', fontsize=16, y=1.02)

# 保存图形
plt.savefig('pareto_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('pareto_analysis.pdf', bbox_inches='tight')

# 显示图形
plt.show()

# 创建多目标决策支持表格
print("\n===== 多目标决策支持表格 =====")
print("\n1. 针对不同优先级的推荐解决方案：")

# 定义不同偏好场景的权重
preference_scenarios = {
    "均衡偏好": [1/3, 1/3, 1/3],
    "周期时间优先": [0.6, 0.2, 0.2],
    "工效学风险优先": [0.2, 0.6, 0.2],
    "机器能耗优先": [0.2, 0.2, 0.6],
    "生产效率优先": [0.5, 0.3, 0.2],
    "人因工程优先": [0.3, 0.5, 0.2],
    "节能优先": [0.3, 0.2, 0.5]
}

# 计算每个场景下的加权得分
for scenario, weights in preference_scenarios.items():
    df[f'{scenario}_得分'] = (weights[0] * df['周期时间_标准化'] +
                           weights[1] * df['工效学风险_标准化'] +
                           weights[2] * df['机器能耗_标准化'])
    best_solution = df.loc[df[f'{scenario}_得分'].idxmin()]
    print(f"\n{scenario}场景的最佳解决方案:")
    print(f"  周期时间: {best_solution['周期时间']:.2f}")
    print(f"  工效学风险: {best_solution['工效学风险']:.4f}")
    print(f"  机器能耗: {best_solution['机器能耗']:.2f}")
    print(f"  加权得分: {best_solution[f'{scenario}_得分']:.4f}")

# 计算目标间相关性
correlation_matrix = df[['周期时间', '工效学风险', '机器能耗']].corr()
print("\n\n2. 目标函数间相关性分析：")
print(correlation_matrix)

'''

# 箱线图的代码
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

# 模拟数据生成（您需要替换为实际数据）
np.random.seed(42)
n_solutions = 30  # 每组解决方案的数量

# 生成三种协作比例的优化结果
# 协作比例 0.0 - 假设性能较差

cycle_time_0 = [
32.7,
35.9,
37.4,
38.6,
39.5,
40.3,
41.8,
42.4,
43.5,
44.5,
46.6,
48.0,
50.1,
55.9,
58.5
]
ergonomics_0 = [
4.1921,
2.7758,
3.0285,
2.7926,
2.3250,
2.7806,
2.2011,
1.7589,
1.1492,
1.7963,
0.6999,
0.8280,
0.7433,
0.6442,
0.8131
]
energy_0 = [
8.0481,
19.4774,
12.3396,
14.6787,
17.4641,
15.7170,
16.3020,
28.5772,
36.7077,
11.4318,
45.8902,
39.6480,
37.2003,
39.9937,
19.3986
]

# 协作比例 0.6 - 假设有所改善
cycle_time_6 = [
28.2,
31.6,
33.7,
34.2,
35.4,
37.2,
38.9,
40.0,
40.8,
41.7,
41.9,
43.5,
51.8,
51.6,
56.7
]
ergonomics_6 = [
4.2179,
4.4279,
3.7149,
3.6989,
2.8786,
2.9019,
2.4171,
1.5867,
2.6609,
2.7711,
1.2218,
1.5774,
0.9176,
0.7928,
0.6986
]
energy_6 = [
17.2938,
9.6373,
6.9961,
8.3421,
14.4723,
13.9710,
16.5446,
32.2621,
9.3150,
7.8084,
30.559,
29.8794,
31.7310,
26.0868,
37.3455
]

# 协作比例 0.8 - 假设进一步改善
cycle_time_8 = [
25.7,
26.7,
28.0,
28.5,
30.6,
31.4,
32.3,
33.1,
33.6,
38.4,
40.8,
42.8,
45.9,
52.5,
54.5
]
ergonomics_8 = [
4.7482,
4.9164,
4.2432,
4.3266,
3.3705,
3.8618,
3.9560,
3.3760,
3.6699,
2.5748,
2.1624,
1.7321,
1.9292,
0.8135,
0.7892
]
energy_8 = [
20.4689,
15.4680,
17.5909,
14.8083,
16.9677,
14.2385,
11.9834,
6.9420,
13.1040,
14.1471,
28.1784,
21.4151,
16.7004,
37.5555,
28.8662
]

# 创建子图
fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

# 箱线图颜色
colors = ['blue', 'green', 'red']
labels = ['0.0', '0.6', '0.8']

# 1. 周期时间箱线图
box_cycle = axs[0].boxplot([cycle_time_0, cycle_time_6, cycle_time_8],
                           patch_artist=True, labels=labels)
for patch, color in zip(box_cycle['boxes'], colors):
    patch.set_facecolor(color)

# axs[0].set_title('周期时间分布', fontsize=14)
axs[0].set_xlabel('Collaboration Task Ratio', fontsize=14)
axs[0].set_ylabel('Cycle Time', fontsize=14)
axs[0].grid(True, linestyle='--', alpha=0.7)

# 2. 工效学风险箱线图
box_ergo = axs[1].boxplot([ergonomics_0, ergonomics_6, ergonomics_8],
                          patch_artist=True, labels=labels)
for patch, color in zip(box_ergo['boxes'], colors):
    patch.set_facecolor(color)

# axs[1].set_title('工效学风险分布', fontsize=14)
axs[1].set_xlabel('Collaboration Task Ratio', fontsize=14)
axs[1].set_ylabel('Ergonomic Risk', fontsize=14)
axs[1].grid(True, linestyle='--', alpha=0.7)

# 3. 机器能耗箱线图
box_energy = axs[2].boxplot([energy_0, energy_6, energy_8],
                            patch_artist=True, labels=labels)
for patch, color in zip(box_energy['boxes'], colors):
    patch.set_facecolor(color)

# axs[2].set_title('机器能耗分布', fontsize=14)
axs[2].set_xlabel('Collaboration Task Ratio', fontsize=14)
axs[2].set_ylabel('Energy Consumption', fontsize=14)
axs[2].grid(True, linestyle='--', alpha=0.7)

# 添加总标题
fig.suptitle('Distribution of Three Objectives Under Different Collaboration Ratios ', fontsize=16)

# 添加图例说明
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                Line2D([0], [0], color=colors[1], lw=4),
                Line2D([0], [0], color=colors[2], lw=4)]
fig.legend(custom_lines, ['Ratio=0.0', ' Ratio=0.6', 'Ratio=0.8'],
           loc='lower center', ncol=3, fontsize=14)

# 调整布局
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# 保存图形
plt.savefig('boxplot_comparison(1).png', dpi=300, bbox_inches='tight')
# plt.savefig('boxplot_comparison.pdf', bbox_inches='tight')

plt.show()