from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from init import decode
from vector import *
# from vector import *

import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_front(Archivepop, curNAP):
    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300
    # 设置字体样式
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 提取目标值
    objectives = np.array([ind.fitness for ind in Archivepop])

    # 按照第一个目标值排序
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_objectives = objectives[sorted_indices]

    # 绘制 Pareto 前沿图
    fig, ax = plt.subplots(figsize=(6, 4))  # 调整图片尺寸为 6x4 英寸
    ax.scatter(sorted_objectives[:, 0], sorted_objectives[:, 1], color='r', s=30, edgecolor='k', label='Pareto Front')
    ax.plot(sorted_objectives[:, 0], sorted_objectives[:, 1], color='b', linestyle='-', linewidth=1)

    # 设置坐标轴标签和标题
    ax.set_xlabel('Objective 1', fontsize=14)
    ax.set_ylabel('Objective 2', fontsize=14)
    ax.set_title('Pareto Front', fontsize=16)

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 显示图例
    ax.legend(fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 保存图像（如果指定了保存路径）
    # if save_path:
    #     plt.savefig(save_path, dpi=300)

    # 显示图像
    plt.show()


st_number=4
def transfer(TA,AM,WTS,CTS, finish_time):
    max_station = max(TA)
    task_ids = [[] for _ in range(max_station + 1)]

    # 遍历 TA 向量，将每个任务的索引添加到对应的工作站列表中
    for index, station in enumerate(TA):
        task_ids[station].append(index + 1)

    print("task_ids:",task_ids)

    operators = []
    for station_tasks in task_ids:
        station_operators = []
        for task_index in station_tasks:
            # 由于 task_index 从 1 开始，所以在 TA 和 AM 中索引要减 1
            am_value = AM[task_index - 1]
            station_num = TA[task_index - 1]

            if am_value == 0:
                # 找到分配给该工作站的工人编号
                worker_num = [i + 1 for i, w in enumerate(WTS) if w == station_num][0]
                operator = [1, f'{worker_num}', '']
            elif am_value == 1:
                # 找到分配给该工作站的协作机器人编号
                robot_num = [i + 1 for i, r in enumerate(CTS) if r == station_num][0]
                operator = [2, '', f'{robot_num}']
            elif am_value == 2:
                # 找到分配给该工作站的工人编号和协作机器人编号
                worker_num = [i + 1 for i, w in enumerate(WTS) if w == station_num][0]
                robot_num = [i + 1 for i, r in enumerate(CTS) if r == station_num][0]
                operator = [3, f'{worker_num}', f'{robot_num}']
            station_operators.append(operator)
        operators.append(station_operators)

    print("operators:", operators)

    durations = []
    for station_tasks in task_ids:
        print("station_tasks:",station_tasks)
        station_durations = []
        for task_index in station_tasks:
            # 任务编号从 1 开始，转换为 0 索引
            task_num = task_index - 1
            am_value = AM[task_num]
            station_num = TA[task_num]

            # 找到分配给该工作站的工人编号和协作机器人编号
            worker_assignments = [i + 1 for i, w in enumerate(WTS) if w == station_num]
            if worker_assignments:
                worker_num = worker_assignments[0]
            else:
                worker_num = None

            # 找到分配给该工作站的协作机器人编号
            robot_assignments = [i + 1 for i, r in enumerate(CTS) if r == station_num]
            if robot_assignments:
                robot_num = robot_assignments[0]
            else:
                robot_num = None

            if am_value == 0:
                # 仅工人操作，从 TWtime 中获取时间
                time = TWtime[task_num][worker_num - 1]
            elif am_value == 1:
                # 仅机器人操作，从 TCtime 中获取时间
                time = TCtime[task_num][robot_num - 1]
            elif am_value == 2:
                # 工人和协作机器人并行操作，从 TWCtime 中获取时间
                # 计算协作时间的索引
                index = (worker_num - 1) * st_number + (robot_num - 1)
                time = TWCtime[task_num][index]
            station_durations.append(time)
        durations.append(station_durations)

    print("durations:", durations)

    num_stations = max(TA) + 1
    completion_times = [[] for _ in range(num_stations)]

    # 遍历每个任务，根据 TA 向量将任务的完成时间添加到对应的工作站列表中
    for i in range(len(TA)):
        station = TA[i]
        time = finish_time[i]
        completion_times[station].append(time)

    print("completion_times:", completion_times)

    return task_ids, operators, durations, completion_times

plt.rcParams['font.family'] = 'Times New Roman'
def draw_improved(task_ids, operators, durations, completion_times):
    # 创建学术风格图表
    fig, ax = plt.subplots(figsize=(10, 6))  # 调整尺寸

    # 学术配色方案（色盲友好）
    colors = {
        'station_bg': '#F5F5F5',  # 工作站背景色
        'human': '#6699CC',  # 蓝色系
        'robot': '#FF9999',  # 橙色系
        'collab': '#99CC99'  # 绿色系
    }

    # 创建分层坐标系统
    y_ticks = []
    y_labels = []
    current_y = 0
    station_spacing = 0.5  # 缩短工作站间距

    # 预计算最大时间
    max_time = max([max(ct) for ct in completion_times if ct]) if any(completion_times) else 0

    for station_idx in range(st_number):
        station_name = f"Station({1 + station_idx // 2},{1 + station_idx % 2})"

        # 添加工作站背景区域
        ax.axhspan(current_y - 0.8, current_y + 0.8, facecolor=colors['station_bg'], alpha=0.3)

        # 工作站标题
        y_ticks.append(current_y)
        y_labels.append(station_name)
        current_y += 1

        # 收集当前工作站的操作员
        operators_in_station = set()
        for op in operators[station_idx]:
            if op[0] == 3:
                operators_in_station.add(('Human', op[1]))
                operators_in_station.add(('Robot', op[2]))
            else:
                type_str = 'Human' if op[0] == 1 else 'Robot'
                operators_in_station.add((type_str, op[1] if op[0] == 1 else op[2]))

        # 绘制操作员轨道
        for op_type, op_id in sorted(operators_in_station):
            y_ticks.append(current_y)
            y_labels.append(f"  {op_type} {op_id}")  # 缩进显示
            current_y += 0.9  # 缩短行间距

        current_y += station_spacing

    # 设置坐标轴
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, max_time * 1.05)
    ax.invert_yaxis()

    # 绘制任务条
    for station_idx in range(st_number):
        station_tasks = task_ids[station_idx]
        for task_idx in range(len(station_tasks)):
            op_type, worker, robot = operators[station_idx][task_idx]
            start = completion_times[station_idx][task_idx] - durations[station_idx][task_idx]
            duration = durations[station_idx][task_idx]
            task_id = station_tasks[task_idx]

            # 确定任务类型
            if op_type == 3:
                # 协作任务处理
                human_label = f"  Human {worker}"
                robot_label = f"  Robot {robot}"
                y_human = y_ticks[y_labels.index(human_label)]
                y_robot = y_ticks[y_labels.index(robot_label)]

                # 相同颜色不同样式
                ax.barh(y_human, duration, left=start, height=0.7,
                        color=colors['collab'], edgecolor='k', linewidth=0.5)
                ax.barh(y_robot, duration, left=start, height=0.7,
                        color=colors['collab'], edgecolor='k', linewidth=0.5,
                        hatch='', alpha=0.9)

                # 任务ID居中显示
                ax.text(start + duration / 2, y_human, f'T{task_id}',
                        ha='center', va='center', color='black', fontsize=10)
                ax.text(start + duration / 2, y_robot, f'T{task_id}',
                        ha='center', va='center', color='black', fontsize=10)
            else:
                # 单独任务处理
                op_label = f"  {'Human' if op_type == 1 else 'Robot'} {worker if op_type == 1 else robot}"
                y_pos = y_ticks[y_labels.index(op_label)]
                color = colors['human'] if op_type == 1 else colors['robot']

                ax.barh(y_pos, duration, left=start, height=0.7,
                        color=color, edgecolor='k', linewidth=0.5)
                ax.text(start + duration / 2, y_pos, f'T{task_id}',
                        ha='center', va='center', color='white', fontsize=10)

    # 优化样式
    ax.set_xlabel('Time (s)', fontsize=12)
    # ax.set_title('Min Cycle Time', fontsize=14, pad=15)

    # 设置网格线
    ax.grid(True, axis='x', linestyle=':', color='grey', alpha=0.6)
    ax.grid(False, axis='y')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['human'], label='Human Only'),
        Patch(facecolor=colors['robot'], label='Robot Only'),
        Patch(facecolor=colors['collab'], label='Collaboration'),
        # Patch(facecolor=colors['collab'], hatch='////', label='Collaboration (Robot)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)

    # 调整子图布局，给底部留出更多空间
    plt.subplots_adjust(bottom=0.15, left=0.15)

    fig.text(0.5, 0.00, "c) Min Energy Consumption", ha='center', va='bottom', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    ##P24 min CT
    # TA=[0, 0, 1, 1, 2, 1, 1, 3, 1, 3, 2, 2, 1, 3, 3, 2, 2, 1, 3, 3, 2, 3, 3, 2]
    # AM=[2, 2, 2, 0, 2, 1, 0, 2, 0, 0, 2, 2, 0, 2, 0, 2, 2, 0, 2, 2, 1, 0, 1, 0]
    # WTS=[3, 2, 0, 1]
    # CTS=[1, 3, 0, 2]
    # finish_time=[1.8, 3.7, 1.2, 5.1, 2.4, 6.3999999999999995, 2.9, 4.999999999999999, 7.3999999999999995, 3.8,
    #                    3.9, 5.9, 10.899999999999999, 6.999999999999999, 1.7, 5.2, 9.4, 11.799999999999999,
    #                    8.399999999999999, 4.3999999999999995, 10.6, 10.499999999999998, 13.599999999999998, 7.0]

    ##P24 min ER
    # TA=[0, 0, 1, 1, 0, 1, 3, 0, 1, 3, 0, 0, 3, 3, 1, 2, 3, 3, 3, 1, 2, 2, 3, 3]
    # AM=[0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    # WTS=[0, -1, 3, -1]
    # CTS=[3, 1, 2, -1]
    # finish_time= [3.5, 6.5, 4.1, 6.699999999999999, 9.4, 7.8999999999999995, 1.2, 12.899999999999999,
    #                    9.399999999999999, 4.3, 11.7, 14.099999999999998, 11.1, 9.0, 11.999999999999998, 2.8, 18.9, 14.0,
    #                    15.2, 1.5, 20.299999999999997, 22.9, 22.099999999999998, 12.7]
    # 目标函数值为: [22.9, 0.9192965431714302, 13.448500000000001]

    ##P24 min EC
    # TA = [0, 0, 1, 1, 0, 3, 1, 1, 2, 1, 0, 2, 3, 3, 1, 2, 2, 3, 2, 3, 2, 3, 3, 3]
    # AM = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # WTS = [1, 0, 2, 3]
    # CTS = [-1, -1, -1, -1]
    # finish_time = [3.2, 6.0, 1.7, 3.4, 9.0, 0.9, 5.6, 10.599999999999998, 1.8, 7.699999999999999, 11.0, 4.9, 4.4,
    #                    7.9, 9.399999999999999, 4.0, 8.0, 9.700000000000001, 10.0, 5.2, 10.9, 13.9, 12.0, 8.8]
    # decode(TA, AM, WTS, CTS)

    #P65
    # TA=[1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 3, 2, 2, 3, 3, 3, 3, 3, 1, 1, 0, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3,
    #            3, 3, 3, 2, 3, 1, 1, 1, 1, 0, 1, 0, 2, 1, 3, 3, 3, 2, 3, 3, 2, 3, 2, 3, 3, 3, 3, 0, 0, 3]
    # AM=[2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2, 1, 2,
    #           2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2]
    # WTS=[2, 3, 0, 1]
    # CTS=[0, 3, 1, 2]
    # finish_time=[2.2, 4.2, 11.500000000000002, 13.9, 16.3, 8.200000000000001, 14.299999999999999,
    #                    15.499999999999998, 16.400000000000002, 20.7, 6.1000000000000005, 6.800000000000001, 2.1,
    #                    22.599999999999998, 1.3, 2.8, 5.1, 5.3, 8.1, 6.8, 11.799999999999999, 9.299999999999999, 9.4,
    #                    10.0, 17.400000000000002, 11.9, 13.1, 28.299999999999997, 4.2, 8.9, 13.1, 17.4, 34.9, 38.9,
    #                    40.99999999999999, 14.6, 28.4, 32.5, 33.7, 39.699999999999996, 14.899999999999999,
    #                    18.099999999999998, 29.499999999999996, 4.9, 18.400000000000002, 25.4, 26.7, 1.4,
    #                    27.599999999999998, 42.99999999999999, 15.9, 18.9, 15.7, 20.099999999999998, 21.4, 16.3,
    #                    31.299999999999997, 19.0, 22.9, 23.5, 25.9, 30.4, 10.200000000000001, 12.700000000000001,
    #                    45.39999999999999]

    #P36 minCT
    # TA=[0, 0, 0, 0, 2, 1, 2, 3, 1, 1, 2, 3, 2, 3, 3, 5, 2, 1, 3, 5, 4, 0, 1, 5, 4, 0, 3, 4, 4, 4, 2, 2, 5, 5, 5,
    #            4]
    # AM=[0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 2, 2, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 1, 2, 2, 0,
    #           0]
    # WTS=[5, 2, 4, 0, 3, 1]
    # CTS=[3, 5, 2]
    # finish_time=[5.1, 8.0, 12.6, 15.8, 3.2, 3.2, 8.4, 11.899999999999999, 18.5, 16.9, 17.499999999999996,
    #                    22.499999999999996, 16.299999999999997, 2.3, 18.599999999999998, 3.7, 25.4, 8.5, 9.2, 6.7, 6.2,
    #                    14.0, 22.7, 11.100000000000001, 11.0, 19.3, 15.099999999999998, 15.2, 28.2, 20.5, 11.8,
    #                    22.799999999999997, 19.7, 5.5, 27.1, 27.2]

    #P36 minER
    # TA=[0, 0, 0, 0, 1, 0, 2, 3, 1, 1, 2, 3, 2, 3, 3, 4, 5, 1, 3, 5, 2, 0, 1, 5, 4, 1, 3, 4, 4, 5, 2, 2, 4, 5, 4,
    #            5]
    # AM=[1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1,
    #           0]
    # WTS=[1, -1, 3, -1, 5, -1]
    # CTS=[0, 2, 4]
    # _,_,_,finish_time=decode(TA, AM, WTS, CTS)

    #P36 minEC
    # TA=[1, 0, 0, 0, 1, 0, 0, 3, 1, 1, 3, 0, 2, 3, 3, 4, 4, 1, 3, 5, 3, 0, 1, 5, 4, 0, 2, 4, 4, 4, 2, 2, 5, 4, 5,
    #            5]
    # AM=[0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 2, 0,
    #           0]
    # WTS=[5, 2, 0, 4, 3, 1]
    # CTS=[4, -1, 0]
    # finish_time= [7.1, 5.7, 8.4, 14.200000000000001, 30.299999999999997, 11.100000000000001, 22.700000000000003,
    #                    3.8, 13.999999999999998, 22.4, 18.699999999999996, 26.500000000000004, 22.4, 6.1, 10.2, 3.2, 5.4,
    #                    12.399999999999999, 16.299999999999997, 5.800000000000001, 23.699999999999996, 11.8, 34.5, 12.8,
    #                    11.8, 17.900000000000002, 6.7, 22.5, 24.8, 20.5, 12.2, 27.9, 28.0, 13.600000000000001, 35.4,
    #                    22.5]

    #P24 minCT
    # TA= [0, 0, 1, 1, 0, 1, 1, 2, 1, 0, 0, 2, 3, 3, 1, 0, 2, 2, 2, 3, 2, 3, 3, 3]
    # AM=[2, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2]
    # WTS=[1, 2, 3, 0]
    # CTS=[0, 2, 1, 3]
    # finish_time= [2.4, 4.6, 1.4, 2.8, 6.8999999999999995, 6.899999999999999, 5.499999999999999, 0.8,
    #                    7.499999999999998, 10.299999999999999, 8.399999999999999, 1.4, 5.1000000000000005,
    #                    2.9000000000000004, 4.199999999999999, 11.899999999999999, 3.9, 6.300000000000001,
    #                    7.6000000000000005, 0.7, 6.6000000000000005, 9.3, 6.5, 7.3]

    #P24 min EC
    TA=[0, 0, 1, 1, 0, 1, 1, 0, 1, 3, 0, 2, 2, 3, 1, 0, 2, 3, 2, 3, 2, 2, 3, 3]
    AM=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    WTS=[1, 0, 3, 2]
    CTS=[-1, -1, -1, -1]
    finish_time= [3.2, 6.0, 1.7, 3.4, 9.0, 6.8, 5.6, 12.1, 9.6, 2.0, 11.0, 0.9, 4.4, 5.6, 8.5, 13.7, 7.4, 7.4, 9.1,
                       2.9, 10.1, 12.0, 9.7, 6.5]

    #P24 min ER
    # TA=[0, 0, 1, 1, 0, 0, 1, 3, 1, 0, 0, 2, 3, 3, 1, 2, 2, 3, 2, 1, 2, 2, 3, 3]
    # AM=[2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    # WTS=[1, 0, 3, 2]
    # CTS=[3, 1, 0, 2]
    # finish_time= [2.0, 6.7, 2.6, 5.2, 10.6, 12.0, 6.6, 1.4, 10.7, 17.7, 15.1, 4.4, 3.5, 8.2, 9.2, 2.9, 8.2,
    #                    14.299999999999999, 9.399999999999999, 12.2, 10.599999999999998, 17.0, 11.399999999999999,
    #                    12.999999999999998]

    #P24 min CT
    # TA=[0, 0, 1, 1, 0, 1, 1, 0, 3, 3, 0, 2, 2, 3, 1, 0, 2, 2, 2, 1, 2, 3, 3, 3]
    # AM=[2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2]
    # WTS=[2, 0, 3, 1]
    # CTS=[3, 2, 1, 0]
    # _,_,_,finish_time = decode(TA, AM, WTS, CTS)

    task_ids, operators, durations, completion_times=transfer(TA, AM, WTS, CTS, finish_time)
    draw_improved(task_ids, operators, durations, completion_times)

