# 侦察蜂阶段：
# 《基于改进人工蜂群算法的机器人任务最优指派》
# 《部分拆装线平衡问题的多目标人工蜂群算法》
from algorithmfunc import *
import os
import numpy as np
import time
from pymoo.vendor import hv as HH

#MOABC算法参数
PN=30
T = 100
limit=10
#外部档案的数量
maxNAP=20

#新增参数
employed_ratio = 0.5  #雇佣蜂比例
employed_bees = int(employed_ratio * PN)
onlooker_bees = PN - employed_bees

#不是很理解怎么使用
scout_prob = 0.2  # 侦查蜂触发概率(在试验次数未超过limit时)
neighborhood_factor = 0.3  # 邻域搜索扰动因子
grid_size = 10  # 网格划分数量(用于存档多样性维护)

def show_result(minitpop, num):
    for i in range(num):
        print('问题编码依次为：')
        print(f'TA向量为:{minitpop[i].TA}')
        print(f'AM向量为：{minitpop[i].AM}')
        print(f'WTS向量为：{minitpop[i].WTS}')
        print(f'CTS向量为：{minitpop[i].CTS}')
        print(f'任务的完成时间为:{minitpop[i].FT}')
        print(f'目标函数值为:{minitpop[i].fitness}')

def binary_tournament_selection(ind, pop):
    available_pop = [p for p in pop if p is not ind]
    index1, index2 = random.sample(range(len(available_pop)), 2)
    if dominates(available_pop[index1].fitness, available_pop[index2].fitness):
        return available_pop[index1]
    elif dominates(available_pop[index2].fitness, available_pop[index1].fitness):
        return available_pop[index2]
    else:
        temp_fit = [available_pop[index1].fitness, available_pop[index2].fitness]
        crowding_dist = crowd(temp_fit, 2, 3)  # 假设3个目标
        if crowding_dist[0] > crowding_dist[1]:
            return available_pop[index1]
        else:
            return available_pop[index2]

def neighborhood_operation(ind):
    op_choice = random.randint(0, 6)
    print('进行领域搜索操作的个体编码为：')
    print(f'-----ind.TA:{ind.TA}')
    if op_choice == 0:
        TAop(ind)
    elif op_choice == 1:
        AMop(ind)
    elif op_choice == 2:
        WTSop(ind)
    elif op_choice == 3:
        CTSop(ind)
    elif op_choice == 4:
        critical_cmax(ind)
    elif op_choice == 5:
        critical_worker(ind)
    else:
        critical_cobot(ind)
    return ind

def crossover_operation(parent1, parent2,t):
    cross_choice = random.randint(0, 3)
    if cross_choice == 0:
        child1, child2 = TAcross(parent1, parent2,t, T)
    elif cross_choice == 1:
        child1, child2 = AMcross(parent1, parent2, t, T)
    elif cross_choice == 2:
        child1, child2 = workercross(parent1, parent2, t,T)
    else:
        child1, child2 = cobotcross(parent1, parent2, t,T)
    return child1, child2

def get_archive_format(minitpop,curNAP,maxNAP):
    fit = [[-1 for _ in range(3)] for _ in range(PN)]
    Archivepop = [indiv() for _ in range(maxNAP)]
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            for i in range(PN):
                fit[i][0], fit[i][1], fit[i][2], minitpop[i].FT = decode(minitpop[i].TA, minitpop[i].AM, minitpop[i].WTS, minitpop[i].CTS)
            # 非支配排序:klevel是总层数，levellength是每层的项数，index是每层的项集合
            klevel, levellength, sortindex = fastsort(fit, PN)

            # copy(minitpop[sortindex[0][0]], Archivepop[0])
            # curNAP = curNAP + 1
            # 初始时，外部档案从当前非支配等级为0的个体中获得;如果个体数超过NAP，则采用拥挤距离计算放入
            if maxNAP > levellength[0]:
                for i in range(levellength[0]):
                    Archivepop, curNAP = archivechange(Archivepop, curNAP, minitpop[sortindex[0][i]], fit[sortindex[0][i]],maxNAP)
            else:
                tempfit = [[0 for _ in range(3)] for _ in range(levellength[0])]
                for i in range(levellength[0]):
                    tempfit[i] = fit[sortindex[0][i]]
                crowdist = crowd(tempfit, levellength[0], 3)
                # 对序号按crowdist中的值排序，选前maxNAP个解放入
                sorted_indices = sort_indices_by_values(crowdist)
                for i in range(maxNAP):
                    Archivepop, curNAP = archivechange(Archivepop, curNAP, minitpop[sortindex[0][i]], fit[sortindex[0][i]],maxNAP)
    return Archivepop, curNAP

def maincode3():
    #种群初始化
    pop=[indiv() for _ in range (PN)]
    hyper = [0 for _ in range(T)]
    space = [0 for _ in range(T)]
    start_time = time.time()
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            popinitial(pop,PN)
    minCT = [-1 for _ in range(T)]
    minER = [-1 for _ in range(T)]
    minEC = [-1 for _ in range(T)]

    #外部存储档案
    curNAP = 0
    Archivepop = [indiv() for _ in range(maxNAP)]

    #记录解的迭代次数
    trial_count = [0] * PN

    # 目标值
    Archivepop, curNAP = get_archive_format(pop,curNAP,maxNAP)

#问题外部存档函数不一致
#ABC算法中外部存档的使用机制
#自身外部存档函数的用法：若新解不与外部存档重复，且支配外部存档中的解，则加入外部存档的最后，否则不加入

    for t in range(T):
        # print(f"------------第{t}次迭代初始种群-------------")
        # show_result(pop, PN)
        improved_solutions =[]

        #雇佣蜂阶段
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(employed_bees):
                    neighbor = neighborhood_operation(pop[i])
                    neighbor.fitness[0], neighbor.fitness[1],neighbor.fitness[2],neighbor.FT  = decode(neighbor.TA, neighbor.AM, neighbor.WTS, neighbor.CTS)
                    if dominates(neighbor.fitness, pop[i].fitness):
                        pop[i] = neighbor
                        trial_count[i] = 0
                        improved_solutions.append(neighbor)
                    else:
                        trial_count[i] = trial_count[i] + 1

                #更新存档
                for j in range(len(improved_solutions)):
                    Archivepop, curNAP = archivechange(Archivepop, curNAP, improved_solutions[j], improved_solutions[j].fitness, maxNAP)

        # print("------------雇佣蜂处理后的种群-------------")
        # show_result(pop, PN)
        # print('trail_count:', trial_count)

        #观察蜂阶段--------
        for i in range(employed_bees, PN):
            child = [indiv() for _ in range(2)]
            parent1 = pop[i]

            with open(os.devnull, 'w') as f:
                with contextlib.redirect_stdout(f):
                    parent2 = binary_tournament_selection(pop[i], pop)
                    child[0], child[1] = crossover_operation(parent1, parent2, t)

            #child[0], child[1] = crossover_operation(parent1, parent2,t)
                    for n in range(2):
                        child[n].fitness[0], child[n].fitness[1], child[n].fitness[2], child[n].FT = decode(child[n].TA, child[n].AM, child[n].WTS, child[n].CTS)

                    if dominates(child[0].fitness, child[1].fitness):
                        child = child[0]
                    elif dominates(child[1].fitness, child[0].fitness):
                        child = child[1]
                    else:
                        temp_fit = [child[0].fitness, child[1].fitness]
                        crowding_dist = crowd(temp_fit, 2, 3)
                        if crowding_dist[0] > crowding_dist[1]:
                            child = child[0]
                        else:
                            child = child[1]

                    if dominates(child.fitness, pop[i].fitness):
                        pop[i] = child
                        trial_count[i] = 0
                        Archivepop, curNAP = archivechange(Archivepop, curNAP,child, child.fitness,maxNAP)
                    else:
                        trial_count[i] = trial_count[i] + 1

        # print("------------观察蜂处理后的种群-------------")
        # show_result(pop, PN)
        # print('trail_count:', trial_count)

        #侦查蜂阶段
        for i  in range(PN):
            if trial_count[i] >= limit or random.random()<scout_prob:
                mpop = indiv()
                with open(os.devnull, 'w') as f:
                    with contextlib.redirect_stdout(f):
                        initindiv1(mpop,1)
                        mpop.fitness[0], mpop.fitness[1], mpop.fitness[2], mpop.FT = decode(mpop.TA, mpop.AM, mpop.WTS, mpop.CTS)
                        pop[i] = mpop
                        trial_count[i] = 0
                        Archivepop, curNAP = archivechange(Archivepop, curNAP,mpop,mpop.fitness,maxNAP)

        # print("------------侦查蜂处理后的种群-------------")
        # show_result(pop, PN)
        # print('trail_count:', trial_count)

        # Archivepop, curNAP = get_archive_format(pop,curNAP,maxNAP)

        print(f"——————————————————第{t}次迭代结果————————————————-")

        domfit = [[0 for _ in range(3)] for _ in range(curNAP)]
        #arcfit = [[-1 for _ in range(3)] for _ in range(maxNAP)]
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(curNAP):
                    Archivepop[i].fitness[0], Archivepop[i].fitness[1], Archivepop[i].fitness[2], Archivepop[i].FT= decode(Archivepop[i].TA, Archivepop[i].AM,
                                                                   Archivepop[i].WTS, Archivepop[i].CTS)
                    domfit[i] = Archivepop[i].fitness

        # 获取所有个体的第一个目标值，并过滤掉0
        ct_values = [individual[0] for individual in domfit if individual[0] != 0]
        er_values = [individual[1] for individual in domfit if individual[1] != 0]
        ec_values = [individual[2] for individual in domfit if individual[2] != 0]

        # 然后计算最小值，注意可能列表为空的情况
        minCT[t] = min(ct_values) if ct_values else 0
        minER[t] = min(er_values) if er_values else 0
        minEC[t] = min(ec_values) if ec_values else 0

        if t == T - 1:
            print('MOABC-domfit:',domfit)
            for i in range(curNAP):
                print('问题编码依次为：')
                print(f'TA向量为:{Archivepop[i].TA}')
                print(f'AM向量为：{Archivepop[i].AM}')
                print(f'WTS向量为：{Archivepop[i].WTS}')
                print(f'CTS向量为：{Archivepop[i].CTS}')
                print(f'任务的完成时间为:{Archivepop[i].FT}')
                print(f'目标函数值为:{Archivepop[i].fitness}')


        data = np.array(domfit)
        #reference_point = np.min(data, axis=0) - 1  # 参考点（设置比最小值更小）

        # max_vals = np.max(data, axis=0)*1.1
        # reference_point = [35, 5, 20]
        reference_point = [70,5,50]

        ind = HH.HyperVolume(reference_point)
        hv = ind.compute(data)

        spacing = calculate_spacing(data)

        print("-----------------解的评估指标----------------—-")
        print(f"Hypervolume (HV): {hv}")
        hyper[t] = hv
        space[t] = spacing
        print(f"Spacing: {spacing}")

    s_min = min(space)
    s_max = max(space)
    s_mean = sum(space) / len(space)
    print(f"Spacing min: {s_min}")
    print(f"Spacing max: {s_max}")
    print(f"Spacing mean: {s_mean}")

    end_time = time.time()
    print(f"算法运行时间：{end_time - start_time} 秒")

    print(f'CT迭代变化：{minCT}')
    print(f'ER迭代变化：{minER}')
    print(f'EC迭代变化：{minEC}')

    # npop = select_top_crowded_solutions(Archivepop, 3)
    # print("基于拥挤度距离获取的10个Pareto解：")
    # for i in range(len(npop)):
    #     print(npop[i].fitness)

    # return domfit
    return Archivepop,hyper,space
    # return hypter[T-1], space[T-1], s_mean

if __name__ == '__main__':
    # 这里是有改进算子
    curNAP = 0
    Archivepop = [indiv() for _ in range(maxNAP)]

    Archivepop,hyper,space = maincode3()

    # for i in range(9):
    #     hv, space, s_mean = maincode3()
    #     print(f"第{i + 1}次运行后的HV为：{hv}")
    #     print(f"第{i + 1}次运行后的Spacing为：{space}")
    #     print(f"第{i + 1}次运行后的Spacing的均值为：{s_mean}")

    # data = np.array(hyper)
    #
    # # 分别提取x, y, z坐标
    # x = [i for i in range(1, T + 1)]
    # y = data
    #
    # # 创建一个新的figure
    # fig, ax = plt.subplots()
    #
    # # 添加一个三维坐标轴
    #
    # # 绘制散点图
    # ax.plot(x, y)
    #
    # # 设置坐标轴标签
    # ax.set_xlabel('Iteration')
    # ax.set_ylabel('HV')
    #
    # # 显示图形
    # plt.show()
    #
    # # 解的顺序：周期时间-疲劳-能耗
    #
    # data1 = np.array(space)

    # # 分别提取x, y, z坐标
    # x1 = [i for i in range(1, T + 1)]
    # y1 = data1
    #
    # # 创建一个新的figure
    # fig1, ax1 = plt.subplots()
    #
    # # 添加一个三维坐标轴
    #
    # # 绘制散点图
    # ax1.plot(x1, y1)
    # # 设置坐标轴标签
    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('Spacing')
    #
    # # 显示图形
    # plt.show()










