from pandas.core.internals.array_manager import NullArrayProxy

from algorithmfunc import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from pymoo.vendor import hv as HH

#NSGA-II参数列表
#迭代次数
T=100
#种群规模
PN=60
#代码修改补充内容
# 交叉概率=0.8
crossover_prob=0.8
# 变异概率=0.2
mutation_prob=0.2

maxNAP=20


# 添加一个Achivepop  ✅
# 对目标函数进行归一化处理 ✅

def tournament_selection(population, fronts, distances):
    """
    二元锦标赛选择
    :param population: 种群
    :param fronts: 非支配等级
    :param distances: 拥挤度距离
    :return: 选择的个体索引
    """
    selected_indices = []
    pop_size = len(population)
    while len(selected_indices) < pop_size:
        i1, i2 = random.sample(range(pop_size), 2)
        if (fronts[i1] < fronts[i2] or (fronts[i1] == fronts[i2] and distances[i1] > distances[i2])) and i1 not in selected_indices:
            selected_indices.append(i1)
        elif i2 not in selected_indices:
            selected_indices.append(i2)
    return selected_indices

def maincode1():
    #初始化
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            pop=[indiv() for _ in range (PN)]
            popinitial(pop,PN)

    curNAP = 0
    Archivepop = [indiv() for _ in range(maxNAP)]

    minCT = [-1 for _ in range(T)]
    minER = [-1 for _ in range(T)]
    minEC = [-1 for _ in range(T)]


    print("------初始化种群--------")
    for i in range(PN):
        print(f'第{i}个个体的编码为：')
        print(f'TA向量为:{pop[i].TA}')
        print(f'AM向量为：{pop[i].AM}')
        print(f'WTS向量为：{pop[i].WTS}')
        print(f'CTS向量为：{pop[i].CTS}')

    start_time = time.time()
    # 目标值
    fit = [[-1 for _ in range(objnum)] for _ in range(PN)]

    hyper = [0 for _ in range(T)]
    space = [0 for _ in range(T)]


    for t in range(T):
        # 评估适应度
        fitness = []

        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(PN):
                    f1,f2,f3, pop[i].FT = decode(pop[i].TA, pop[i].AM, pop[i].WTS, pop[i].CTS)
                    pop[i].fitness = [f1, f2, f3]
                    fitness.append([f1, f2, f3])
                fitness = np.array(fitness)
                #非支配排序
                klevel, levellength, sortindex = fastsort(fitness, PN)
                fronts = [[] for _ in range(klevel)]
                for i in range(klevel):
                    fronts[i] = sortindex[i]
                    print(f'-----------------\n{fronts[i]}')
                all_distances = []
                for i in range(klevel):
                    distances = crowd(fitness[fronts[i]], len(fronts[i]), objnum)
                    all_distances.extend(distances)

                #选择操作
                selected_indices = tournament_selection(pop, [rank for rank in range(klevel) for _ in range(len(fronts[rank]))], all_distances)
                selected_pop = [pop[i] for i in selected_indices]

                #交叉操作
                offspring = []

                for i in range(0, len(selected_pop), 2):
                    parent1 = selected_pop[i]
                    parent2 = selected_pop[i + 1]
                    cross_choice = random.randint(0, 3)
                    #为什么要传入t?????
                    if np.random.rand()<crossover_prob:
                        if cross_choice == 0:
                            child1, child2 = TAcross(parent1, parent2, t, T)
                        # print(f'-----child1:{child1.TA}')
                        # print(f'-----child2:{child2.TA}')
                        elif cross_choice == 1:
                            child1, child2 = AMcross(parent1, parent2, t, T)
                        elif cross_choice == 2:
                            child1, child2 = workercross(parent1, parent2, t, T)
                        else:
                            child1, child2 = cobotcross(parent1, parent2, t, T)
                        offspring.extend([child1, child2])
                    else:
                        offspring.extend([parent1, parent2])

                #变异操作
                test = []
                mutated_offspring=[]
                combined_pop = [indiv() for _ in range(PN+len(offspring)) ]
                for individual in offspring:
                    mutated_choice =random.randint(0, 6)
                    if np.random.rand()<mutation_prob:
                        if mutated_choice == 0:
                            mutated_individual = TAop(individual)
                        elif mutated_choice == 1:
                            mutated_individual = AMop(individual)
                        elif mutated_choice == 2:
                            mutated_individual = WTSop(individual)
                        elif mutated_choice == 3:
                            mutated_individual = CTSop(individual)
                        elif mutated_choice == 4:
                            mutated_individual = critical_worker(individual)
                        elif mutated_choice == 5:
                            mutated_individual = critical_cobot(individual)
                        else:
                            mutated_individual = critical_cmax(individual)



                        mutated_offspring.append(mutated_individual)
                mutated_offspring=np.array(mutated_offspring)
                combined_pop = pop.copy()
                # 合并父代和子代种群
                combined_pop.extend(mutated_offspring)


        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                combined_fitness = []
                for ind in combined_pop:
                    ind.fitness[0], ind.fitness[1], ind.fitness[2], ind.FT = decode(ind.TA, ind.AM, ind.WTS, ind.CTS)
                    combined_fitness.append([ind.fitness[0], ind.fitness[1], ind.fitness[2]])
                combined_fitness = np.array(combined_fitness)
                klevel, levellength, sortindex = fastsort(combined_fitness, len(combined_pop))
                next_population = []
                next_pop_size = 0
                i = 0
                while next_pop_size + len(sortindex[i]) <= PN:
                    next_population.extend([combined_pop[index] for index in sortindex[i]])
                    next_pop_size += len(sortindex[i])
                    i += 1
                if next_pop_size < PN:
                    remaining = PN - next_pop_size
                    last_front = sortindex[i]
                    last_front_distances = crowd(combined_fitness[last_front], len(last_front), 3)
                    sorted_last_front = [x for _, x in sorted(zip(last_front_distances, last_front), key=lambda pair: pair[0], reverse=True)]
                    next_population.extend([combined_pop[index] for index in sorted_last_front[:remaining]])
                pop = next_population


        print(f"——————————————————第{t}次迭代结果————————————————-")
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(PN):
                    fit[i][0], fit[i][1], fit[i][2], FT = decode(pop[i].TA, pop[i].AM,
                                                                  pop[i].WTS, pop[i].CTS)
                    klevel, levellength, sortindex = fastsort(fit, PN)

        normalizer= Normalizer(3)
        fit_array = np.array(fit)
        normalizer.update_bounds(fit_array)

        domfit=[[0 for _ in range(3)] for _ in range(levellength[0])]
        for i in range(levellength[0]):
            domfit[i]=fit[sortindex[0][i]]

        # 获取所有个体的第一个目标值，并过滤掉0
        ct_values = [individual[0] for individual in domfit if individual[0] != 0]
        er_values = [individual[1] for individual in domfit if individual[1] != 0]
        ec_values = [individual[2] for individual in domfit if individual[2] != 0]

        # 然后计算最小值，注意可能列表为空的情况
        minCT[t] = min(ct_values) if ct_values else 0
        minER[t] = min(er_values) if er_values else 0
        minEC[t] = min(ec_values) if ec_values else 0

        if t == T - 1:
            print('NSGA-II-domfit:',domfit)
            for i in range(levellength[0]):
                print('问题编码依次为：')
                print(f'TA向量为:{pop[sortindex[0][i]].TA}')
                print(f'AM向量为：{pop[sortindex[0][i]].AM}')
                print(f'WTS向量为：{pop[sortindex[0][i]].WTS}')
                print(f'CTS向量为：{pop[sortindex[0][i]].CTS}')
                print(f'任务的完成时间为:{pop[sortindex[0][i]].FT}')
                print(f'目标函数值为:{pop[sortindex[0][i]].fitness}')

        data = np.array(domfit)
        #reference_point = np.min(data, axis=0) - 1  # 参考点（设置比最小值更小）

        # normalize_objectives = normalizer.normalize(data)
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


    # for i in range(PN):
    #     print('问题编码依次为：')
    #     print(f'TA向量为:{pop[i].TA}')
    #     print(f'AM向量为：{pop[i].AM}')
    #     print(f'WTS向量为：{pop[i].WTS}')
    #     print(f'CTS向量为：{pop[i].CTS}')
    #     print(f'任务的完成时间为:{pop[i].FT}')
    #     print(f'目标函数值为:{pop[i].fitness}')
    #     for j in range(3):
    #         if pop[i].fitness[j] < 0:
    #             print("报个错")

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

    # npop=[]
    # for i in range(levellength[0]):
    #     npop.append(pop[sortindex[0][i]])
    # npop = select_top_crowded_solutions(npop, 3)
    # print("基于拥挤度距离获取的10个Pareto解：")
    # for i in range(len(npop)):
    #     print(npop[i].fitness)

    return pop, hyper, space

if __name__ == '__main__':
    # 这里是有改进算子
    pop = [indiv() for _ in range(PN)]
    pop, hyper, space= maincode1()

    data = np.array(hyper)

    # 分别提取x, y, z坐标
    x = [i for i in range(1, T + 1)]
    y = data

    # 创建一个新的figure
    fig, ax = plt.subplots()

    # 添加一个三维坐标轴

    # 绘制散点图
    ax.plot(x, y)

    # 设置坐标轴标签
    ax.set_xlabel('Iteration')
    ax.set_ylabel('HV')

    # 显示图形
    plt.show()

    # 解的顺序：周期时间-能耗-疲劳

    data1 = np.array(space)

    # 分别提取x, y, z坐标
    x1 = [i for i in range(1, T + 1)]
    y1 = data1

    # 创建一个新的figure
    fig1, ax1 = plt.subplots()

    # 添加一个三维坐标轴

    # 绘制散点图
    ax1.plot(x1, y1)
    # 设置坐标轴标签
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Spacing')

    # 显示图形
    plt.show()

    # with open(os.devnull, 'w') as f:
    #     with contextlib.redirect_stdout(f):
    #         for i in range(PN):
    #             copy(minitpop[i], pop[i])
    #         for i in range(PN):
    #             fit[i][0], fit[i][1], fit[i][2], pop[i].FT = decode(pop[i].TA, pop[i].AM, pop[i].WTS, pop[i].CTS)
    #         # 非支配排序:klevel是总层数，levellength是每层的项数，index是每层的项集合
    #         klevel, levellength, sortindex = fastsort(fit, PN)
            # Leaderindex, flindex, frindex, freeindex, Followindex = partpop(klevel, levellength, sortindex)
            # print(Leaderindex, flindex, frindex, freeindex, Followindex)

            # copy(pop[sortindex[0][0]], Archivepop[0])
            # curNAP = curNAP + 1
            # 初始时，外部档案从当前非支配等级为0的个体中获得;如果个体数超过NAP，则采用拥挤距离计算放入
            # if maxNAP > levellength[0]:
            #     for i in range(levellength[0]):
            #         Archivepop, curNAP = archivechange(Archivepop, curNAP, pop[sortindex[0][i]], fit[sortindex[0][i]])
            # else:
            #     tempfit = [[0 for _ in range(3)] for _ in range(levellength[0])]
            #     for i in range(levellength[0]):
            #         tempfit[i] = fit[sortindex[0][i]]
            #     crowdist = crowd(tempfit, levellength[0], 3)
            #     # 对序号按crowdist中的值排序，选前maxNAP个解放入
            #     sorted_indices = sort_indices_by_values(crowdist)
            #     for i in range(maxNAP):
            #         Archivepop, curNAP = archivechange(Archivepop, curNAP, pop[sortindex[0][i]], fit[sortindex[0][i]])
