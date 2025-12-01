from algorithmfunc import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from pymoo.vendor import hv as HH

#MOMBO算法参数列表
#种群规模
PN=30
#迭代次数
T=100
#巡回次数
FN=10
#共享领域解的数量
NStoshared=3
#领域解的数量
NS=7
maxNAP=20

#领域搜索方式限定为4种

#左右领飞鸟划分
def partpop1(klevel, levellength, sortindex):
    # 从第一层随机选择一个个体作为领飞鸟
    Leaderindex = random.choice(sortindex[0][0:levellength[0]])
    #print(f"领飞鸟为{Leaderindex}")
    #print(f'一共有{klevel}层')
    # 顺序查找PN/2个任务
    Followindex = [-1 for _ in range(PN-1)]
    fn = 0
    k = 0
    while fn < PN-1:
        for i in range(int(levellength[k])):
            if (sortindex[k][i] != Leaderindex) & (sortindex[k][i] != -1):
                Followindex[fn] = sortindex[k][i]
                fn = fn + 1
                if fn >= PN-1:
                    break
        #print(f'fn={fn}')
        if fn >= PN-1:
            break
        else:
            k = k + 1
        #print(f'k={k}')
    #print(f'len(Followindex)={len(Followindex)}')
    # 左侧鸟
    flindex = [Followindex[i] for i in range(len(Followindex)) if i % 2 != 0]
    # 右侧鸟
    frindex = [Followindex[i] for i in range(len(Followindex)) if i % 2 == 0]
    #print(flindex, frindex)

    return Leaderindex, flindex, frindex, Followindex

#左右两侧共享解的分配
def select_shared_solution(templead, tempfit, max_index, is_dominated):
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            if is_dominated:
                # 领飞鸟的邻域解支配领飞鸟，排除支配的解后进行非支配排序
                remaining_tempfit = [fit for i, fit in enumerate(tempfit) if i != max_index]
                remaining_templead = [ind for i, ind in enumerate(templead) if i != max_index]
                _, _, tsortindex = fastsort(remaining_tempfit, len(remaining_tempfit))
            else:
                # 领飞鸟的邻域解不支配领飞鸟，对所有邻域解进行非支配排序
                _, _, tsortindex = fastsort(tempfit, len(tempfit))


    # 选取前 6 个解
    top_6_indices = []
    # 先从第 0 层取数
    top_6_indices.extend(tsortindex[0])

    # 如果第 0 层的解数量不足 6 个，从第 1 层取数
    if len(top_6_indices) < 6:
        remaining_num = 6 - len(top_6_indices)
        top_6_indices.extend(tsortindex[1][:remaining_num])

    left_shared_solutions = [indiv() for _ in range(3)]
    right_shared_solutions = [indiv() for _ in range(3)]
    left_index = 0
    right_index = 0
    for i, index in enumerate(top_6_indices):
        if is_dominated:
            # 排除支配的解后，重新映射索引
            original_index = [j for j in range(len(tempfit)) if j != max_index][index]
        else:
            original_index = index
        templead[original_index].fitness = tempfit[original_index]
        if i % 2 == 0:
            if right_index < 3:
                right_shared_solutions[right_index] = templead[original_index]
                right_index += 1
        else:
            if left_index < 3:
                left_shared_solutions[left_index] = templead[original_index]
                #print("left_shared_solutions[left_index].TA:", left_shared_solutions[left_index].TA)
                left_index += 1
    return left_shared_solutions, right_shared_solutions

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

def show_result(minitpop, num):
    for i in range(num):
        print('问题编码依次为：')
        print(f'TA向量为:{minitpop[i].TA}')
        print(f'AM向量为：{minitpop[i].AM}')
        print(f'WTS向量为：{minitpop[i].WTS}')
        print(f'CTS向量为：{minitpop[i].CTS}')
        print(f'任务的完成时间为:{minitpop[i].FT}')
        print(f'目标函数值为:{minitpop[i].fitness}')


def maincode2():
    #种群初始化
    start_time = time.time()
    pop=[indiv() for _ in range (PN)]
    fit = [[-1 for _ in range(3)] for _ in range(PN)]

    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            popinitial(pop,PN)
    minCT = [-1 for _ in range(T)]
    minER = [-1 for _ in range(T)]
    minEC = [-1 for _ in range(T)]


    hyper = [0 for _ in range(T)]
    space = [0 for _ in range(T)]

    for i in range(PN):
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                fit[i][0], fit[i][1], fit[i][2], pop[i].FT = decode(pop[i].TA, pop[i].AM, pop[i].WTS, pop[i].CTS)
        pop[i].fitness = fit[i]
        # 非支配排序:klevel是总层数，levellength是每层的项数，index是每层的项集合
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            klevel, levellength, sortindex = fastsort(fit, PN)
            Leaderindex, flindex, frindex, Followindex = partpop1(klevel, levellength, sortindex)

    curNAP = 0
    Archivepop = [indiv() for _ in range(maxNAP)]
    Archivepop, curNAP = get_archive_format(pop,curNAP,maxNAP)

    for t in range(T):
        for fn in range(FN):
            # 领飞鸟更新:生成七个邻域，采用非支配排序+拥挤度
            templead = [indiv() for _ in range(4)]
            tempfit = [[-1 for _ in range(3)] for _ in range(4)]

            #左侧跟飞鸟产生的领域解数量
            num_neighbors_left = 2
            #右侧跟飞鸟产生的领域解数量
            num_neighbors_right = 2
            #领飞鸟产生的共享解数量
            num_shared_solutions = 3

            for i in range(4):
                copy(pop[Leaderindex], templead[i])

            with open(os.devnull, 'w') as f:
                with contextlib.redirect_stdout(f):
                    is_dominated = False

                    print(f"当前领飞鸟安排为{pop[Leaderindex].TA}")

                    TAop(templead[0])
                    print(f"TA邻域安排为{templead[0].TA}")

                    AMop(templead[1])
                    print(f"AM邻域安排为{templead[1].TA}")

                    WTSop(templead[2])
                    print(f"WTS邻域安排为{templead[2].TA}")

                    CTSop(templead[3])
                    print(f"CTS邻域安排为{templead[3].TA}")

                    #考虑要不要删除
                    # critical_cmax(templead[4])
                    # print(f"critical_cmax邻域安排为{templead[4].TA}")
                    #
                    # critical_worker(templead[5])
                    # print(f"critical_worker邻域安排为{templead[5].TA}")
                    #
                    # critical_cobot(templead[6])
                    # print(f"critical_cobot{templead[6].TA}")

                    # 计算目标值
                    for i in range(4):
                        tempfit[i][0], tempfit[i][1], tempfit[i][2], templead[i].FT = decode(templead[i].TA, templead[i].AM,
                                                                             templead[i].WTS, templead[i].CTS)
                    # 非支配排序
                    tklevel, tlevellength, tsortindex = fastsort(tempfit, 4)
                    # 拥挤度计算
                    crowdist = crowd(tempfit, 4, 3)

                    k = 0
                    while tempfit[k] == fit[Leaderindex]:
                        k = k + 1
                        if k >= 4:
                            break

                    # 找不到和当前解不一样的个体
                    if k < 4:
                        max_index = k
                        for i in range(k, tlevellength[0]):
                            if tempfit[i] == fit[Leaderindex]:
                                continue
                            if crowdist[tsortindex[0][i]] > crowdist[tsortindex[0][max_index]]:
                                max_index = i
                        print("选出来的解是", max_index)
                        print(crowdist)
                        # 判断这个解是否支配lead
                        if dominates(tempfit[max_index], fit[Leaderindex]):
                            pop[Leaderindex] = templead[max_index]
                            fit[Leaderindex] = tempfit[max_index]
                            is_dominated = True
                            print("个体支配旧解")
                        # 如果互不支配，则替换
                        else:
                            if dominates(fit[Leaderindex], tempfit[max_index]) == False:
                                pop[Leaderindex] = templead[max_index]
                                fit[Leaderindex] = tempfit[max_index]
                                is_dominated = True

            left_shared_solutions, right_shared_solutions = select_shared_solution(templead,tempfit, max_index, is_dominated)

            #左跟飞鸟侧重于机器能耗和工人疲劳值最小
            #右跟飞鸟侧重于完工时间最短
            with open(os.devnull, 'w') as f:
                with contextlib.redirect_stdout(f):
                    for i in Followindex:
                        # 如果是左边，则采用TA和WTS
                        if i % 2 != 0:
                            left_tempf = [indiv() for _ in range(NS)]
                            left_tfit = [[-1 for _ in range(3)] for _ in range(NS)]
                            for ind in range(2):
                                copy(pop[i], left_tempf[ind])
                            TAop(left_tempf[0])
                            WTSop(left_tempf[1])
                            # critical_cobot(left_tempf[2])
                            # critical_worker(left_tempf[3])
                            # 计算目标值
                            for i in range(2):
                                left_tfit[i][0], left_tfit[i][1], left_tfit[i][2], left_tempf[i].FT = decode(left_tempf[i].TA, left_tempf[i].AM, left_tempf[i].WTS,
                                                                                 left_tempf[i].CTS)

                            for i in range(num_shared_solutions):
                                copy(left_shared_solutions[i], left_tempf[i + num_neighbors_left])

                            # 非支配排序
                            tklevel, tlevellength, tsortindex = fastsort(left_tfit, NS)
                            # 拥挤度计算
                            crowdist = crowd(left_tfit, NS, 3)
                            max_index = 0
                            for i in range(tlevellength[0]):
                                if crowdist[tsortindex[0][i]] > crowdist[tsortindex[0][max_index]]:
                                    max_index = i

                            if dominates(left_tfit[max_index], fit[i]):
                                pop[i] = left_tempf[max_index]
                                fit[i] = left_tfit[max_index]
                        else:
                            # 如果是右边，则采用AM和CTS
                            right_tempf = [indiv() for _ in range(NS)]
                            right_tfit = [[-1 for _ in range(3)] for _ in range(NS)]
                            for ind in range(2):
                                copy(pop[i], right_tempf[ind])
                            AMop(right_tempf[0])
                            CTSop(right_tempf[1])
                            # critical_cmax(right_tempf[2])
                            # 计算目标值
                            for i in range(2):
                                right_tfit[i][0], right_tfit[i][1], right_tfit[i][2], right_tempf[i].FT = decode(right_tempf[i].TA, right_tempf[i].AM, right_tempf[i].WTS,
                                                                                 right_tempf[i].CTS)

                            for i in range(num_shared_solutions):
                                copy(right_shared_solutions[i], right_tempf[i + num_neighbors_right])

                            # 非支配排序
                            tklevel, tlevellength, tsortindex = fastsort(right_tfit, NS)
                            # 拥挤度计算
                            crowdist = crowd(right_tfit, NS, 3)
                            max_index = 0
                            for i in range(tlevellength[0]):
                                if crowdist[tsortindex[0][i]] > crowdist[tsortindex[0][max_index]]:
                                    max_index = i

                            if dominates(right_tfit[max_index], fit[i]):
                                pop[i] = right_tempf[max_index]
                                fit[i] = right_tfit[max_index]

        #MBO算法通过模拟候鸟的迁徙行为，利用群体间的协作和信息共享来优化问题的解
        #不需要进行交叉处理，在巡回后进行领飞鸟的替换，并且将领飞鸟放到pop队列的最后

        # 重新计算适应度并排序
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                klevel, levellength, sortindex = fastsort(fit, PN)
                for i in range(levellength[0]):
                    Archivepop, curNAP = archivechange(Archivepop, curNAP, pop[sortindex[0][i]], fit[sortindex[0][i]],
                                                       maxNAP)
                Leaderindex, flindex, frindex, Followindex = partpop1(klevel, levellength, sortindex)


        #这里对领飞鸟的替换，采用随机选择第0层的个体，进而重新划分左右跟飞鸟

        print(f"——————————————————第{t}次迭代结果————————————————-")

        domfit = [[0 for _ in range(3)] for _ in range(curNAP)]
        #arcfit = [[-1 for _ in range(3)] for _ in range(maxNAP)]
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(curNAP):
                    Archivepop[i].fitness[0], Archivepop[i].fitness[1], Archivepop[i].fitness[2], Archivepop[i].FT = decode(Archivepop[i].TA, Archivepop[i].AM,
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
            print('MOMBO-domfit:',domfit)
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

    return Archivepop,hyper,space
    # return hyper[T - 1], space[T - 1], s_mean


if __name__ == '__main__':
    # 这里是有改进算子
    # for i in range(9):
    #     hv,space,s_mean = maincode2()
    #     print(f"第{i + 1}次运行后的HV为：{hv}")
    #     print(f"第{i + 1}次运行后的Spacing为：{space}")
    #     print(f"第{i + 1}次运行后的Spacing的均值为：{s_mean}")

    Archivepop, hyper, space =maincode2()
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

