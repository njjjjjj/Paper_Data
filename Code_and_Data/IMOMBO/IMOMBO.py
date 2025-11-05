# from draw import obj_iteration
from algorithmfunc import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from pymoo.vendor import hv as HH

# IMOMBO算法的参数列表
# 种群规模
PN = 30
# 迭代次数
T = 50
# 巡回次数
FN = 5
# 最大外部存档数
maxNAP = 20

def maincode():
    # 种群初始化
    start_time = time.time()
    minitpop = [indiv() for _ in range(PN)]
    popinitial(minitpop, PN)
    minCT = [-1 for _ in range(T)]
    minER = [-1 for _ in range(T)]
    minEC = [-1 for _ in range(T)]

    curNAP = 0
    Archivepop = [indiv() for _ in range(maxNAP)]

    # 目标值
    fit = [[-1 for _ in range(objnum)] for _ in range(PN)]
    pop = [indiv() for _ in range(PN)]
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            # 初始化
            for i in range(PN):
                copy(minitpop[i], pop[i])
            for i in range(PN):
                fit[i][0], fit[i][1], fit[i][2], pop[i].FT = decode(pop[i].TA, pop[i].AM, pop[i].WTS, pop[i].CTS)
            # 非支配排序:klevel是总层数，levellength是每层的项数，index是每层的项集合
            klevel, levellength, sortindex = fastsort(fit, PN)
            Leaderindex, flindex, frindex, freeindex, Followindex = partpop(klevel, levellength, sortindex, PN)
            print(Leaderindex, flindex, frindex, freeindex, Followindex)

    print("------初始化种群--------")
    for i in range(PN):
        print(f'第{i}个个体的编码为：')
        print(f'TA向量为:{pop[i].TA}')
        print(f'AM向量为：{pop[i].AM}')
        print(f'WTS向量为：{pop[i].WTS}')
        print(f'CTS向量为：{pop[i].CTS}')
        print(f'目标函数值为：{fit[i]}')

    #外部存档方式对吗？
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            copy(pop[sortindex[0][0]], Archivepop[0])
            curNAP = curNAP + 1
            # 初始时，外部档案从当前非支配等级为0的个体中获得;如果个体数超过NAP，则采用拥挤距离计算放入
            if maxNAP > levellength[0]:
                for i in range(levellength[0]):
                    Archivepop, curNAP = archivechange(Archivepop, curNAP, pop[sortindex[0][i]], fit[sortindex[0][i]],
                                                       maxNAP)
            else:
                tempfit = [[0 for _ in range(objnum)] for _ in range(levellength[0])]
                for i in range(levellength[0]):
                    tempfit[i] = fit[sortindex[0][i]]
                crowdist = crowd(tempfit, levellength[0], objnum)
                # 对序号按crowdist中的值排序，选前maxNAP个解放入
                sorted_indices = sort_indices_by_values(crowdist)
                for i in sorted_indices:
                    Archivepop, curNAP = archivechange(Archivepop, curNAP, pop[sortindex[0][i]], fit[sortindex[0][i]],
                                                       maxNAP)

    hyper = [0 for _ in range(T)]
    space = [0 for _ in range(T)]

    for t in range(T):
        # 巡回ing
        for fn in range(FN):
            # 领飞鸟更新:生成七个邻域，采用非支配排序+拥挤度
            templead = [indiv() for _ in range(7)]
            tempfit = [[-1 for _ in range(objnum)] for _ in range(7)]
            for i in range(7):
                copy(pop[Leaderindex], templead[i])

            # if t==T-1:
            #     print(f"当前领飞鸟安排为{pop[Leaderindex].TA}")
            #     templead[0]=TAop(templead[0])
            #     print(f"TA邻域安排为{templead[0].TA}")

            with open(os.devnull, 'w') as f:
                with contextlib.redirect_stdout(f):
                    # print(f"当前领飞鸟安排为{pop[Leaderindex].TA}")
                    #
                    TAop(templead[0])
                    print(f"TA邻域安排为{templead[0].TA}")

                    AMop(templead[1])
                    # print(f"AM邻域安排为{templead[0].AM}")

                    WTSop(templead[2])
                    # print(f"WTS邻域安排为{templead[1].WTS}")

                    CTSop(templead[3])
                    # print(f"CTS邻域安排为{templead[2].CTS}")

                    critical_cmax(templead[4])
                    # print(f"critical_cmax邻域安排为{templead[3].TA}")

                    critical_worker(templead[5])
                    # print(f"critical_worker邻域安排为{templead[1].TA}")

                    critical_cobot(templead[6])
                    # print(f"critical_cobot{templead[2].TA}")
                    # 计算目标值
                    for i in range(7):
                        tempfit[i][0], tempfit[i][1], tempfit[i][2], templead[i].FT = decode(templead[i].TA,
                                                                                             templead[i].AM,
                                                                                             templead[i].WTS,
                                                                                             templead[i].CTS)
                    # 非支配排序
                    tklevel, tlevellength, tsortindex = fastsort(tempfit, 7)
                    # 拥挤度计算
                    crowdist = crowd(tempfit, 7, objnum)

                    k = 0
                    f = 0
                    while tempfit[k] != fit[Leaderindex]:
                        k = k + 1
                        if k >= 7:
                            break

                    # 找不到和当前解不一样的个体
                    if k < 7:
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
                            print("个体支配旧解")
                        # 如果互不支配，则替换
                        else:
                            if dominates(fit[Leaderindex], tempfit[max_index]) == False:
                                pop[Leaderindex] = templead[max_index]
                                print("互不支配")
                                '''else:
                                    randdata=random.random()
                                    if randdata<math.exp(-t/T):
                                        pop[Leaderindex]=templead[max_index]'''

                    # 跟飞鸟更新：每个生成一个邻域更新
                    for i in Followindex:
                        tempf = [indiv() for _ in range(4)]
                        tfit = [[-1 for _ in range(objnum)] for _ in range(4)]
                        for ind in range(4):
                            copy(pop[i], tempf[ind])

                        # 如果是左边，则采用TA和WTS
                        if i % 2 != 0:
                            TAop(tempf[0])
                            WTSop(tempf[1])
                            critical_cmax(tempf[2])
                            critical_worker(tempf[3])
                            # 计算目标值
                            for i in range(4):
                                tfit[i][0], tfit[i][1], tfit[i][2], tempf[i].FT = decode(tempf[i].TA, tempf[i].AM,
                                                                                         tempf[i].WTS,
                                                                                         tempf[i].CTS)

                            # 非支配排序
                            tklevel, tlevellength, tsortindex = fastsort(tfit, 4)
                            # 拥挤度计算
                            crowdist = crowd(tfit, 4, objnum)
                            max_index = 0
                            for i in range(tlevellength[0]):
                                if crowdist[tsortindex[0][i]] > crowdist[tsortindex[0][max_index]]:
                                    max_index = i

                            if dominates(tfit[max_index], fit[i]):
                                pop[i] = tempf[max_index]
                                fit[i] = tfit[max_index]
                        else:
                            # 如果是右边，则采用AM和CTS
                            AMop(tempf[0])
                            CTSop(tempf[1])
                            critical_cmax(tempf[2])
                            critical_cobot(tempf[3])
                            # 计算目标值
                            for i in range(4):
                                tfit[i][0], tfit[i][1], tfit[i][2], tempf[i].FT = decode(tempf[i].TA, tempf[i].AM,
                                                                                         tempf[i].WTS,
                                                                                         tempf[i].CTS)

                            # 非支配排序
                            tklevel, tlevellength, tsortindex = fastsort(tfit, 4)
                            # 拥挤度计算
                            crowdist = crowd(tfit, 4, objnum)
                            max_index = 0
                            for i in range(tlevellength[0]):
                                if crowdist[tsortindex[0][i]] > crowdist[tsortindex[0][max_index]]:
                                    max_index = i

                            if dominates(tfit[max_index], fit[i]):
                                pop[i] = tempf[max_index]
                                fit[i] = tfit[max_index]
                            # Archivepop,curNAP=archivechange(Archivepop,curNAP,pop[i],fit[i])

                    '''print("===============================跟飞鸟判断")
                    for i in Followindex:     
                        #判断能否放入外部档案
                        Archivepop,curNAP=archivechange(Archivepop,curNAP,pop[i],fit[i])'''

        # 如果达到巡回次数
        # 左右跟飞鸟交叉进化
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(int(PN / 3)):
                    # 随机生成一种策略
                    crossselect = random.randint(0, 3)
                    print(f"父代{flindex[i]}和{frindex[i]}安排为{pop[flindex[i]].TA, pop[frindex[i]].TA},模式={crossselect}")
                    if crossselect == 0:
                        pop[flindex[i]], pop[frindex[i]] = TAcross(pop[flindex[i]], pop[frindex[i]], t,T)
                    if crossselect == 1:
                        pop[flindex[i]], pop[frindex[i]] = AMcross(pop[flindex[i]], pop[frindex[i]], t,T)
                    if crossselect == 2:
                        pop[flindex[i]], pop[frindex[i]] = workercross(pop[flindex[i]], pop[frindex[i]], t, T)
                    if crossselect == 3:
                        pop[flindex[i]], pop[frindex[i]] = cobotcross(pop[flindex[i]], pop[frindex[i]], t,T)

        # 自由鸟更新：随机选择一种变异策略/交叉策略，两两交叉
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(len(freeindex)):
                    print("=============================")
                    mutationselect = random.randint(0, 6)
                    if mutationselect == 0:
                        AMop(pop[freeindex[i]])
                    elif mutationselect == 1:
                        CTSop(pop[freeindex[i]])
                    elif mutationselect == 2:
                        TAop(pop[freeindex[i]])
                    elif mutationselect == 3:
                        WTSop(pop[freeindex[i]])
                    elif mutationselect == 4:
                        critical_cmax(pop[freeindex[i]])
                    elif mutationselect == 5:
                        critical_worker(pop[freeindex[i]])
                    else:
                        critical_cobot(pop[freeindex[i]])

        # 重新计算适应度并排序,更新外部存档
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(PN):
                    wcadjust(pop[i].TA, pop[i].AM, pop[i].WTS, pop[i].CTS)
                    fit[i][0], fit[i][1], fit[i][2], pop[i].FT = decode(pop[i].TA, pop[i].AM, pop[i].WTS, pop[i].CTS)
                    # 非支配排序:klevel是总层数，levellength是每层的项数，index是每层的项集合
                klevel, levellength, sortindex = fastsort(fit, PN)
                Leaderindex, flindex, frindex, freeindex, Followindex = partpop(klevel, levellength, sortindex, PN)
                # 判断非支配解能否放入外部档案
                print("===============================非支配解判断")
                for i in range(levellength[0]):
                    Archivepop, curNAP = archivechange(Archivepop, curNAP, pop[sortindex[0][i]],
                                                       fit[sortindex[0][i]],maxNAP)

        # normalizer= Normalizer(3)
        # fit_array = np.array(fit)
        # normalizer.update_bounds(fit_array)

        print(f"——————————————————第{t}次迭代结果————————————————-")
        # 输出非支配等级为0的个体
        '''domfit=[[0 for _ in range(3)] for _ in range(levellength[0])]
        for i in range(levellength[0]):  
            print(fit[sortindex[0][i]])
            domfit[i]=fit[sortindex[0][i]]'''

        domfit = [[0 for _ in range(objnum)] for _ in range(curNAP)]
        # arcfit = [[-1 for _ in range(3)] for _ in range(maxNAP)]
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(curNAP):
                    Archivepop[i].fitness[0], Archivepop[i].fitness[1], Archivepop[i].fitness[2], Archivepop[
                        i].FT = decode(Archivepop[i].TA, Archivepop[i].AM,
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
            print('IMOMBO-domfit:',domfit)
            for i in range(curNAP):
                print('问题编码依次为：')
                print(f'TA向量为:{Archivepop[i].TA}')
                print(f'AM向量为：{Archivepop[i].AM}')
                print(f'WTS向量为：{Archivepop[i].WTS}')
                print(f'CTS向量为：{Archivepop[i].CTS}')
                print(f'任务的完成时间为:{Archivepop[i].FT}')
                print(f'目标函数值为:{Archivepop[i].fitness}')

        data = np.array(domfit)
        # reference_point = np.min(data, axis=0) - 1  # 参考点（设置比最小值更小）

        # reference_point = [15, 2.5, 9] #P12
        # reference_point = [150,5,80]  #P36

        # reference_point = [35, 5, 20]  #P24
        reference_point = [70,5,50]

        # normalize_objectives = normalizer.normalize(data)
        # reference_point = [1.0, 1.0, 1.0]

        ind = HH.HyperVolume(reference_point)
        hv = ind.compute(data)

        spacing = calculate_spacing(data)

        print("-----------------解的评估指标----------------—-")
        print(f"Hypervolume (HV): {hv}")
        hyper[t] = hv
        space[t] = spacing
        print(f"Spacing: {spacing}")

    # s_min = min(space)
    # s_max = max(space)
    # s_mean = sum(space) / len(space)
    # print(f"Spacing min: {s_min}")
    # print(f"Spacing max: {s_max}")
    # print(f"Spacing mean: {s_mean}")

    print('minCT',minCT)
    print('minER',minER)
    print('minEC',minEC)

    end_time = time.time()
    print(f"算法运行时间：{end_time - start_time} 秒")
    # print(f'CT迭代变化：{minCT}')
    # print(f'ER迭代变化：{minER}')
    # print(f'EC迭代变化：{minEC}')

    # npop = select_top_crowded_solutions(Archivepop, 3)
    # print("基于拥挤度距离获取的10个Pareto解：")
    # for i in range(len(npop)):
    #     print(npop[i].fitness)

    # return domfit
    return Archivepop, hyper, space, curNAP

    # return hyper[T-1], space[T-1], s_mean


if __name__ == '__main__':
    # 这里是有改进算子
    curNAP = 0
    Archivepop = [indiv() for _ in range(maxNAP)]

    Archivepop, hyper, space, curNAP = maincode()

    # for i in range(4):
    #     hv,space,space_mean = maincode()
        # print(f"第{i + 1}次运行后的HV为：{hv}")
        # print(f"第{i + 1}次运行后的Spacing为：{space}")
        # print(f"第{i + 1}次运行后的Spacing的均值为：{space_mean}")
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
