from algorithmfunc import *
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from pymoo.vendor import hv as HH
from Show_Gante import plot_pareto_front

# IMOMBO算法的参数列表
# 种群规模
PN =20
# 迭代次数
T = 30
# 巡回次数
FN = 6
# 最大外部存档数
maxNAP = PN


def maincode():
    # 种群初始化
    minitpop = [indiv() for _ in range(PN)]
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            popinitial(minitpop, PN)

    curNAP = 0
    Archivepop = [indiv() for _ in range(maxNAP)]

    start_time = time.time()
    # 目标值
    fit = [[-1 for _ in range(objnum)] for _ in range(PN)]
    pop = [indiv() for _ in range(PN)]
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            # 初始化
            for i in range(PN):
                copy(minitpop[i], pop[i])
            for i in range(PN):
                fit[i][0], fit[i][1], pop[i].FT = decode1(pop[i].TA, pop[i].AM, pop[i].WTS, pop[i].CTS)
            # 非支配排序:klevel是总层数，levellength是每层的项数，index是每层的项集合
            klevel, levellength, sortindex = fastsort(fit, PN)
            Leaderindex, flindex, frindex, freeindex, Followindex = partpop(klevel, levellength, sortindex, PN)
            print(Leaderindex, flindex, frindex, freeindex, Followindex)

    print("------初始化种群--------")
    for i in range(PN):
        print(f'第{i}个个体的编码为：')
        print(f'TA向量为:{minitpop[i].TA}')
        print(f'AM向量为：{minitpop[i].AM}')
        print(f'WTS向量为：{minitpop[i].WTS}')
        print(f'CTS向量为：{minitpop[i].CTS}')
        print(f'目标函数值为：{fit[i]}')

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
                for i in range(maxNAP):
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
                    print(f"当前领飞鸟安排为{pop[Leaderindex].TA}")

                    TAop(templead[0])
                    print(f"TA邻域安排为{templead[0].TA}")

                    AMop(templead[1])
                    print(f"AM邻域安排为{templead[1].TA}")

                    WTSop(templead[2])
                    print(f"WTS邻域安排为{templead[2].TA}")

                    CTSop(templead[3])
                    print(f"CTS邻域安排为{templead[3].TA}")

                    critical_cmax(templead[4])
                    print(f"critical_cmax邻域安排为{templead[4].TA}")

                    critical_worker(templead[5])
                    print(f"critical_worker邻域安排为{templead[5].TA}")

                    critical_cobot(templead[6])
                    print(f"critical_cobot{templead[6].TA}")
                    # 计算目标值
                    for i in range(7):
                        tempfit[i][0], tempfit[i][1], templead[i].FT = decode1(templead[i].TA,
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
                        tempf = [indiv() for _ in range(3)]
                        tfit = [[-1 for _ in range(objnum)] for _ in range(3)]
                        for ind in range(3):
                            copy(pop[i], tempf[ind])

                        # 如果是左边，则采用TA和WTS
                        if i % 2 != 0:
                            TAop(tempf[0])
                            WTSop(tempf[1])
                            # critical_cmax(tempf[2])
                            critical_worker(tempf[objnum])
                            # 计算目标值
                            for i in range(3):
                                tfit[i][0], tfit[i][1], tempf[i].FT = decode1(tempf[i].TA, tempf[i].AM,
                                                                                          tempf[i].WTS,
                                                                                          tempf[i].CTS)

                            # 非支配排序
                            tklevel, tlevellength, tsortindex = fastsort(tfit, 3)
                            # 拥挤度计算
                            crowdist = crowd(tfit, 3, objnum)
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
                            # critical_cobot(tempf[3])
                            # 计算目标值
                            for i in range(3):
                                tfit[i][0], tfit[i][1], tempf[i].FT = decode1(tempf[i].TA, tempf[i].AM,
                                                                                          tempf[i].WTS,
                                                                                          tempf[i].CTS)

                            # 非支配排序
                            tklevel, tlevellength, tsortindex = fastsort(tfit, 3)
                            # 拥挤度计算
                            crowdist = crowd(tfit, 3, objnum)
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
                    print(
                        f"父代{flindex[i]}和{frindex[i]}安排为{pop[flindex[i]].TA, pop[frindex[i]].TA},模式={crossselect}")
                    if crossselect == 0:
                        pop[flindex[i]], pop[frindex[i]] = TAcross(pop[flindex[i]], pop[frindex[i]], t, T)
                    if crossselect == 1:
                        pop[flindex[i]], pop[frindex[i]] = AMcross(pop[flindex[i]], pop[frindex[i]], t, T)
                    if crossselect == 2:
                        pop[flindex[i]], pop[frindex[i]] = workercross(pop[flindex[i]], pop[frindex[i]], t, T)
                    if crossselect == 3:
                        pop[flindex[i]], pop[frindex[i]] = cobotcross(pop[flindex[i]], pop[frindex[i]], t, T)

        # 自由鸟更新：随机选择一种交叉策略，两两交叉
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


        # 在整体更新迭代以后，是否更新了领飞鸟？在partpop中已更新

        # 重新计算适应度并排序,更新外部存档
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                for i in range(PN):
                    wcadjust(pop[i].TA, pop[i].AM, pop[i].WTS, pop[i].CTS)
                    fit[i][0], fit[i][1], pop[i].FT = decode1(pop[i].TA, pop[i].AM, pop[i].WTS, pop[i].CTS)
                    # 非支配排序:klevel是总层数，levellength是每层的项数，index是每层的项集合
                klevel, levellength, sortindex = fastsort(fit, PN)
                Leaderindex, flindex, frindex, freeindex, Followindex = partpop(klevel, levellength, sortindex, PN)
                # 判断非支配解能否放入外部档案
                print("===============================非支配解判断")
                for i in range(levellength[0]):
                    Archivepop, curNAP = archivechange(Archivepop, curNAP, pop[sortindex[0][i]],
                                                       fit[sortindex[0][i]], maxNAP)

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
                    Archivepop[i].fitness[0], Archivepop[i].fitness[1], Archivepop[
                        i].FT = decode1(Archivepop[i].TA, Archivepop[i].AM,
                                        Archivepop[i].WTS, Archivepop[i].CTS)

                    domfit[i] = Archivepop[i].fitness

        # for i in range(curNAP):
        #     print('问题编码依次为：')
        #     print(f'TA向量为:{Archivepop[i].TA}')
        #     print(f'AM向量为：{Archivepop[i].AM}')
        #     print(f'WTS向量为：{Archivepop[i].WTS}')
        #     print(f'CTS向量为：{Archivepop[i].CTS}')
        #     print(f'任务的完成时间为:{Archivepop[i].FT}')
        #     print(f'目标函数值为:{Archivepop[i].fitness}')

        if t == T - 1:
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

        reference_point = [20, 5]  #P12
        # reference_point = [90 ,10]

        ind = HH.HyperVolume(reference_point)
        hv = ind.compute(data)

        spacing = calculate_spacing(data)

        print("-----------------解的评估指标----------------—-")
        print(f"Hypervolume (HV): {hv}")
        hyper[t] = hv
        space[t] = spacing
        print(f"Spacing: {spacing}")

    end_time = time.time()
    print(f"算法运行时间：{end_time - start_time} 秒")
    # return domfit
    return Archivepop, hyper, space, curNAP


if __name__ == '__main__':
    # 这里是有改进算子
    curNAP = 0
    Archivepop = [indiv() for _ in range(maxNAP)]

    Archivepop, hyper, space, curNAP = maincode()

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

    # plot_pareto_front(Archivepop,curNAP)
