from localsearch import *


# TA交叉,t表示当前迭代次数，用于接收劣解
def TAcross(mindiv1, mindiv2, t, T):
    print(f'mindiv1.TA和mindiv2.TA分别为：{mindiv1.TA, mindiv2.TA}')

    # 生成Ntask长的0-1序列
    randflag = [random.randint(0, 1) for _ in range(Ntask)]
    print(randflag)

    tempmindiv = [indiv() for _ in range(2)]

    # 生成子代
    for i in range(Ntask):
        tempmindiv[0].TA[i] = mindiv1.TA[i] * (1 - randflag[i]) + mindiv2.TA[i] * (randflag[i])
        tempmindiv[1].TA[i] = mindiv1.TA[i] * (randflag[i]) + mindiv2.TA[i] * (1 - randflag[i])

    for ind in range(2):
        # 调整使任务符合先序约束
        TSA = [-1 for _ in range(Ntask)]
        dirc = [-1 for _ in range(Ntask)]
        for i in range(Ntask):
            TSA[i] = int(tempmindiv[ind].TA[i] / 2)
            dirc[i] = int(tempmindiv[ind].TA[i] % 2)
        mode = random.randint(0, 1)
        TSA, TA = adjust(TSA, mode)
        for i in range(Ntask):
            tempmindiv[ind].TA[i] = TSA[i] * 2 + dirc[i]

    for i in range(Ntask):
        tempmindiv[0].AM[i] = mindiv1.AM[i]
        tempmindiv[1].AM[i] = mindiv2.AM[i]
    for i in range(NW):
        tempmindiv[0].WTS[i] = mindiv1.WTS[i]
        tempmindiv[1].WTS[i] = mindiv2.WTS[i]
    for i in range(NC):
        tempmindiv[0].CTS[i] = mindiv1.CTS[i]
        tempmindiv[1].CTS[i] = mindiv2.CTS[i]

    # 计算实际分组
    for ind in range(2):

        PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
        PSnum = [0 for _ in range(NS * 2)]
        for i in range(Ntask):
            # 根据TA判断每个配对站的任务
            ps = tempmindiv[ind].TA[i]
            PSset[ps][PSnum[ps]] = i
            PSnum[ps] = PSnum[ps] + 1
        print(PSset)
        print(PSnum)

        # 机器和工人分配矩阵
        SW = [-1 for _ in range(NS * 2)]
        SC = [-1 for _ in range(NS * 2)]
        for i in range(NW):
            if tempmindiv[ind].WTS[i] != -1:
                SW[tempmindiv[ind].WTS[i]] = i
        for i in range(NC):
            if tempmindiv[ind].CTS[i] != -1:
                SC[tempmindiv[ind].CTS[i]] = i
        print(f"个体{ind}===")
        print(f"各工作站工人分配为:{SW}")
        print(f"各工作站机器人分配为:{SC}")
        print(f"WTS={tempmindiv[ind].WTS}")
        flag = [0 for _ in range(NW)]
        for i in range(NW):
            # 代表工人i已分配
            if tempmindiv[ind].WTS[i] != -1:
                flag[i] = 1
        print(f"flag={flag}")

        for station in range(NS * 2):
            if (SW[station] == -1) & (SC[station] == -1):
                # 从可选工人中随机选择一个
                zero_items = [index for index, value in enumerate(flag) if value == 0]
                random_Windex = random.choice(zero_items)
                SW[station] = random_Windex
                tempmindiv[ind].WTS[random_Windex] = station
                flag[random_Windex] = 1
                print(f"由于{station}啥也没有，所以将工人{random_Windex}分给它")

        print(f"分完后各工作站工人分配为:{SW},WTS={tempmindiv[ind].WTS}")
        # 判断调整后，各个任务模式是否可用，是否需要调整; 优先内部调整，内部调整不了在换工作站
        for station in range(NS * 2):
            for j in range(PSnum[station]):
                curtask = PSset[station][j]
                # 判断任务在当前工作站的可选模式
                flag = [0 for _ in range(3)]
                if SW[station] != -1:
                    flag[0] = 1
                if (TCtime[curtask][SC[station]] != -1) & (SC[station] != -1):
                    flag[1] = 1
                if (TWCtime[curtask][SW[station] * NC + SC[station]] != -1) & (SC[station] != -1) & (SW[station] != -1):
                    flag[2] = 1
                # 判断当前模式是否可用,可用则不处理，不可用则调整
                # 不可用
                if flag[tempmindiv[ind].AM[curtask]] != 1:
                    # 当前有可用其他模式，则调整模式
                    if sum(flag) != 0:
                        # 随机选择一个模式赋值给当前任务
                        non_zero_indices = [index for index, value in enumerate(flag) if value != 0]
                        tempmindiv[ind].AM[curtask] = random.choice(non_zero_indices)
                    else:
                        # 否则，调整
                        print(f"任务{curtask}需要调整")
                        tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW, tempmindiv[ind].WTS, tempmindiv[
                            ind].CTS = changemodestation(curtask, tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW)
        tempmindiv[ind].WTS, tempmindiv[ind].CTS = wcadjust(tempmindiv[ind].TA, tempmindiv[ind].AM, tempmindiv[ind].WTS,
                                                            tempmindiv[ind].CTS)
        print(f"最终调整后:", tempmindiv[ind].TA, tempmindiv[ind].AM)
        print(f"各工作站工人分配为:{SW}")
        print(f"各工作站机器人分配为:{SC}")
        print(f"WTS,CTS:", tempmindiv[ind].WTS, tempmindiv[ind].CTS)
        print(f"个体{ind}结束===")

    tempfit = [[-1 for _ in range(objnum)] for _ in range(2)]
    orignfit = [[-1 for _ in range(objnum)] for _ in range(2)]
    orignfit[0][0], orignfit[0][1], orignfit[0][2], mindiv1.FT = decode(mindiv1.TA, mindiv1.AM, mindiv1.WTS, mindiv1.CTS)
    orignfit[1][0], orignfit[1][1], orignfit[1][2], mindiv2.FT = decode(mindiv2.TA, mindiv2.AM, mindiv2.WTS, mindiv2.CTS)
    for ind in range(2):
        # 计算新个体适应度值
        tempfit[ind][0], tempfit[ind][1], tempfit[ind][2], tempmindiv[ind].FT = decode(tempmindiv[ind].TA, tempmindiv[ind].AM,
                                                                   tempmindiv[ind].WTS, tempmindiv[ind].CTS)
    # orignfit[0][0], orignfit[0][1], mindiv1.FT = decode1(mindiv1.TA, mindiv1.AM, mindiv1.WTS, mindiv1.CTS)
    # orignfit[1][0], orignfit[1][1], mindiv2.FT = decode1(mindiv2.TA, mindiv2.AM, mindiv2.WTS, mindiv2.CTS)
    # for ind in range(2):
    #     # 计算新个体适应度值
    #     tempfit[ind][0], tempfit[ind][1], tempmindiv[ind].FT = decode1(tempmindiv[ind].TA, tempmindiv[ind].AM,
    #                                                                tempmindiv[ind].WTS, tempmindiv[ind].CTS)
        # 如果新解被旧解支配，则以一定概率不更新旧解
        if dominates(orignfit[ind], tempfit[ind]):
            randdata = random.random()
            if randdata >= math.exp(-t / T):
                if ind == 0:
                    tempmindiv[0] = mindiv1
                if ind == 1:
                    tempmindiv[1] = mindiv2

    return tempmindiv[0], tempmindiv[1]

def AMcross(mindiv1, mindiv2, t,T):
    # 生成Ntask长的0-1序列
    randflag = [random.randint(0, 1) for _ in range(Ntask)]
    print(randflag)

    tempmindiv = [indiv() for _ in range(2)]
    print("交叉前:", mindiv1.AM, mindiv2.AM)
    # 生成子代
    for i in range(Ntask):
        tempmindiv[0].TA[i] = mindiv1.TA[i]
        tempmindiv[1].TA[i] = mindiv2.TA[i]
    for i in range(NW):
        tempmindiv[0].WTS[i] = mindiv1.WTS[i]
        tempmindiv[1].WTS[i] = mindiv2.WTS[i]
    for i in range(NC):
        tempmindiv[0].CTS[i] = mindiv1.CTS[i]
        tempmindiv[1].CTS[i] = mindiv2.CTS[i]

    for i in range(Ntask):
        if randflag[i] == 1:
            tempmindiv[0].AM[i] = mindiv1.AM[i]
            tempmindiv[1].AM[i] = mindiv2.AM[i]

    for ind in range(2):
        print(f"子代{ind}============")
        if ind == 0:
            ref = mindiv2
        else:
            ref = mindiv1
        k = 0
        for i in range(Ntask):
            # 没被选中过
            if randflag[i] == 0:
                while tempmindiv[ind].AM[k] != -1:
                    k = k + 1
                tempmindiv[ind].AM[k] = ref.AM[i]
                k = k + 1
    print("交叉后:", tempmindiv[0].AM, tempmindiv[1].AM)
    PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
    PSnum = [0 for _ in range(NS * 2)]
    for i in range(Ntask):
        # 根据TA判断每个配对站的任务
        ps = tempmindiv[ind].TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1
    print(PSset)
    print(PSnum)

    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if tempmindiv[ind].WTS[i] != -1:
            SW[tempmindiv[ind].WTS[i]] = i
    for i in range(NC):
        if tempmindiv[ind].CTS[i] != -1:
            SC[tempmindiv[ind].CTS[i]] = i

    print(f"各工作站工人分配为:{SW}")
    print(f"各工作站机器人分配为:{SC}")
    # 判断调整后，各个任务模式是否可用，是否需要调整; 优先内部调整，内部调整不了在换工作站
    for station in range(NS * 2):
        for j in range(PSnum[i]):
            curtask = PSset[station][j]
            # 判断任务在当前工作站的可选模式
            flag = [0 for _ in range(3)]
            if SW[station] != -1:
                flag[0] = 1
            if (TCtime[curtask][SC[station]] != -1) & (SC[station] != -1):
                flag[1] = 1
            if (TWCtime[curtask][SW[station] * NC + SC[station]] != -1) & (SC[station] != -1) & (SW[station] != -1):
                flag[2] = 1
            # 判断当前模式是否可用,可用则不处理，不可用则调整
            # 不可用
            if flag[tempmindiv[ind].AM[curtask]] != 1:
                # 当前有可用其他模式，则调整模式
                if sum(flag) != 0:
                    # 随机选择一个模式赋值给当前任务
                    non_zero_indices = [index for index, value in enumerate(flag) if value != 0]
                    tempmindiv[ind].AM[curtask] = random.choice(non_zero_indices)
                else:
                    # 否则，调整
                    print(f"任务{curtask}需要调整")
                    tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW, tempmindiv[ind].WTS, tempmindiv[ind].CTS = changemodestation(curtask, tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW)

    print(f"最终调整后:", tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW)

    # 计算实际分组
    for ind in range(2):
        print(f"子代{ind}============")
        PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
        PSnum = [0 for _ in range(NS * 2)]
        for i in range(Ntask):
            # 根据TA判断每个配对站的任务
            ps = tempmindiv[ind].TA[i]
            PSset[ps][PSnum[ps]] = i
            PSnum[ps] = PSnum[ps] + 1
        print(PSset)
        print(PSnum)

        # 机器和工人分配矩阵
        SW = [-1 for _ in range(NS * 2)]
        SC = [-1 for _ in range(NS * 2)]
        for i in range(NW):
            if tempmindiv[ind].WTS[i] != -1:
                SW[tempmindiv[ind].WTS[i]] = i
        for i in range(NC):
            if tempmindiv[ind].CTS[i] != -1:
                SC[tempmindiv[ind].CTS[i]] = i

        print(f"各工作站工人分配为:{SW}")
        print(f"各工作站机器人分配为:{SC}")

        print(f"WTS={tempmindiv[ind].WTS}")
        flag = [0 for _ in range(NW)]
        for i in range(NW):
            # 代表工人i已分配
            if tempmindiv[ind].WTS[i] != -1:
                flag[i] = 1
        print(f"flag={flag}")

        for station in range(NS * 2):
            if (SW[station] == -1) & (SC[station] == -1):
                # 从可选工人中随机选择一个
                zero_items = [index for index, value in enumerate(flag) if value == 0]
                random_Windex = random.choice(zero_items)
                SW[station] = random_Windex
                tempmindiv[ind].WTS[random_Windex] = station
                flag[random_Windex] = 1
                print(f"由于{station}啥也没有，所以将工人{random_Windex}分给它")

        print(f"分完后各工作站工人分配为:{SW},WTS={tempmindiv[ind].WTS}")
        # 判断调整后，各个任务模式是否可用，是否需要调整; 优先内部调整，内部调整不了在换工作站
        for station in range(NS * 2):
            if (PSset[station][0] != -1):
                for j in range(PSnum[station]):
                    curtask = PSset[station][j]
                    # 判断任务在当前工作站的可选模式
                    flag = [0 for _ in range(3)]
                    if SW[station] != -1:
                        flag[0] = 1
                    if (TCtime[curtask][SC[station]] != -1) & (SC[station] != -1):
                        flag[1] = 1
                    if (TWCtime[curtask][SW[station] * NC + SC[station]] != -1) & (SC[station] != -1) & (
                            SW[station] != -1):
                        flag[2] = 1
                    # 判断当前模式是否可用,可用则不处理，不可用则调整
                    # 不可用
                    if flag[tempmindiv[ind].AM[curtask]] != 1:
                        # 当前有可用其他模式，则调整模式
                        if sum(flag) != 0:
                            # 随机选择一个模式赋值给当前任务
                            non_zero_indices = [index for index, value in enumerate(flag) if value != 0]
                            tempmindiv[ind].AM[curtask] = random.choice(non_zero_indices)
                        else:
                            # 否则，调整
                            print(f"任务{curtask}需要调整")
                            tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW, tempmindiv[ind].WTS, tempmindiv[ind].CTS = changemodestation(curtask, tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW)
        tempmindiv[ind].WTS, tempmindiv[ind].CTS = wcadjust(tempmindiv[ind].TA, tempmindiv[ind].AM, tempmindiv[ind].WTS,tempmindiv[ind].CTS)
        print(f"最终调整后:", tempmindiv[ind].TA, tempmindiv[ind].AM)
        print(f"各工作站工人分配为:{SW}")
        print(f"各工作站机器人分配为:{SC}")
        print(f"WTS,CTS:", tempmindiv[ind].WTS, tempmindiv[ind].CTS)
        print(f"个体{ind}结束===")
    tempfit = [[-1 for _ in range(objnum)] for _ in range(2)]
    orignfit = [[-1 for _ in range(objnum)] for _ in range(2)]
    #修改
    orignfit[0][0], orignfit[0][1], orignfit[0][2], mindiv1.FT = decode(mindiv1.TA, mindiv1.AM, mindiv1.WTS, mindiv1.CTS)
    orignfit[1][0], orignfit[1][1], orignfit[1][2], mindiv2.FT = decode(mindiv2.TA, mindiv2.AM, mindiv2.WTS, mindiv2.CTS)
    for ind in range(2):
        # 计算新个体适应度值
        tempfit[ind][0], tempfit[ind][1], tempfit[ind][2], tempmindiv[ind].FT = decode(tempmindiv[ind].TA, tempmindiv[ind].AM,
                                                                   tempmindiv[ind].WTS, tempmindiv[ind].CTS)
    # orignfit[0][0], orignfit[0][1], mindiv1.FT = decode1(mindiv1.TA, mindiv1.AM, mindiv1.WTS, mindiv1.CTS)
    # orignfit[1][0], orignfit[1][1], mindiv2.FT = decode1(mindiv2.TA, mindiv2.AM, mindiv2.WTS, mindiv2.CTS)
    # for ind in range(2):
    #     # 计算新个体适应度值
    #     tempfit[ind][0], tempfit[ind][1], tempmindiv[ind].FT = decode1(tempmindiv[ind].TA, tempmindiv[ind].AM,
    #                                                                tempmindiv[ind].WTS, tempmindiv[ind].CTS)
        # 如果新解被旧解支配，则以一定概率不更新旧解
        if dominates(orignfit[ind], tempfit[ind]):
            randdata = random.random()
            if randdata >= math.exp(-t / T):
                if ind == 0:
                    tempmindiv[0] = mindiv1
                if ind == 1:
                    tempmindiv[1] = mindiv2
    return tempmindiv[0], tempmindiv[1]

def workercross(mindiv1, mindiv2, t, T):
    # 随机选择一个位置
    randloc = random.randint(0, NW - 1)
    tempmindiv = [indiv() for _ in range(2)]
    print("交叉前:", mindiv1.WTS, mindiv2.WTS)
    print(randloc)
    # 对于位置之前的直接放入
    for i in range(randloc):
        tempmindiv[0].WTS[i] = mindiv1.WTS[i]
        tempmindiv[1].WTS[i] = mindiv2.WTS[i]
    # 生成子代
    for i in range(Ntask):
        tempmindiv[0].TA[i] = mindiv1.TA[i]
        tempmindiv[1].TA[i] = mindiv2.TA[i]
        tempmindiv[0].AM[i] = mindiv1.AM[i]
        tempmindiv[1].AM[i] = mindiv2.AM[i]
    for i in range(NC):
        tempmindiv[0].CTS[i] = mindiv1.CTS[i]
        tempmindiv[1].CTS[i] = mindiv2.CTS[i]

    # 对于位置之后的
    for ind in range(2):
        print(f"子代{ind}============")
        flag = [0 for _ in range(NW + 1)]
        for i in range(randloc):
            if tempmindiv[ind].WTS[i] != -1:
                flag[tempmindiv[ind].WTS[i]] = 1
            else:
                flag[NW] = 1
        if ind == 0:
            ref = mindiv2
        else:
            ref = mindiv1

        k = randloc
        print(flag)
        for i in range(NW):
            if (ref.WTS[i] == -1) & (k < NW):
                tempmindiv[ind].WTS[k] = -1
                flag[NW] = 1
                k = k + 1

            else:
                if (flag[ref.WTS[i]] != 1) & (k < NW):
                    tempmindiv[ind].WTS[k] = ref.WTS[i]
                    flag[ref.WTS[i]] = 1
                    k = k + 1
        print(f"交叉后,WTS={tempmindiv[ind].WTS}")

    # 计算实际分组
    for ind in range(2):
        print(f"子代{ind}============")
        PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
        PSnum = [0 for _ in range(NS * 2)]
        for i in range(Ntask):
            # 根据TA判断每个配对站的任务
            ps = tempmindiv[ind].TA[i]
            PSset[ps][PSnum[ps]] = i
            PSnum[ps] = PSnum[ps] + 1
        print(PSset)
        print(PSnum)

        # 机器和工人分配矩阵
        SW = [-1 for _ in range(NS * 2)]
        # 机器和工人分配矩阵
        SC = [-1 for _ in range(NS * 2)]
        for i in range(NW):
            if tempmindiv[ind].WTS[i] != -1:
                SW[tempmindiv[ind].WTS[i]] = i

        for i in range(NC):
            if tempmindiv[ind].CTS[i] != -1:
                SC[tempmindiv[ind].CTS[i]] = i

        print(f"各工作站工人分配为:{SW}")
        print(f"各工作站机器人分配为:{SC}")
        print(f"WTS={tempmindiv[ind].WTS}")
        flag = [0 for _ in range(NW)]
        for i in range(NW):
            # 代表工人i已分配
            if tempmindiv[ind].WTS[i] != -1:
                flag[i] = 1
        print(f"flag={flag}")

        for station in range(NS * 2):
            if (SW[station] == -1) & (SC[station] == -1):
                # 从可选工人中随机选择一个
                zero_items = [index for index, value in enumerate(flag) if value == 0]
                random_Windex = random.choice(zero_items)
                SW[station] = random_Windex
                tempmindiv[ind].WTS[random_Windex] = station
                flag[random_Windex] = 1
                print(f"由于{station}啥也没有，所以将工人{random_Windex}分给它")

        print(f"分完后各工作站工人分配为:{SW},WTS={tempmindiv[ind].WTS}")

        # 判断调整后，各个任务模式是否可用，是否需要调整; 优先内部调整，内部调整不了在换工作站
        for station in range(NS * 2):
            if (PSset[station][0] != -1):
                for j in range(PSnum[station]):
                    curtask = PSset[station][j]
                    # 判断任务在当前工作站的可选模式
                    flag = [0 for _ in range(3)]
                    if SW[station] != -1:
                        flag[0] = 1
                    if (TCtime[curtask][SC[station]] != -1) & (SC[station] != -1):
                        flag[1] = 1
                    if (TWCtime[curtask][SW[station] * NC + SC[station]] != -1) & (SC[station] != -1) & (
                            SW[station] != -1):
                        flag[2] = 1
                    # 判断当前模式是否可用,可用则不处理，不可用则调整
                    # 不可用
                    if flag[tempmindiv[ind].AM[curtask]] != 1:
                        # 当前有可用其他模式，则调整模式
                        if sum(flag) != 0:
                            # 随机选择一个模式赋值给当前任务
                            non_zero_indices = [index for index, value in enumerate(flag) if value != 0]
                            tempmindiv[ind].AM[curtask] = random.choice(non_zero_indices)
                        else:
                            # 否则，调整
                            print(f"任务{curtask}需要调整")
                            tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW, tempmindiv[ind].WTS, tempmindiv[ind].CTS = changemodestation(curtask, tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW)
        tempmindiv[ind].WTS, tempmindiv[ind].CTS = wcadjust(tempmindiv[ind].TA, tempmindiv[ind].AM, tempmindiv[ind].WTS, tempmindiv[ind].CTS)
        print(f"最终调整后:", tempmindiv[ind].TA, tempmindiv[ind].AM)
        print(f"各工作站工人分配为:{SW}")
        print(f"各工作站机器人分配为:{SC}")
        print(f"WTS,CTS:", tempmindiv[ind].WTS, tempmindiv[ind].CTS)
        print(f"个体{ind}结束===")
    tempfit = [[-1 for _ in range(objnum)] for _ in range(2)]
    orignfit = [[-1 for _ in range(objnum)] for _ in range(2)]
    orignfit[0][0], orignfit[0][1], orignfit[0][2], mindiv1.FT = decode(mindiv1.TA, mindiv1.AM, mindiv1.WTS, mindiv1.CTS)
    orignfit[1][0], orignfit[1][1], orignfit[1][2], mindiv2.FT = decode(mindiv2.TA, mindiv2.AM, mindiv2.WTS, mindiv2.CTS)
    for ind in range(2):
        # 计算新个体适应度值
        tempfit[ind][0], tempfit[ind][1], tempfit[ind][2], tempmindiv[ind].FT = decode(tempmindiv[ind].TA, tempmindiv[ind].AM,
                                                                   tempmindiv[ind].WTS, tempmindiv[ind].CTS)
    # orignfit[0][0], orignfit[0][1], mindiv1.FT = decode1(mindiv1.TA, mindiv1.AM, mindiv1.WTS, mindiv1.CTS)
    # orignfit[1][0], orignfit[1][1], mindiv2.FT = decode1(mindiv2.TA, mindiv2.AM, mindiv2.WTS, mindiv2.CTS)
    # for ind in range(2):
    #     # 计算新个体适应度值
    #     tempfit[ind][0], tempfit[ind][1], tempmindiv[ind].FT = decode1(tempmindiv[ind].TA, tempmindiv[ind].AM,
    #                                                                tempmindiv[ind].WTS, tempmindiv[ind].CTS)
        # 如果新解被旧解支配，则以一定概率不更新旧解
        if dominates(orignfit[ind], tempfit[ind]):
            randdata = random.random()
            if randdata >= math.exp(-t / T):
                if ind == 0:
                    tempmindiv[0] = mindiv1
                if ind == 1:
                    tempmindiv[1] = mindiv2
    return tempmindiv[0], tempmindiv[1]

def cobotcross(mindiv1, mindiv2, t,T):
    # 随机选择一个位置
    randloc = random.randint(0, NC - 1)
    tempmindiv = [indiv() for _ in range(2)]
    print("交叉前:", mindiv1.CTS, mindiv2.CTS)
    print(randloc)
    # 对于位置之前的直接放入
    for i in range(randloc):
        tempmindiv[0].CTS[i] = mindiv1.CTS[i]
        tempmindiv[1].CTS[i] = mindiv2.CTS[i]
    # 生成子代
    for i in range(Ntask):
        tempmindiv[0].TA[i] = mindiv1.TA[i]
        tempmindiv[1].TA[i] = mindiv2.TA[i]
        tempmindiv[0].AM[i] = mindiv1.AM[i]
        tempmindiv[1].AM[i] = mindiv2.AM[i]
    for i in range(NW):
        tempmindiv[0].WTS[i] = mindiv1.WTS[i]
        tempmindiv[1].WTS[i] = mindiv2.WTS[i]

    # 对于位置之后的
    for ind in range(2):
        print(f"子代{ind}============")
        flag = [0 for _ in range(NC + 1)]
        for i in range(randloc):
            #不是很懂flag标识的到底是什么？
            if tempmindiv[ind].CTS[i] != -1:
                try:
                    flag[tempmindiv[ind].CTS[i]] = 1
                except:
                    print(
                        f"IndexError: tempmindiv[{ind}].CTS[{i}] = {tempmindiv[ind].CTS[i]} is out of range for flag list of length {len(flag)}")
            else:
                flag[NC] = 1
        if ind == 0:
            ref = mindiv2
        else:
            ref = mindiv1
        k = randloc
        for i in range(NC):
            if (ref.CTS[i] == -1) and (k < NC):
                # 如果没有机器
                if (flag[NC] != 1):
                    tempmindiv[ind].CTS[k] = -1
                    flag[NC] = 1
                    k = k + 1

            else:
                try:
                    if (flag[ref.CTS[i]] != 1) & (k < NC):
                        tempmindiv[ind].CTS[k] = ref.CTS[i]
                        flag[ref.CTS[i]] = 1
                        k = k + 1
                except IndexError:
                    print(
                        f"IndexError: ref.CTS[{i}] = {ref.CTS[i]} is out of range for flag list of length {len(flag)}")
        print(f"交叉后,CTS={tempmindiv[ind].CTS}")

    # 计算实际分组
    for ind in range(2):
        print(f"子代{ind}============")
        PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
        PSnum = [0 for _ in range(NS * 2)]
        for i in range(Ntask):
            # 根据TA判断每个配对站的任务
            ps = tempmindiv[ind].TA[i]
            PSset[ps][PSnum[ps]] = i
            PSnum[ps] = PSnum[ps] + 1
        print(PSset)
        print(PSnum)

        # 机器和工人分配矩阵
        SW = [-1 for _ in range(NS * 2)]
        # 机器和工人分配矩阵
        SC = [-1 for _ in range(NS * 2)]
        for i in range(NW):
            if tempmindiv[ind].WTS[i] != -1:
                SW[tempmindiv[ind].WTS[i]] = i

        for i in range(NC):
            if tempmindiv[ind].CTS[i] != -1:
                SC[tempmindiv[ind].CTS[i]] = i

        print(f"各工作站工人分配为:{SW}")
        print(f"各工作站机器人分配为:{SC}")
        print(f"CTS={tempmindiv[ind].CTS}")
        flag = [0 for _ in range(NC)]
        for i in range(NC):
            # 代表机器人i已分配
            if tempmindiv[ind].CTS[i] != -1:
                flag[i] = 1
        print(f"flag={flag}")

        for station in range(NS * 2):
            if (SW[station] == -1) and (SC[station] == -1):
                # 从可选工人中随机选择一个
                zero_items = [index for index, value in enumerate(flag) if value == 0]
                if zero_items:
                    random_Cindex = random.choice(zero_items)
                    SC[station] = random_Cindex
                    tempmindiv[ind].CTS[random_Cindex] = station
                    flag[random_Cindex] = 1
                    print(f"由于{station}啥也没有，所以将机器人{random_Cindex}分给它")

        print(f"分完后各工作站机器人分配为:{SC},CTS={tempmindiv[ind].CTS}")
        print(f"各工作站工人分配为:{SW}")
        print(f"各工作站机器人分配为:{SC}")

        # 判断调整后，各个任务模式是否可用，是否需要调整; 优先内部调整，内部调整不了在换工作站
        for station in range(NS * 2):
            if (PSset[station][0] != -1):
                for j in range(PSnum[station]):
                    curtask = PSset[station][j]
                    # 判断任务在当前工作站的可选模式
                    flag = [0 for _ in range(3)]
                    if SW[station] != -1:
                        flag[0] = 1
                    if (TCtime[curtask][SC[station]] != -1) & (SC[station] != -1):
                        flag[1] = 1
                    if (TWCtime[curtask][SW[station] * NC + SC[station]] != -1) & (SC[station] != -1) & (
                            SW[station] != -1):
                        flag[2] = 1
                    # 判断当前模式是否可用,可用则不处理，不可用则调整
                    # 不可用
                    if flag[tempmindiv[ind].AM[curtask]] != 1:
                        # 当前有可用其他模式，则调整模式
                        if sum(flag) != 0:
                            # 随机选择一个模式赋值给当前任务
                            non_zero_indices = [index for index, value in enumerate(flag) if value != 0]
                            tempmindiv[ind].AM[curtask] = random.choice(non_zero_indices)
                        else:
                            # 否则，调整
                            print(f"任务{curtask}需要调整")
                            tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW, tempmindiv[ind].WTS, tempmindiv[ind].CTS = changemodestation(curtask, tempmindiv[ind].TA, tempmindiv[ind].AM, SC, SW)
        tempmindiv[ind].WTS, tempmindiv[ind].CTS = wcadjust(tempmindiv[ind].TA, tempmindiv[ind].AM, tempmindiv[ind].WTS, tempmindiv[ind].CTS)
        print(f"最终调整后:", tempmindiv[ind].TA, tempmindiv[ind].AM)
        print(f"各工作站工人分配为:{SW}")
        print(f"各工作站机器人分配为:{SC}")
        print(f"WTS,CTS:", tempmindiv[ind].WTS, tempmindiv[ind].CTS)
        print(f"个体{ind}结束===")
    tempfit = [[-1 for _ in range(objnum)] for _ in range(2)]
    orignfit = [[-1 for _ in range(objnum)] for _ in range(2)]
    orignfit[0][0], orignfit[0][1], orignfit[0][2], mindiv1.FT = decode(mindiv1.TA, mindiv1.AM, mindiv1.WTS, mindiv1.CTS)
    orignfit[1][0], orignfit[1][1], orignfit[1][2], mindiv2.FT = decode(mindiv2.TA, mindiv2.AM, mindiv2.WTS, mindiv2.CTS)
    for ind in range(2):
        # 计算新个体适应度值
        tempfit[ind][0], tempfit[ind][1], tempfit[ind][2], tempmindiv[ind].FT = decode(tempmindiv[ind].TA, tempmindiv[ind].AM,
                                                                   tempmindiv[ind].WTS, tempmindiv[ind].CTS)
    # orignfit[0][0], orignfit[0][1], mindiv1.FT = decode1(mindiv1.TA, mindiv1.AM, mindiv1.WTS, mindiv1.CTS)
    # orignfit[1][0], orignfit[1][1], mindiv2.FT = decode1(mindiv2.TA, mindiv2.AM, mindiv2.WTS, mindiv2.CTS)
    # for ind in range(2):
    #     # 计算新个体适应度值
    #     tempfit[ind][0], tempfit[ind][1], tempmindiv[ind].FT = decode1(tempmindiv[ind].TA, tempmindiv[ind].AM,
    #                                                                tempmindiv[ind].WTS, tempmindiv[ind].CTS)

        # 如果新解被旧解支配，则以一定概率不更新旧解
        if dominates(orignfit[ind], tempfit[ind]):
            randdata = random.random()
            if randdata >= math.exp(-t / T):
                if ind == 0:
                    tempmindiv[0] = mindiv1
                if ind == 1:
                    tempmindiv[1] = mindiv2
    return tempmindiv[0], tempmindiv[1]




