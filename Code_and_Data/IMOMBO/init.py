from numpy.distutils.system_info import dfftw_info
import numpy as np
from vector_for_general import *
# 放置分层：全局通用的内容
tasklevel = [[-1 for _ in range(Ntask)] for _ in range(Ntask)]
levelnum = [-1 for _ in range(Ntask)]

import random

# 根据前序关系进行分层
def sortlevel(tasklevel, curlevel):
    # 临时先序关系矩阵
    temppreor = [[0 for _ in range(Ntask)] for _ in range(Ntask)]
    for i in range(Ntask):
        for j in range(Ntask):
            temppreor[i][j] = preor[i][j]
    # 每个任务的前序任务数
    prenum = Ntask * [0]

    # 标记已经分层任务
    ftask = Ntask * [0]

    while (sum(ftask) < Ntask):
        # print(temppreor)
        # 根据temppreor统计每个任务的前序任务数
        prenum = [sum(row[i] == 1 for row in temppreor) for i in range(Ntask)]
        # 找到前序任务数为0的任务
        positions = [index for index, value in enumerate(prenum) if value == 0]
        # 放入对应层级
        k = 0
        for i in range(len(positions)):
            if ftask[positions[i]] == 0:
                tasklevel[curlevel][k] = positions[i]
                k = k + 1
                levelnum[curlevel] = k

        curlevel = curlevel + 1
        # 将temppreor对应行全设为0，对应任务标识设为1
        for i in range(len(positions)):
            for j in range(Ntask):
                temppreor[positions[i]][j] = 0
            ftask[positions[i]] = 1

    # print(prenum)

    return curlevel


curlevel = 0
Nlevel = sortlevel(tasklevel, curlevel)
print(f"任务共{Nlevel}层，分层为{tasklevel},每层任务数为{levelnum}")

# 按类似于NSGA-II非支配排序的思路将任务按照工作站数进行均分
def storeassign(MTA, Nlevel, mode):
    store = [[-1 for _ in range(Ntask)] for _ in range(NS)]
    tmplevel = [[-1 for _ in range(Ntask)] for _ in range(Ntask)]
    templevellnum=[-1 for _ in range(Ntask)]
    for i in range(Ntask):
        templevellnum[i]=levelnum[i]
    for i in range(Nlevel):
        for j in range(Ntask):
            tmplevel[i][j] = tasklevel[i][j]
    # 任务配对工作站分配
    TSA = [-1 for _ in range(Ntask)]
    if Ntask%2 == 0:
        avenum = int(Ntask / NS)
    else:
        avenum = int(Ntask / NS)+1

   # print(avenum)
    nsum = 0
    #当前工作站编号
    k = 0
    # 当前任务站中任务数
    curtn = 0
    i = 0
    while i < Nlevel:
        print(f"第{i}层，当前层任务为{tasklevel[i]}")
        print(f'(k+1)*avenum = {(k + 1) * avenum}')
        print(f'tasklevel[i] = {tasklevel[i]}')
        # 如果当前层全放入但是未满足个数需求或刚满足,则全放入
        print(f"第{i}层共{templevellnum[i]}个任务")
        if nsum + templevellnum[i] <= (k + 1) * avenum + 1:
            print(f"当前层可全部放入，放入前store{k}为{store[k]}")
            store[k][curtn:curtn + templevellnum[i]] = tmplevel[i][0:templevellnum[i]]
            for j in range(templevellnum[i]):
                TSA[tmplevel[i][j]] = k
            curtn = curtn + templevellnum[i]
            print(f"放入后store为{store[k]}")
            if nsum + templevellnum[i] == (k + 1) * avenum + 1:
                k = k + 1
                curtn = 0
            nsum = nsum + templevellnum[i]
            i = i + 1
        # 如果全放入超过个数要求，则随机选择部分放入，且使得i下一次还访问当前层
        else:
            # 随机选择任务
            index = [j for j in range(templevellnum[i])]
            numbers = random.sample(index, (k + 1) * avenum - nsum)
            print(f"第{i}层，随机选择的任务标号为{numbers}")
            # 将这些任务放入store中
            for j in range((k + 1) * avenum - nsum):
                print(f'numbers[j]={numbers[j]}')
                # store[k][nsum % avenum + j] = tmplevel[i][numbers[j]]
                store[k][curtn + j] = tmplevel[i][numbers[j]]
                print(f"选择任务{tmplevel[i][numbers[j]]}放入第{k}层，放入后store为{store[k]}")
                TSA[tmplevel[i][numbers[j]]] = k
            # 删除tasklevel[i]中的这些元素，修改templevellnum值
            for index in sorted(numbers, reverse=True):
                del tmplevel[i][index]
            # curtn = curtn + (k + 1) * avenum - nsum
            templevellnum[i] = templevellnum[i] - ((k + 1) * avenum - nsum)
            nsum = (k + 1) * avenum
            k = k + 1
            curtn = 0

    print(f'TSA = {TSA}')
    # 如果存在方位约束，则根据方位约束确认配对工作站
    for i in range(Ntask):
        if direction[i] <= 1:
            MTA[i] = TSA[i] * 2 + direction[i]

    # 随机模式
    if mode == 0:
        # 对于不存在方位约束的任务;随机生成方位
        for i in range(Ntask):
            if MTA[i] == -1:
                MTA[i] = TSA[i] * 2 + random.randint(0, 1)
    # 均衡模式
    else:
        # 统计配对站任务分配
        NPS = [0 for _ in range(NS)]
        # 统计每个配对工作站中未分配左右侧的任务索引
        UAindex = [[-1 for _ in range(Ntask)] for _ in range(NS)]
        UAnum = [0 for _ in range(NS)]
        # 每个工作站已经分配的任务数
        PSnum = [0 for _ in range(NS * 2)]
        for i in range(Ntask):
            NPS[TSA[i]] = NPS[TSA[i]] + 1
            if MTA[i] == -1:
                UAindex[TSA[i]][UAnum[TSA[i]]] = i
                UAnum[TSA[i]] = UAnum[TSA[i]] + 1
            else:
                PSnum[MTA[i]] = PSnum[MTA[i]] + 1
        for i in range(NS):
            # 如果左侧已超过1/2,则全部放入右侧
            if PSnum[i * 2] >= int(NPS[i] / 2):
                for j in range(UAnum[i]):
                    MTA[UAindex[i][j]] = i * 2 + 1
                    print(f'任务{UAindex[i][j]}放入第{i}层右侧，放入后store为{store[i]}',)

            else:
                # 如果右侧已超过1/2,则全部放入左侧
                if PSnum[i * 2 + 1] >= int(NPS[i] / 2):
                    for j in range(UAnum[i]):
                        MTA[UAindex[i][j]] = i * 2
                        print(f'任务{UAindex[i][j]}放入第{i}层左侧，放入后store为{store[i]}', )
                # 如果都没超过，则随机选择NPS[i]/2-PSnum[i*2]个任务放入左侧，剩余放入右侧
                else:
                    t1=0
                    t2=0
                    print('NPS[i]/1-PSnum[i*2]:',{NPS[i] / 2 - PSnum[i * 2]})
                    left_index = random.sample(UAindex[i][0:UAnum[i]], int(NPS[i] / 2 - PSnum[i * 2]))
                    print(left_index)
                    print(UAnum[i])
                    for j in range(UAnum[i]):
                        if UAindex[i][j] in left_index:
                            MTA[UAindex[i][j]] = i * 2
                            t1+=1
                        else:
                            MTA[UAindex[i][j]] = i * 2 + 1
                            t2+=1
                    print(f'两侧均没超过一半，有{t1}个任务放入左侧,{t2}个任务放入右侧')

    print(f'MTA = {MTA}')

# 模式设置
def ModeSelect(AM, TA):
    # 根据TA判断每个配对站的任务
    PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
    PSnum = [0 for _ in range(NS * 2)]
    for i in range(Ntask):
        ps = TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1
    # print(PSset)
    # print(PSnum)

    # 统计哪些工作站有任务
    non_zero_indices = [index for index, value in enumerate(PSnum) if value != 0]
    # 计算最大可合作工作站数量
    maxWCN = min(NW + NC - NS * 2, NS * 2, len(non_zero_indices))

    # 随机生成合作工作站数量
    if np.random.rand()<0.5:
        WCN = maxWCN
    else:
        WCN = random.randint(0, maxWCN)
    # 随机生成合作的工作站编号
    numbers = random.sample(non_zero_indices, WCN)
    # print(f"合作工作站为{numbers}")

    # 随机生成任务站安排次序
    index = [j for j in range(NS * 2)]
    random.shuffle(index)
    # print(index)

    rand = random.randint(1, 400)
    if rand < 100:
        ModeSelect1(AM, PSnum, PSset, index, numbers)
    elif rand < 200:
        ModeSelect2(AM, PSnum, PSset, index, numbers)
    elif rand<300:
        ModeSelect3(AM, PSnum, PSset, index, numbers)
    else:
        ModeSelect4(AM, PSnum, PSset, index, numbers)

    return numbers

# 按照次序生成模式；0工人模式，1机器人模式，2合作模式
#周期时间最短
def ModeSelect1(AM, PSnum, PSset, index, numbers):
    #AMS[PSset[station][j]][k] == 0:
    for i in range(NS * 2):
        station = index[i]
        if station in numbers:
            for j in range(PSnum[station]):
                tempt = PSset[station][j]
                if AMS[PSset[station][j]][2] != 0:
                    AM[tempt] = 2
                elif AMS[PSset[station][j]][1] != 0:
                    if np.random.rand() < 0.5:
                        AM[tempt] = 1
                    else:
                        AM[tempt] = 0
                else:
                    AM[tempt] = 0
        else:
            # 全部为同一模式
            # 判断当前工作站任务可选模式
            ms = [1 for _ in range(2)]
            for j in range(PSnum[station]):
                for k in range(2):
                    if AMS[PSset[station][j]][k] == 0:
                        ms[k] = 0
            # m = random.randint(0, 1)
            # while ms[m] == 0:
            #     m = random.randint(0, 1)
            # if ms[1]!=0:
            #     m = random.randint(0, 1)
            # else:
            m=0
            for j in range(PSnum[station]):
                tempt = PSset[station][j]
                AM[tempt] = m


#工效学风险最低
def ModeSelect2(AM, PSnum, PSset, index, numbers):
    print(f"模式选择2")
    for i in range(NS * 2):
        station = index[i]
        if station in numbers:
            for j in range(PSnum[station]):
                tempt = PSset[station][j]
                if AMS[PSset[station][j]][1] != 0:
                    AM[tempt] = 1
                elif AMS[PSset[station][j]][2] != 0:
                    # if np.random.rand()<0.5:
                    AM[tempt] = 2
                    # else:
                    #     AM[tempt] = 0
                else:
                    AM[tempt] = 0
        else:
            ms = [1 for _ in range(2)]
            for j in range(PSnum[station]):
                for k in range(2):
                    if AMS[PSset[station][j]][k] == 0:
                        ms[k] = 0
            # m = random.randint(0, 1)
            # while ms[m] != 0:
            #     m = random.randint(0, 1)
            if ms[1]!=0:
                m = 1
            else:
                m = 0
            for j in range(PSnum[station]):
                tempt = PSset[station][j]
                AM[tempt] = m


#能耗最小
def ModeSelect3(AM, PSnum, PSset, index, numbers):
    print(f"模式选择3")
    for i in range(NS * 2):
        station = index[i]
        if station in numbers:
            for j in range(PSnum[station]):
                tempt = PSset[station][j]
                if AMS[PSset[station][j]][2] != 0:
                    if np.random.rand()<0.5:
                        AM[tempt] = 2
                    else:
                        AM[tempt] = 0
                else:
                    AM[tempt] = 1
        else:
            ms = [1 for _ in range(2)]
            for j in range(PSnum[station]):
                for k in range(2):
                    if AMS[PSset[station][j]][k] == 0:
                        ms[k] = 0
            m = 0
            for j in range(PSnum[station]):
                tempt = PSset[station][j]
                AM[tempt] = m

def ModeSelect4(AM, PSnum, PSset, index, numbers):
    for i in range(NS * 2):
        station = index[i]
        # print(f"工作站{station}")
        # 当前任务模式选择(对于在合作工作站中的任务，至少有一个是为2)
        if station in numbers:
            # print(f"属于合作工作站,其中任务为{PSset[station]},共{PSnum[station]}个任务")
            # 确认进行合作加工的任务数
            WCtasknum = random.randint(1, PSnum[station])
            # 改成三种模式下的装配模式初始化

            # 将PSset随机打乱，选前WCtasknum为合作模式；其他任务随机生成模式
            random.shuffle(PSset[station][0:PSnum[station]])
            for j in range(PSnum[station]):
                tempt = PSset[station][j]
                if j < WCtasknum:
                    AM[tempt] = 2
                else:
                    AM[tempt] = random.randint(0, 1)
                # print(f"任务{tempt}模式为{AM[tempt]}")

        else:
            # 全部为同一模式
            # print(f"不属于合作工作站,其中任务为{PSset[station]},共{PSnum[station]}个任务")
            # 判断当前工作站任务可选模式
            ms = [1 for _ in range(2)]
            for j in range(PSnum[station]):
                for k in range(2):
                    if AMS[PSset[station][j]][k] == 0:
                        ms[k] = 0
                # m = random.randint(0, 1)
                # while ms[m] != 0:
                #     m = random.randint(0, 1)

            for j in range(PSnum[station]):
                tempt = PSset[station][j]
                if ms[1] == 0:
                    m = 0
                else:
                    m = random.randint(0, 1)
                AM[tempt] = m
                # print(f"任务{tempt}模式为{AM[tempt]}")
        # print(AM)
    return numbers


# 根据实际的分配调整工人和机器
def wcadjust(TA, AM, WTS, CTS):
    PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
    PSnum = [0 for _ in range(NS * 2)]
    # 对于所有工作站,计算PSset和PSnum:
    # 根据TA判断每个配对站的任务
    for i in range(Ntask):
        ps = TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1
    # print(PSset)
    # print(PSnum)

    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if WTS[i] != -1:
            SW[WTS[i]] = i
    for i in range(NC):
        if CTS[i] != -1:
            SC[CTS[i]] = i
    # 对于每个工作站：统计各模式数量
    for i in range(NS * 2):
        nworkmode = 0
        nwcmode = 0
        ncobotmode = 0
        for j in range(PSnum[i]):
            if AM[PSset[i][j]] == 0:
                nworkmode = nworkmode + 1
            if AM[PSset[i][j]] == 1:
                ncobotmode = ncobotmode + 1
            if AM[PSset[i][j]] == 2:
                nwcmode = nwcmode + 1
        # print(nworkmode,ncobotmode,nwcmode)
        # print(i,nworkmode+nwcmode,nwcmode+ncobotmode)
        if nworkmode + nwcmode == 0:
            SW[i] = -1
        if nwcmode + ncobotmode == 0:
            SC[i] = -1

    for i in range(NS * 2):
        if PSnum[i] == 0:
            SW[i] = -1
            SC[i] = -1
    # print(SC)
    # 根据新的SW和SC确认WTS和CTS
    WTS = [-1 for _ in range(NW)]
    for i in range(NS * 2):
        # 第i个工作站有工人SW[i]，则工人SW[i]被分配到工作站i
        if SW[i] != -1:
            WTS[SW[i]] = i

    CTS = [-1 for _ in range(NC)]
    for i in range(NS * 2):
        # 第i个工作站有工人SW[i]，则工人SW[i]被分配到工作站i
        if SC[i] != -1:
            CTS[SC[i]] = i
    return WTS, CTS


# 对于一个任务，如果当前模式/工作站无法满足要求，调整方法
def changemodestation(curtask, TA, AM, SC, SW):
    # 找该任务最早最晚可用配对工作站
    pre_indices = [index for index, row in enumerate(preor) if row[curtask] != 0]
    # print(f"前序任务为:", pre_indices)
    # 找到其后续任务集合
    dom_indices = [index for index, value in enumerate(preor[curtask][:]) if value != 0]
    # print(f"后序任务为:", dom_indices)
    # 根据前序找到最小可用任务站
    minstation = 0
    pairstation = [-1 for _ in range(Ntask)]
    for i in range(Ntask):
        pairstation[i] = int(TA[i] / 2)

    for j in pre_indices:
        minstation = max(int(TA[j] / 2), minstation)
    # 根据后序找到最小可用任务站
    maxstation = NS - 1
    for j in dom_indices:
        maxstation = min(int(TA[j] / 2), maxstation)

    # print(f"当前任务分配为:{pairstation}")
    # print(f"当前工作站机器分配为:{SW}")
    # print(f"当前工作站机器分配为:{SC}")
    # print(f"最小工作站为{minstation},最大工作站为{maxstation}")
    # 对所有可用任务站，判断哪些可以放入且符合机器/组合要求
    k1 = 0
    # 标记符合机器要求的工作站
    avacobotstation = [-1 for _ in range(NS * 2)]
    k2 = 0
    # 标记符合组合要求的工作站
    avawctstation = [-1 for _ in range(NS * 2)]
    # 标记符合工人要求的工作站
    avawokerstation = [-1 for _ in range(NS * 2)]
    k3 = 0

    for j in range(minstation, maxstation + 1):
        if direction[curtask] == 0:
            # 查看机器/组合要求
            if (TCtime[curtask][SC[j * 2]] != -1) & (SC[j * 2] != -1):
                avacobotstation[k1] = j * 2
                k1 = k1 + 1
            if (TWCtime[curtask][SW[j * 2] * NC + SC[j * 2]] != -1) & (SC[j * 2] != -1) & (SW[j * 2] != -1):
                avawctstation[k2] = j * 2
                # print(f"1.工作站{avawctstation[k2]}加入可用组合")
                k2 = k2 + 1
            if SW[j * 2] != -1:
                avawokerstation[k3] = j * 2
                k3 = k3 + 1
        else:
            if direction[curtask] == 1:
                if (TCtime[curtask][SC[j * 2 + 1]] != -1) & (SC[j * 2 + 1] != -1):
                    avacobotstation[k1] = j * 2 + 1
                    k1 = k1 + 1
                if (TWCtime[curtask][SW[j * 2 + 1] * NC + SC[j * 2 + 1]] != -1) & (SC[j * 2 + 1] != -1) & (
                        SW[j * 2 + 1] != -1):
                    avawctstation[k2] = j * 2 + 1
                    # print(f"2.工作站{avawctstation[k2]}加入可用组合")
                    k2 = k2 + 1
                if SW[j * 2 + 1] != -1:
                    avawokerstation[k3] = j * 2 + 1
                    k3 = k3 + 1
            else:
                for l in range(2):
                    if (TCtime[curtask][SC[j * 2 + l]] != -1) & (SC[j * 2 + l] != -1):
                        avacobotstation[k1] = j * 2 + l
                        k1 = k1 + 1
                    if (TWCtime[curtask][SW[j * 2 + l] * NC + SC[j * 2 + l]] != -1) & (SC[j * 2 + l] != -1) & (
                            SW[j * 2 + l] != -1):
                        avawctstation[k2] = j * 2 + l
                        # print(f"3.工作站{avawctstation[k2]}加入可用组合，工人为{SW[j*2+l]},机器为{SC[j*2+l]},时间为{TWCtime[i][SW[j*2+l]*NC+SC[j*2+l]]}")
                        k2 = k2 + 1
                    if SW[j * 2 + l] != -1:
                        avawokerstation[k3] = j * 2 + l
                        k3 = k3 + 1
    # print(f"可用组合工作站为:{avawctstation[0:k2]}")
    # print(f"可用机器工作站为:{avacobotstation[0:k1]}")
    # print(f"可用工人工作站为:{avawokerstation[0:k3]}")
    flag = 0  # 0代表没找到符合要求的工作站
    # 如果当前任务是机器模式
    if AM[curtask] == 1:
        if k1 != 0:
            TA[curtask] = random.choice(avacobotstation[0:k1])
            flag = 1
            # print(f"任务{curtask}被重新分到工作站{TA[curtask]},模式还是1")
        else:
            if k2 != 0:
                TA[curtask] = random.choice(avawctstation[0:k2])
                AM[curtask] = 2
                flag = 1
                # print(f"任务{curtask}被重新分到工作站{TA[curtask]}，模式变为2")
    if AM[curtask] == 2:
        if k2 != 0:
            TA[curtask] = random.choice(avawctstation[0:k2])
            flag = 1
            # print(f"任务{curtask}被重新分到工作站{TA[curtask]},模式还是2")
        else:
            if k1 != 0:
                TA[curtask] = random.choice(avacobotstation[0:k1])
                AM[curtask] = 1
                flag = 1
                # print(f"任务{curtask}被重新分到工作站{TA[curtask]}，模式变为1")

    # 如果都没找到，则将模式改为工人模式
    if flag == 0:
        # 如果有工人工作站
        if k3 != 0:
            AM[curtask] = 0
            TA[curtask] = random.choice(avawokerstation[0:k3])
            # print(f"任务{curtask}被重新分到工作站{TA[curtask]}，模式变为0")
        # 如果没有工人工作站,就创造一个工作站
        else:
            station = TA[curtask]
            if SW[station] == -1:
                fflag = [1 for _ in range(NW)]
                for ns in range(NS * 2):
                    if SW[ns] != -1:
                        fflag[SW[ns]] = 0
                # print(f"当前工人使用情况为{fflag}")
                # 从未被分配的工人中选择一个工人分配给当前工作站
                non_zero_items = [index for index, value in enumerate(fflag) if value != 0]
                random_Windex = random.choice(non_zero_items)

                SW[station] = random_Windex
                AM[curtask] = 0

                # print(f"任务{curtask}工作站{station}不变，模式变为0,将工人{random_Windex}分给它，当前SW={SW}")

    # 判断当前还有没有合作模式的任务，如果没有了，则当前工作站模式改为0(此时任务不需要调整)
    # 根据TA判断每个配对站的任务
    PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
    PSnum = [0 for _ in range(NS * 2)]

    for i in range(Ntask):
        ps = TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1

    WTS = [-1 for _ in range(NW)]
    for i in range(NS * 2):
        # 第i个工作站有工人SW[i]，则工人SW[i]被分配到工作站i
        if SW[i] != -1:
            WTS[SW[i]] = i

    CTS = [-1 for _ in range(NC)]
    for i in range(NS * 2):
        # 第i个工作站有工人SW[i]，则工人SW[i]被分配到工作站i
        if SC[i] != -1:
            CTS[SC[i]] = i

    return TA, AM, SC, SW, WTS, CTS


# 计算工人和机器人分配
def WCA(TA, AM, WTS, CTS, PSset, PSnum, conumbers):
    # 根据TA判断每个配对站的任务
    for i in range(Ntask):
        ps = TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1
    print(PSset)
    print(PSnum)
    print(AM)
    # 随机生成合作的工作站编号
    index = [j for j in range(NS * 2)]
    # 随机生成任务站安排次序
    random.shuffle(index)
    # 标记整个过程中已使用过的工人;1代表可用，0代表不可用
    TWF = [1 for _ in range(NW)]
    TCF = [1 for _ in range(NC)]

    # 标记每个工作站中的可用集合
    SWsetflag = [[0 for _ in range(NW)] for _ in range(NS * 2)]
    SCsetflag = [[0 for _ in range(NC)] for _ in range(NS * 2)]

    # 记录最终工作站中工人和机器人
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]

    print('合作工作站为：', conumbers)
    # 对于工作站
    for i in range(NS*2):
        station = index[i]
        # print(f"当前工作站为{station}==========")
        if PSnum[station] == 0:
            continue
        # 如果是合作工作站
        if station in conumbers:
            # 统计里面的合作任务
            cotask = [-1 for _ in range(PSnum[station])]
            conum = 0
            for j in range(PSnum[station]):
                # print(f"当前任务为{PSset[station][j]},加工方式为{AM[PSset[station][j]]}")
                if AM[PSset[station][j]] == 2:
                    cotask[conum] = PSset[station][j]
                    conum = conum + 1
            print(f"合作任务为", cotask, conum)
            # 统计合作任务可用合作方式集合
            coset = [1 for _ in range(NC * NW)]
            cosetnum = 0
            for k in range(NC * NW):
                tsum = 0
                for j in range(conum):
                    # print('cotask[j]',cotask[j])
                    # print('k+NC',k+NC)
                    print(f"任务{cotask[j]}在模式{k+NC}下加工可用性为:{TC[cotask[j]][k+NC]}")
                    tsum = tsum + TC[cotask[j]][k + NC]
                # 即，所有合作任务合集
                if tsum == conum:
                    coset[cosetnum] = k
                    cosetnum = cosetnum + 1
            print(f"合作合集为", coset)
            # 根据合作方式集合得到对应的工人和机器人集合
            Cflag = [0 for _ in range(NC)]
            Wflag = [0 for _ in range(NW)]

            for j in range(cosetnum):
                Cflag[coset[j] % NC] = 1
                Wflag[int(coset[j] / NW)] = 1

            # 对于选择工人模式的任务，不用统计并集
            # 对于选择机器人模式的任务，统计机器人并集
            for j in range(PSnum[station]):
                if AM[PSset[station][j]] == 1:
                    for k in range(NC):
                        if TC[PSset[station][j]][k] == 0:
                            Cflag[k] = 0
            for j in range(NW):
                SWsetflag[station][j] = Wflag[j]
            for j in range(NC):
                SCsetflag[station][j] = Cflag[j]

                # 整合任务可用以及整体可用
            for j in range(NW):
                Wflag[j] = int((Wflag[j] + TWF[j]) / 2)
            for j in range(NC):
                Cflag[j] = int((Cflag[j] + TCF[j]) / 2)

            print(f"当前工作站可用工人集合为{SWsetflag[station]},当前分配情况为{TWF},当前实际可用工人为{Wflag}")
            print(f"当前工作站可用机器人集合为{SCsetflag[station]},当前分配情况为{TCF},当前实际可用机器人为{Cflag}")

            # 如果实际无可选工人
            if sum(Wflag) == 0:
                # print("^实际无可用工人")
                # 从当前未分配工人中选择一个，后续由于合作/机器不可用造成的调整在后续会处理
                non_zero_items = [(j, value) for j, value in enumerate(TWF) if value != 0]
            else:
                # 从所有可用工人中选择一个工人
                non_zero_items = [(j, value) for j, value in enumerate(Wflag) if value != 0]
            random_Windex, _ = random.choice(non_zero_items)
            WTS[random_Windex] = station
            TWF[random_Windex] = 0
            SW[station] = random_Windex
            print(f"选择工人{random_Windex}放入工作站{station}")

            #统计该工人对应的可选机器人
            CCflag = [0 for _ in range(NC)]
            for j in range(cosetnum):
                if int(coset[j] / NW)==SW[station]:
                    CCflag[coset[j] % NC] = 1
            for j in range(NC):
                CCflag[j] = int((CCflag[j] + TCF[j]) / 2)
            # 对于选择机器人模式的任务，统计机器人并集
            for j in range(PSnum[station]):
                if AM[PSset[station][j]] == 1:
                    for k in range(NC):
                        if TC[PSset[station][j]][k] == 0:
                            CCflag[k] = 0
            print(f"工人{random_Windex}对应的机器人合集为:{CCflag}")
            # 如果实际无可选机器人
            if sum(CCflag) == 0:
                print("^实际无可用机器人")
                # 实际上也没有机器人空着了
                if sum(TCF) == 0:
                    # 当前工作站模式直接改成工人模式
                    for j in range(PSnum[station]):
                        AM[PSset[station][j]] = 0
                    print(f"当前工作站{station}改为工人模式")
                else:
                    # 从空着的机器人中随机选择一个机器人
                    non_zero_items = [index for index, value in enumerate(TCF) if value != 0]
                    randindex = random.choice(non_zero_items)
                    CTS[randindex] = station
                    TCF[randindex] = 0
                    SC[station] = randindex
                    k = 0
                    adjtask = [-1 for _ in range(PSnum[station])]
                    print(f"选择机器人{randindex}放入工作站{station}")
                    for kt in range(PSnum[station]):
                        print(f"任务{PSset[station][kt]}当前模式为{AM[PSset[station][kt]]}")
                        # 判断任务当前模式能否加工,能加工的就不处理
                        if AM[PSset[station][kt]] == 0:
                            continue
                        else:
                            if AM[PSset[station][kt]] == 1 & TC[PSset[station][kt]][randindex] == 1:
                                continue
                            else:
                                if (AM[PSset[station][kt]] == 2) & (
                                        TWCtime[PSset[station][kt]][SW[station] * NC + randindex] != -1):
                                    continue
                                else:
                                    print(f"任务{PSset[station][kt]}原有模式不可用")
                                    # 判断任务在当前工作站的可选模式
                                    ffflag = [0 for _ in range(3)]
                                    # 工人模式肯定可选
                                    if TWtime[PSset[station][kt]][SC[station]] != -1:
                                        ffflag[0] = 1
                                    if TCtime[PSset[station][kt]][SC[station]] != -1:
                                        ffflag[1] = 1
                                    if TWCtime[PSset[station][kt]][SW[station] * NC + SC[station]] != -1:
                                        ffflag[2] = 1
                                    # 随机选择一个模式赋值给当前任务
                                    non_zero_indices = [index for index, value in enumerate(ffflag) if value != 0]
                                    print(f"任务{PSset[station][kt]}在当前工作站可用模式为{ffflag}")
                                    if len(non_zero_indices) != 0:
                                        AM[PSset[station][kt]] = random.choice(non_zero_indices)
                                        print(f"任务{PSset[station][kt]}在当前工作站最终模式为{AM[PSset[station][kt]]}")
                                    else:
                                        adjtask[k] = PSset[station][kt]
                                        k = k + 1
                                        print(f"任务{PSset[station][kt]}此时无法在工作站{station}继续加工")

                    for kt in range(k):
                        print(f"对于任务{adjtask[kt]}进行调整")
                        TA, AM, SC, SW, WTS, CTS = changemodestation(adjtask[kt], TA, AM, SC, SW)
                        # 搞完以后TWF和TCF要改改
                        for fi in range(NW):
                            TWF[fi] = 1
                        for fi in range(NC):
                            TCF[fi] = 1
                        for fi in range(NW):
                            if SW[fi] != -1:
                                TWF[SW[fi]] = 0
                        for fi in range(NC):
                            if SC[fi] != -1:
                                TCF[SC[fi]] = 0

            else:
                # 从所有可用机器人中选择一个
                non_zero_items = [(j, value) for j, value in enumerate(CCflag) if value != 0]
                random_Cindex, _ = random.choice(non_zero_items)
                print(f"选择机器人{random_Cindex}放入工作站{station}")
                CTS[random_Cindex] = station
                TCF[random_Cindex] = 0
                SC[station] = random_Cindex
            print(f"当前各工人分配矩阵为{WTS},各机器人分配矩阵为{CTS}")
        # 如果是非合作工作站
        else:
            print(f"当前加工方式为{AM[PSset[station][0]]}")
            # 如果是工人模式,则全可用
            if AM[PSset[station][0]] == 0:
                Wflag = [1 for _ in range(NW)]
                for j in range(NW):
                    SWsetflag[station][j] = Wflag[j]
                for j in range(NW):
                    Wflag[j] = int((Wflag[j] + TWF[j]) / 2)
                # 从所有可用工人中选择一个工人
                non_zero_items = [(j, value) for j, value in enumerate(TWF) if value != 0]
                random_Windex, _ = random.choice(non_zero_items)
                print(f"当前工作站可用工人集合为{SWsetflag[station]},当前分配情况为{TWF},当前实际可用工人为{Wflag}")
                print(f"选择工人{random_Windex}放入工作站{station}")
                WTS[random_Windex] = station
                TWF[random_Windex] = 0
                SW[station] = random_Windex
            # 如果是机器人模式，则统计可用合集
            if AM[PSset[station][0]] == 1:
                Cflag = [1 for _ in range(NC)]
                for j in range(PSnum[station]):
                    for k in range(NC):
                        if TC[PSset[station][j]][k] == 0:
                            Cflag[k] = 0
                for j in range(NC):
                    SCsetflag[station][j] = Cflag[j]
                for j in range(NC):
                    Cflag[j] = int((Cflag[j] + TCF[j]) / 2)
                # 有可用机器
                if sum(Cflag) != 0:
                    non_zero_items = [(j, value) for j, value in enumerate(Cflag) if value != 0]
                    random_Cindex, _ = random.choice(non_zero_items)
                    print(
                        f"当前工作站可用机器人集合为{SCsetflag[station]},当前分配情况为{TCF},当前实际可用机器人为{Cflag}")
                    print(f"选择机器人{random_Cindex}放入工作站{station}")
                    CTS[random_Cindex] = station
                    TCF[random_Cindex] = 0
                    SC[station] = random_Cindex
                else:
                    print("^实际无可用机器人")
                    # 实际上也没有机器人空着了
                    if sum(TCF) == 0:
                        # 当前工作站模式直接改成工人模式
                        for j in range(PSnum[station]):
                            AM[PSset[station][j]] = 0
                        print(f"当前工作站{station}改为工人模式")
                        # 从当前可用工人中选择一个工人
                        non_zero_items = [(j, value) for j, value in enumerate(TWF) if value != 0]
                        random_Windex, _ = random.choice(non_zero_items)
                        WTS[random_Windex] = station
                        TWF[random_Windex] = 0
                        SW[station] = random_Windex

                    else:
                        # 从空着的机器人中随机选择一个机器人
                        non_zero_items = [index for index, value in enumerate(TCF) if value != 0]
                        randindex = random.choice(non_zero_items)
                        CTS[randindex] = station
                        TCF[randindex] = 0
                        SC[station] = randindex
                        k2 = 0
                        kt2 = 0
                        adjtask2 = [-1 for _ in range(PSnum[station])]
                        print(f"选择机器人{randindex}放入工作站{station}")
                        for kt2 in range(PSnum[station]):
                            # 如果不可加工
                            if TCtime[PSset[station][kt2]][SC[station]] == -1:
                                adjtask2[k2] = PSset[station][kt2]
                                k2 = k2 + 1
                                print(f"任务{PSset[station][kt2]}此时无法在工作站{station}继续加工")
                        for kt2 in range(k2):
                            print(PSset[station])
                            TA, AM, SC, SW, WTS, CTS = changemodestation(adjtask2[kt2], TA, AM, SC, SW)
                            # 搞完以后TWF和TCF要改改
                            for fi in range(NW):
                                TWF[fi] = 1
                            for fi in range(NC):
                                TCF[fi] = 1
                            for fi in range(NW):
                                if SW[fi] != -1:
                                    TWF[SW[fi]] = 0
                            for fi in range(NC):
                                if SC[fi] != -1:
                                    TCF[SC[fi]] = 0

            print(f"当前各工人分配矩阵为{WTS},各机器人分配矩阵为{CTS}")
    WTS, CTS = wcadjust(TA, AM, WTS, CTS)
    return TA, AM, WTS, CTS#计算工人和机器人分配


def decode(TA, AM, WTS, CTS):
    # 对于每个工人，定义单独工作时长worker time alone
    WTA = [0 for _ in range(NW)]
    # 对于每个工人，定义合作工作时长worker time together
    WTT = [0 for _ in range(NW)]
    # 对于每个机器人，定义单独工作时长
    CTA = [0 for _ in range(NC)]
    # 对于每个机器人，定义合作工作时长
    CTT = [0 for _ in range(NC)]
    # 任务带来的累积疲劳
    Fjk = [0 for _ in range(NS * 2)]
    # 每个任务的完成时间
    ctask = [0 for _ in range(Ntask)]
    # 根据TA判断每个配对站的任务
    PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
    PSnum = [0 for _ in range(NS * 2)]
    for i in range(Ntask):
        ps = TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1
    # print(PSset)
    # print(PSnum)

    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if WTS[i] != -1:
            SW[WTS[i]] = i
    for i in range(NC):
        if CTS[i] != -1:
            SC[CTS[i]] = i
    #print(f"各工作站工人分配为:{SW}")
    #print(f"各工作站机器人分配为:{SC}")

    # 获得每个任务对应分层
    taskl = [-1 for _ in range(Ntask)]
    for i in range(Nlevel):
        for j in range(levelnum[i]):
            taskl[tasklevel[i][j]] = i

    # 对于每个工作站，定义当前时间
    curtime = [0 for _ in range(NS * 2)]

    # 针对每个配对工作站
    for i in range(NS):
        #print(f"初步排序前第{i}个配对站，左侧为{PSset[i * 2]},右侧为{PSset[i * 2 + 1]}")
        # 按分层+序号排序
        left = PSset[i * 2]
        right = PSset[i * 2 + 1]
        # 将left先按序号排序
        left[:PSnum[i * 2]] = sorted(left[:PSnum[i * 2]])
        right[:PSnum[i * 2 + 1]] = sorted(right[:PSnum[i * 2 + 1]])

        for j in range(PSnum[i * 2]):
            for k in range(j, PSnum[i * 2]):
                # 按分层排
                if taskl[left[k]] < taskl[left[j]]:
                    tpdata = left[k]
                    left[k] = left[j]
                    left[j] = tpdata

        for j in range(PSnum[i * 2 + 1]):
            for k in range(j, PSnum[i * 2 + 1]):
                # 按分层排
                if taskl[right[k]] < taskl[right[j]]:
                    tpdata = right[k]
                    right[k] = right[j]
                    right[j] = tpdata
        print(f"初步排序后第{i}个配对站，左侧为{left},右侧为{right}")
        '''with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                truesort(left,right,PSnum[i*2],PSnum[i*2+1],AM,WTS,CTS)'''
        # 只有当两侧都有任务时才进行NEH排序；否则不需要排
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                if (PSnum[i * 2] != 0) and (PSnum[i * 2 + 1] != 0):
                    left,right=truesort(left, right, PSnum[i * 2], PSnum[i * 2 + 1], AM, WTS, CTS, i * 2 + 1)
        print(f"最终排序后第{i}个配对站，左侧为{left},右侧为{right}")
        for j in range(PSnum[i * 2]):
            PSset[i * 2][j] = left[j]
        for j in range(PSnum[i * 2 + 1]):
            PSset[i * 2 + 1][j] = right[j]
        # 根据配对站两边任务，计算该配对站两侧目标值相关内容
        point = -1  # 表示左侧
        k1 = 0
        k2 = 0
        #print("开始算目标值")

        while (k1 < PSnum[i * 2]) | (k2 < PSnum[i * 2 + 1]):
            #print(f"左侧已完成{k1}个任务，右侧已完成{k2}个任务")
            if k1 == PSnum[i * 2]:
                point = 1
            if k2 == PSnum[i * 2 + 1]:
                point = -1
            # 如果当前是左侧
            if point == -1:
                print(f"当前在左侧,当前工作站{i * 2}时间为{curtime[i * 2]}")
                # 判断当前任务在同配对站另一侧的前序任务是否已经加工(同侧肯定已经加工);顺便统计下最早可用时间
                avatime = curtime[i * 2]
                for j in range(PSnum[i * 2 + 1]):
                    if preor[PSset[i * 2 + 1][j]][PSset[i * 2][k1]] == 1:
                        if ctask[PSset[i * 2 + 1][j]] == 0:
                            print(f"任务{PSset[i * 2][k1]}存在先序任务在右侧未加工,该任务为任务{PSset[i * 2 + 1][j]}")
                            point = 1
                            break
                        else:
                            avatime = max(avatime, ctask[PSset[i * 2 + 1][j]])
                            print(f'完成先序任务后，当前工作站{i*2}的时间为{avatime}')
                if point == 1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2][k1]
                am = AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2]]
                    WTA[SW[i * 2]] = WTA[SW[i * 2]] + ptime
                    print(f"任务{task}选择工人模式，工人为{SW[i * 2]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2]]
                        CTA[SC[i * 2]] = CTA[SC[i * 2]] + ptime
                        print(f"任务{task}选择机器人模式，机器人为{SC[i * 2]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2] * NC + SC[i * 2]]
                        WTT[SW[i * 2]] = WTT[SW[i * 2]] + ptime
                        CTT[SC[i * 2]] = CTT[SC[i * 2]] + ptime
                        print(f"任务{task}选择合作模式，工人为{SW[i * 2]},机器人为{SC[i * 2]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2]]}，该机器人合作工作时间为{CTT[SC[i * 2]]}")
                curtime[i * 2] = avatime + ptime
                ctask[task] = curtime[i*2]
                k1 = k1 + 1
                if AM[task]!=1:
                    Fjk[i * 2] = Fjk[i * 2] + AT[task] * ptime
                print(f"加工完任务{task}后,当前时间为{curtime[i * 2]}")

            # 如果当前是右侧
            else:
                print(f"当前在右侧,当前工作站{i * 2 + 1}时间为{curtime[i * 2 + 1]}")
                # 判断当前任务在同配对站另一侧的前序任务是否已经加工(同侧肯定已经加工);顺便统计下最早可用时间
                avatime = curtime[i * 2 + 1]

                for j in range(PSnum[i * 2]):
                    if preor[PSset[i * 2][j]][PSset[i * 2 + 1][k2]] == 1:
                        if ctask[PSset[i * 2][j]] == 0:
                            print(f"任务{PSset[i * 2 + 1][k2]}存在先序任务在右侧未加工")
                            point = 1
                            break
                        else:
                            avatime = max(avatime, ctask[PSset[i * 2][j]])
                            print(f'完成先序任务{PSset[i * 2][j]}后，当前工作站{i * 2}的时间为{avatime}')
                if point == -1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2 + 1][k2]
                am = AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2 + 1]]
                    WTA[SW[i * 2 + 1]] = WTA[SW[i * 2 + 1]] + ptime
                    print(f"任务{task}选择工人模式，工人为{SW[i * 2 + 1]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2 + 1]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2 + 1]]
                        CTA[SC[i * 2 + 1]] = CTA[SC[i * 2 + 1]] + ptime
                        print( f"任务{task}选择机器人模式，机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2 + 1]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2 + 1] * NC + SC[i * 2 + 1]]
                        WTT[SW[i * 2 + 1]] = WTT[SW[i * 2 + 1]] + ptime
                        CTT[SC[i * 2 + 1]] = CTT[SC[i * 2 + 1]] + ptime
                        print(f"任务{task}选择合作模式，工人为{SW[i * 2 + 1]},机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2 + 1]]}，该机器人合作工作时间为{CTT[SC[i * 2 + 1]]}")
                curtime[i * 2 + 1] = avatime + ptime
                ctask[task] = curtime[i*2+1]
                k2 = k2 + 1
                if AM[task]!=1:
                    Fjk[i * 2 + 1] = Fjk[i * 2 + 1] + AT[task] * ptime
                print(f"加工完任务{task}后,当前时间为{curtime[i * 2 + 1]}")

    print('Fjk:',Fjk)
    # 最大完成时间目标
    cmax = curtime[0]
    for i in range(NS * 2):
        if curtime[i] > cmax:
            cmax = curtime[i]
        Fjk[i] = 1 - math.exp(-Fjk[i])

    #修改后目标函数计算
    #计算机器人能耗目标
    TCE = 0
    for i in range(NC):
        if CTS[i] != -1:
            TCE = TCE + CE[i] * (CTA[i] + CTT[i]) + CIE[i] * (cmax - CTA[i] - CTT[i])

    # 工人疲劳
    Rjk = [0 for _ in range(NS * 2)]
    for i in range(NS * 2):
        if SW[i] == -1:
            Rjk[i] = 0
        else:
            Rjk[i] = Fjk[i] * math.exp(-1 * BW[SW[i]] * (cmax - WTA[SW[i]] - WTT[SW[i]]))
    F3 = sum(Rjk)
    print('Rjk',Rjk)

    print("周期时间为:", cmax)
    print("工人工效学风险为:", F3)
    print("机器人能耗为:", TCE)



    print("关键中间变量值:")
    print("SC", SC)
    print("SW", SW)
    print("CTS:", CTS)
    print("WTS:", WTS)
    print("WTA:", WTA)
    print("WTT:", WTT)
    print("CTA:", CTA)
    print("CTT:", CTT)
    print("Fjk:", Fjk)
    print("curtime:", curtime)

    return cmax, F3, TCE, ctask

def decode1(TA, AM, WTS, CTS):
    # 对于每个工人，定义单独工作时长worker time alone
    WTA = [0 for _ in range(NW)]
    # 对于每个工人，定义合作工作时长worker time together
    WTT = [0 for _ in range(NW)]
    # 对于每个机器人，定义单独工作时长
    CTA = [0 for _ in range(NC)]
    # 对于每个机器人，定义合作工作时长
    CTT = [0 for _ in range(NC)]
    # 任务带来的累积疲劳
    Fjk = [0 for _ in range(NS * 2)]
    # 每个任务的完成时间
    ctask = [0 for _ in range(Ntask)]
    # 根据TA判断每个配对站的任务
    PSset = [[-1 for _ in range(Ntask)] for _ in range(NS * 2)]
    PSnum = [0 for _ in range(NS * 2)]
    for i in range(Ntask):
        ps = TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1
    # print(PSset)
    # print(PSnum)

    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if WTS[i] != -1:
            SW[WTS[i]] = i
    for i in range(NC):
        if CTS[i] != -1:
            SC[CTS[i]] = i
    #print(f"各工作站工人分配为:{SW}")
    #print(f"各工作站机器人分配为:{SC}")

    # 获得每个任务对应分层
    taskl = [-1 for _ in range(Ntask)]
    for i in range(Nlevel):
        for j in range(levelnum[i]):
            taskl[tasklevel[i][j]] = i

    # 对于每个工作站，定义当前时间
    curtime = [0 for _ in range(NS * 2)]

    # 针对每个配对工作站
    for i in range(NS):
        #print(f"初步排序前第{i}个配对站，左侧为{PSset[i * 2]},右侧为{PSset[i * 2 + 1]}")
        # 按分层+序号排序
        left = PSset[i * 2]
        right = PSset[i * 2 + 1]
        # 将left先按序号排序
        left[:PSnum[i * 2]] = sorted(left[:PSnum[i * 2]])
        right[:PSnum[i * 2 + 1]] = sorted(right[:PSnum[i * 2 + 1]])

        for j in range(PSnum[i * 2]):
            for k in range(j, PSnum[i * 2]):
                # 按分层排
                if taskl[left[k]] < taskl[left[j]]:
                    tpdata = left[k]
                    left[k] = left[j]
                    left[j] = tpdata

        for j in range(PSnum[i * 2 + 1]):
            for k in range(j, PSnum[i * 2 + 1]):
                # 按分层排
                if taskl[right[k]] < taskl[right[j]]:
                    tpdata = right[k]
                    right[k] = right[j]
                    right[j] = tpdata
        print(f"初步排序后第{i}个配对站，左侧为{left},右侧为{right}")
        '''with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                truesort(left,right,PSnum[i*2],PSnum[i*2+1],AM,WTS,CTS)'''
        # 只有当两侧都有任务时才进行NEH排序；否则不需要排
        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f):
                if (PSnum[i * 2] != 0) and (PSnum[i * 2 + 1] != 0):
                    left,right=truesort(left, right, PSnum[i * 2], PSnum[i * 2 + 1], AM, WTS, CTS, i * 2 + 1)
        print(f"最终排序后第{i}个配对站，左侧为{left},右侧为{right}")
        for j in range(PSnum[i * 2]):
            PSset[i * 2][j] = left[j]
        for j in range(PSnum[i * 2 + 1]):
            PSset[i * 2 + 1][j] = right[j]
        # 根据配对站两边任务，计算该配对站两侧目标值相关内容
        point = -1  # 表示左侧
        k1 = 0
        k2 = 0
        #print("开始算目标值")

        while (k1 < PSnum[i * 2]) | (k2 < PSnum[i * 2 + 1]):
            #print(f"左侧已完成{k1}个任务，右侧已完成{k2}个任务")
            if k1 == PSnum[i * 2]:
                point = 1
            if k2 == PSnum[i * 2 + 1]:
                point = -1
            # 如果当前是左侧
            if point == -1:
                print(f"当前在左侧,当前工作站{i * 2}时间为{curtime[i * 2]}")
                # 判断当前任务在同配对站另一侧的前序任务是否已经加工(同侧肯定已经加工);顺便统计下最早可用时间
                avatime = curtime[i * 2]
                for j in range(PSnum[i * 2 + 1]):
                    if preor[PSset[i * 2 + 1][j]][PSset[i * 2][k1]] == 1:
                        if ctask[PSset[i * 2 + 1][j]] == 0:
                            print(f"任务{PSset[i * 2][k1]}存在先序任务在右侧未加工,该任务为任务{PSset[i * 2 + 1][j]}")
                            point = 1
                            break
                        else:
                            avatime = max(avatime, ctask[PSset[i * 2 + 1][j]])
                            print(f'完成先序任务后，当前工作站{i*2}的时间为{avatime}')
                if point == 1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2][k1]
                am = AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2]]
                    WTA[SW[i * 2]] = WTA[SW[i * 2]] + ptime
                    print(f"任务{task}选择工人模式，工人为{SW[i * 2]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2]]
                        CTA[SC[i * 2]] = CTA[SC[i * 2]] + ptime
                        print(f"任务{task}选择机器人模式，机器人为{SC[i * 2]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2] * NC + SC[i * 2]]
                        WTT[SW[i * 2]] = WTT[SW[i * 2]] + ptime
                        CTT[SC[i * 2]] = CTT[SC[i * 2]] + ptime
                        print(f"任务{task}选择合作模式，工人为{SW[i * 2]},机器人为{SC[i * 2]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2]]}，该机器人合作工作时间为{CTT[SC[i * 2]]}")
                curtime[i * 2] = avatime + ptime
                ctask[task] = curtime[i*2]
                k1 = k1 + 1
                Fjk[i * 2] = Fjk[i * 2] + AT[task] * ptime
                print(f"加工完任务{task}后,当前时间为{curtime[i * 2]}")

            # 如果当前是右侧
            else:
                print(f"当前在右侧,当前工作站{i * 2 + 1}时间为{curtime[i * 2 + 1]}")
                # 判断当前任务在同配对站另一侧的前序任务是否已经加工(同侧肯定已经加工);顺便统计下最早可用时间
                avatime = curtime[i * 2 + 1]

                for j in range(PSnum[i * 2]):
                    if preor[PSset[i * 2][j]][PSset[i * 2 + 1][k2]] == 1:
                        if ctask[PSset[i * 2][j]] == 0:
                            print(f"任务{PSset[i * 2 + 1][k2]}存在先序任务在右侧未加工")
                            point = 1
                            break
                        else:
                            avatime = max(avatime, ctask[PSset[i * 2][j]])
                            print(f'完成先序任务{PSset[i * 2][j]}后，当前工作站{i * 2}的时间为{avatime}')
                if point == -1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2 + 1][k2]
                am = AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2 + 1]]
                    WTA[SW[i * 2 + 1]] = WTA[SW[i * 2 + 1]] + ptime
                    print(f"任务{task}选择工人模式，工人为{SW[i * 2 + 1]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2 + 1]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2 + 1]]
                        CTA[SC[i * 2 + 1]] = CTA[SC[i * 2 + 1]] + ptime
                        print( f"任务{task}选择机器人模式，机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2 + 1]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2 + 1] * NC + SC[i * 2 + 1]]
                        WTT[SW[i * 2 + 1]] = WTT[SW[i * 2 + 1]] + ptime
                        CTT[SC[i * 2 + 1]] = CTT[SC[i * 2 + 1]] + ptime
                        print(f"任务{task}选择合作模式，工人为{SW[i * 2 + 1]},机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2 + 1]]}，该机器人合作工作时间为{CTT[SC[i * 2 + 1]]}")
                curtime[i * 2 + 1] = avatime + ptime
                ctask[task] = curtime[i*2+1]
                k2 = k2 + 1
                Fjk[i * 2 + 1] = Fjk[i * 2 + 1] + AT[task] * ptime
                print(f"加工完任务{task}后,当前时间为{curtime[i * 2 + 1]}")


    # 最大完成时间目标
    cmax = curtime[0]
    for i in range(NS * 2):
        if curtime[i] > cmax:
            cmax = curtime[i]
        Fjk[i] = 1 - math.exp(-Fjk[i])



    #修改后目标函数计算
    # 计算机器人能耗目标
    # TCE = 0
    # for i in range(NC):
    #     if CTS[i] != -1:
    #         TCE = TCE + CE[i] * (CTA[i] + CTT[i]) + CIE[i] * (cmax - CTA[i] - CTT[i])

    # 工人疲劳
    Rjk = [0 for _ in range(NS * 2)]
    for i in range(NS * 2):
        if SW[i] == -1:
            Rjk[i] = 0
        else:
            Rjk[i] = Fjk[i] * math.exp(-1 * BW[SW[i]] * (cmax - WTA[SW[i]] - WTT[SW[i]]))
    F3 = sum(Rjk)

    print("周期时间为:", cmax)
    print("工人工效学风险为:", F3)
    # print("机器人能耗为:", TCE)

    print("关键中间变量值:")
    print("SC", SC)
    print("SW", SW)
    print("CTS:", CTS)
    print("WTS:", WTS)
    print("WTA:", WTA)
    print("WTT:", WTT)
    print("CTA:", CTA)
    print("CTT:", CTT)
    print("Fjk:", Fjk)
    print("curtime:", curtime)

    return cmax, F3, ctask




#非支配比较
def dominates(a, b):
    #检查解a是否支配解b
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


# v1,v2代表两个序列；n1,n2代表对应的数量，这里这么写是为了之后反过来排也能用
# 加一个逻辑：

def truesort(v1, v2, n1, n2, AM, WTS, CTS, orderstation):
    #找出v2中不存在先序任务的任务
    # no_pre_tasks = []
    # has_pre_tasks = []
    # v = v1+v2
    # for task in v2:
    #     if task!=-1:
    #         if all(preor[i][task] == 0 for i in v):
    #             no_pre_tasks.append(task)
    #         else:
    #             has_pre_tasks.append(task)

    # 将无先序任务的任务排列在队列最开始
    # v2 = no_pre_tasks + has_pre_tasks
    # n2 = len(v2)

    # 根据v1中任务次序，生成一个新的任务支配表，以用于确保左侧工作站内假设存在任务i在任务j前，则要求在右侧工作站中任务i的前序不能在任务j的后序之后
    domtask = [[-1 for _ in range(Ntask)] for _ in range(Ntask)]
    for i in range(n1):
        # 找到当前任务i的前序任务
        #真实所在行
        rrw = v1[i]
        pre_indices = [index for index, value in enumerate(row[rrw] for row in preor) if value != 0]
        for j in range(n1):
            if j != i:
                # 找到任务j的后续任务
                rrw1 = v1[j]
                dom_indices = [index for index, value in enumerate(preor[rrw1]) if value != 0]
                for k1 in pre_indices:
                    for k2 in dom_indices:
                        domtask[k1][k2] = 1
    print(f"防死锁关系矩阵为")
    for row in domtask:
        print(row)
    # 对于v2中任务，按次序，结合NEH，结合preor约束和domtask约束生成次序
    tempv2 = [-1 for _ in range(n2)]
    final2 = [-1 for _ in range(n2)]
    bestcmax = 10000000
    tempv2[0] = v2[0]
    final2[0] = v2[0]
    print("开始NEH===============")
    # 对于第i个任务，可选位置为0,1,..,i
    for i in range(1, n2):
        print("-------------i=", i)
        final2[i] = v2[i]
        # print("final2",final2)
        for j in range(n2):
            tempv2[j] = final2[j]
        # 判断是否符合两个约束
        flag = 0
        print(f"当前次序为{tempv2}")
        for j in range(i):
            # 后续任务在前
            if preor[tempv2[i]][tempv2[j]] == 1 or domtask[tempv2[i]][tempv2[j]] == 1:
                print("tempv2:",preor[tempv2[i]][tempv2[j]])
                print("domtsk:",domtask[tempv2[i]][tempv2[j]])
                print(f"根据支配关系，{tempv2[i]}不应在{tempv2[j]}前")
                flag = 1
                break

        if flag == 1:
            tempcmax = 1000000
        else:
            # print("tempv2",tempv2)
            #新增
            # for h in range(n1):
            #     if preor[v1[h]][tempv2[0]]==1:
            #         curtime_temp = calcmax(v1,h+1, AM, WTS, CTS, orderstation-1,0)
            #     else:
            #         curtime_temp = 0
            tempcmax = calcmax(tempv2, i + 1, AM, WTS, CTS, orderstation)
        # 交换&计算
        for j in range(i, 0, -1):
            # print("final2",final2)
            tp = tempv2[j]
            tempv2[j] = tempv2[j - 1]
            tempv2[j - 1] = tp
            flag = 0
            print(f"当前次序为{tempv2}")
            for k in range(j):
                # 后续任务在前 6 7   7 8
                if preor[tempv2[j]][tempv2[k]] == 1 or domtask[tempv2[j]][tempv2[k]] == 1:
                    print(f"根据支配关系，{tempv2[k]}不应在{tempv2[j]}前")
                    flag = 1
                    break
            for k in range(j + 1, i):
                # 后续任务在前
                if preor[tempv2[k]][tempv2[j]] == 1 | domtask[tempv2[k]][tempv2[j]] == 1:
                    print(f"根据支配关系，{tempv2[j]}不应在{tempv2[k]}前")
                    flag = 1
                    break
            if flag == 1:
                continue
            # for h in range(n1):
            #     if preor[v1[h]][tempv2[0]]==1:
            #         curtime_temp = calcmax(v1,h+1, AM, WTS, CTS, orderstation-1,0)
            #     else:
            #         curtime_temp = 0
            t = calcmax(tempv2, i + 1, AM, WTS, CTS, orderstation)
            if t < tempcmax:
                print("更新")
                tempcmax = t
                for k in range(n2):
                    final2[k] = tempv2[k]
                print("final2", final2)

    for j in range(n2):
        v2[j] = final2[j]
    return v1,v2


def calcmax(order,n,AM,WTS,CTS,orderstation):
    print(f"次序为{order[0:n]}")
    #机器和工人分配矩阵
    curtime=0
    SW=[-1 for _ in range(NS*2)]
    SC=[-1 for _ in range(NS*2)]
    for i in range(NW):
        if WTS[i]!=-1:
            SW[WTS[i]]=i
    for i in range(NC):
        if CTS[i]!=-1:
            SC[CTS[i]]=i

    #计算时间时出错了
    for j in range(n):
        task=order[j]
        am=AM[task]
        ptime=0#处理时长
        #工人模式
        if am==0:
            ptime=TWtime[task][SW[orderstation]]
            print(f"任务{task}选择工人模式，工人为{SW[orderstation]},处理时间为{ptime}")
        #机器人模式
        else:
            if am==1:
                ptime=TCtime[task][SC[orderstation]]
                print(f"任务{task}选择机器人模式，机器人为{SC[orderstation]},处理时间为{ptime}")
            #合作模式
            else:
                ptime=TWCtime[task][SW[orderstation]*NC+SC[orderstation]]
                print(f"任务{task}选择合作模式，工人为{SW[orderstation]},机器人为{SC[orderstation]},处理时间为{ptime}")
        curtime=curtime+ptime
    print(f"次序{order[0:n]}完成时间为{curtime}")
    return curtime

def pregene(mode):
    TA=[-1 for _ in range(Ntask)]
    TSA2=[-1 for _ in range(Ntask)]
    #根据Nlevel和tasklevel随机生成配对站
    for i in range(Nlevel):
        for j in range(levelnum[i]):
            #当前任务
            task=tasklevel[i][j]
            #先找当前任务的先序任务最大的工作站索引
            maxprestation=0
            for k in range(Ntask):
                #是先序任务
                if preor[k][task]==1:
                    maxprestation=max(maxprestation,TSA2[k])
                    #print(f"任务{k}是任务{task}的前序，所以任务{task}至少从{TSA2[k]}开始")
            #生成最大工作站索引-NS之间的随机数
            TSA2[task]=random.randint(maxprestation,NS-1)
            #print(f"任务{task}分到工作站{TSA2[task]},当前所有任务分配为{TSA2}")
    print(f"配对站安排:{TSA2}")
    TA=TAtoTSA(TSA2,mode)
    return TA


# 根据配对站生成工作站
def TAtoTSA(TSA2, mode):
    MTA = [-1 for _ in range(Ntask)]
    # 如果存在方位约束，则根据方位约束确认配对工作站
    for i in range(Ntask):
        if direction[i] <= 1:
            MTA[i] = TSA2[i] * 2 + direction[i]

    # 随机模式
    if mode == 0:
        # 对于不存在方位约束的任务;随机生成方位
        for i in range(Ntask):
            if MTA[i] == -1:
                MTA[i] = TSA2[i] * 2 + random.randint(0, 1)
    # 均衡模式
    else:
        # 统计配对站任务分配
        NPS = [0 for _ in range(NS)]
        # 统计每个配对工作站中未分配左右侧的任务索引
        UAindex = [[-1 for _ in range(Ntask)] for _ in range(NS)]
        UAnum = [0 for _ in range(NS)]
        # 每个工作站已经分配的任务数
        PSnum = [0 for _ in range(NS * 2)]
        for i in range(Ntask):
            NPS[TSA2[i]] = NPS[TSA2[i]] + 1
            if MTA[i] == -1:
                UAindex[TSA2[i]][UAnum[TSA2[i]]] = i
                UAnum[TSA2[i]] = UAnum[TSA2[i]] + 1
            else:
                PSnum[MTA[i]] = PSnum[MTA[i]] + 1
        for i in range(NS):
            # 如果左侧已超过1/2,则全部放入右侧
            if PSnum[i * 2] >= int(NPS[i] / 2):
                for j in range(UAnum[i]):
                    MTA[UAindex[i][j]] = i * 2 + 1

            else:
                # 如果右侧已超过1/2,则全部放入左侧
                if PSnum[i * 2 + 1] >= int(NPS[i] / 2):
                    for j in range(UAnum[i]):
                        MTA[UAindex[i][j]] = i * 2
                # 如果都没超过，则随机选择NPS[i]/2-PSnum[i*2]个任务放入左侧，剩余放入右侧
                else:
                    left_index = random.sample(UAindex[i][0:UAnum[i]], int(NPS[i] / 2 - PSnum[i * 2]))
                    for j in range(UAnum[i]):
                        if j in left_index:
                            MTA[UAindex[i][j]] = i * 2
                        else:
                            MTA[UAindex[i][j]] = i * 2 + 1

    return MTA


# 对一个随机序列TA进行调整
def adjust(TSA, mode):

    #print(TSA)
    #print(tasklevel)
    # print(Nlevel)
    # print(levelnum)
    # 从所有分层中随机选择一层
    randlevel = random.randint(0, Nlevel - 1)
    # print(randlevel)
    # 对该层进行扰动
    tasklevel[randlevel][:levelnum[randlevel]] = random.sample(tasklevel[randlevel][:levelnum[randlevel]],levelnum[randlevel])
    # 随机选择一个位置
    randloc = random.randint(0, levelnum[randlevel] - 1)
    # print(f"随机层={randlevel}; 扰动后:{tasklevel};随机位置={randloc}")
    # 统计该位置及之前层的任务
    before = [-1 for _ in range(Ntask)]
    after = [-1 for _ in range(Ntask)]
    k1 = 0
    k2 = 0
    for i in range(Nlevel):
        if i < randlevel:
            for j in range(levelnum[i]):
                before[k1] = tasklevel[i][j]
                k1 = k1 + 1

        else:
            if i == randlevel:
                for j in range(levelnum[i]):
                    if j <= randloc:
                        before[k1] = tasklevel[i][j]
                        k1 = k1 + 1
                    else:
                        after[k2] = tasklevel[i][j]
                        k2 = k2 + 1
            else:
                for j in range(levelnum[i]):
                    after[k2] = tasklevel[i][j]
                    k2 = k2 + 1
    # print(f"随机位置={randloc},before={before},after={after}")
    # 对于之前的任务；从前向后调整
    for i in range(randlevel + 1):
        if i < randlevel:
            for j in range(levelnum[i]):
                # 当前任务
                task = tasklevel[i][j]
                for k in before:
                    # task是k的先序任务且k在
                    if preor[task][k] == 1:
                        # print(TSA[task],TSA[k])
                        TSA[task] = min(TSA[task], TSA[k])
        else:
            for j in range(randloc + 1):
                # 当前任务
                task = tasklevel[i][j]
                for k in before:
                    # task是k的先序任务且k在
                    if preor[task][k] == 1:
                        TSA[task] = min(TSA[task], TSA[k])

    # print(f"调整后={TSA}")
    # 对于之后的任务；从后向前调整
    for i in range(Nlevel - 1, randlevel - 1, -1):
        if i > randlevel:
            for j in range(levelnum[i]):
                # 当前任务
                task = tasklevel[i][j]
                for k in after:
                    # task是k的后序任务且k在after中
                    if preor[k][task] == 1:
                        TSA[task] = max(TSA[task], TSA[k])
        else:
            for j in range(randloc + 1):
                # 当前任务
                task = tasklevel[i][j]
                for k in after:
                    # task是k的后序任务
                    if preor[k][task] == 1:
                        TSA[task] = max(TSA[task], TSA[k])
                        # print(f"调整后={TSA}")
    # 最后再从前向后调整一遍
    for i in range(Nlevel):
        for j in range(levelnum[i]):
            # 当前任务
            task = tasklevel[i][j]
            for k in range(Ntask):
                # task是k的先序任务且k在
                if preor[task][k] == 1:
                    # print(TSA[task],TSA[k])
                    TSA[task] = min(TSA[task], TSA[k])

    # print(f"调整后={TSA}")
    TA = [-1 for _ in range(Ntask)]
    TA = TAtoTSA(TSA, mode)
    # print(TA)
    return TSA, TA

# for i in range(100):
#     mindiv= [-1 for _ in range(Ntask)]
#     storeassign(mindiv, Nlevel, 0)
#     print(mindiv)
#     print()