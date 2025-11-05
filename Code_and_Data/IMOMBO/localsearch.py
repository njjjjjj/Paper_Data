from popinit import *

tasklevel = [[-1 for _ in range(Ntask)] for _ in range(Ntask)]
levelnum = [-1 for _ in range(Ntask)]
curlevel = 0
Nlevel = sortlevel(tasklevel, curlevel)

def TAop(mindiv):
    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if mindiv.WTS[i] != -1:
            SW[mindiv.WTS[i]] = i
    for i in range(NC):
        if mindiv.CTS[i] != -1:
            SC[mindiv.CTS[i]] = i
    # 随机选择一个任务
    rtask = random.randint(0, Ntask - 1)

    print(f"任务站安排:{mindiv.TA}")
    print(f"模式选择:{mindiv.AM}")
    # 找到其前序任务集合
    pre_indices = [index for index, row in enumerate(preor) if row[rtask] != 0]
    print(pre_indices)
    # 找到其后续任务集合
    dom_indices = [index for index, value in enumerate(preor[rtask][:]) if value != 0]
    print(dom_indices)
    # 根据前序找到最小可用任务站
    minstation = 0
    for i in pre_indices:
        minstation = max(int(mindiv.TA[i] / 2), minstation)
    # 根据后序找到最小可用任务站
    maxstation = NS - 1
    for i in dom_indices:
        maxstation = min(int(mindiv.TA[i] / 2), maxstation)
    print(f"任务{rtask}的最小最大工作站为:{minstation},{maxstation}")
    # 随机生成新的工作站
    newstation = 0
    if minstation == maxstation:
        newstation = mindiv.TA[rtask]  # 不换工作站，只随机更换模式
    elif minstation > maxstation:
        print('报个错，min>max')
    else:
        newpairstation = random.randint(minstation, maxstation)
        if (direction[rtask] == 0) | (direction[rtask] == 1):
            newstation = newpairstation * 2 + direction[rtask]
        else:
            newstation = newpairstation * 2 + random.randint(0, 1)
    print(f"新的工作站为:{newstation}")
    # 当前模式
    am = mindiv.AM[rtask]
    # 新工作站是工人模式
    if (SW[newstation] != -1) and (SC[newstation] == -1):
        print(f"任务{rtask}在{newstation}加工，模式更改为0，当前模式工人为{SW[newstation]}")
        # 更改任务模式和工作站
        mindiv.AM[rtask] = 0
        mindiv.TA[rtask] = newstation
    # 如果新工作在是机器人模式且能加工当前任务
    else:
        if (SW[newstation] == -1) and (SC[newstation] != -1) and (TCtime[rtask][SC[newstation]] != -1):
            mindiv.AM[rtask] = 1
            mindiv.TA[rtask] = newstation
            print(f"任务{rtask}在{newstation}加工，模式更改为1，当前模式机器人为{SC[newstation]}")
        # 如果新工作站是合作模式
        else:
            if (SW[newstation] != -1) and (SC[newstation] != -1):
                # 判断任务在当前工作站的可选模式
                flag = [0 for _ in range(3)]
                # 工人模式肯定可选
                flag[0] = 1
                if TWCtime[rtask][SW[newstation] * NC + SC[newstation]] != -1:
                    flag[2] = 1
                    mindiv.AM[rtask]=2
                if TCtime[rtask][SC[newstation]] != -1:
                    flag[1] = 1
                    mindiv.AM[rtask] = random.randint(0, 1)
                #     # 随机选择一个模式赋值给当前任务
                # non_zero_indices = [index for index, value in enumerate(flag) if value != 0]
                # mindiv.AM[rtask] = random.choice(non_zero_indices)
                mindiv.TA[rtask] = newstation
                print(
                    f"任务{rtask}在{newstation}加工，模式更改为{mindiv.AM[rtask]}，当前模式工人为{SW[newstation]}，机器人为{SC[newstation]}")
    # 判断之前工作站工人/机器是否需要去掉
    print(f"TA:{mindiv.TA}")
    print(f"AM:{mindiv.AM}")

    mindiv.WTS, mindiv.CTS = wcadjust(mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS)

    print(f"调整后任务站安排:{mindiv.TA}")
    print(f"模式选择:{mindiv.AM}")
    return mindiv

def TAchange(mindiv, changetask):
    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if mindiv.WTS[i] != -1:
            SW[mindiv.WTS[i]] = i
    for i in range(NC):
        if mindiv.CTS[i] != -1:
            SC[mindiv.CTS[i]] = i
    # 随机选择一个任务
    rtask = changetask

    print(f"任务站安排:{mindiv.TA}")
    print(f"模式选择:{mindiv.AM}")
    # 找到其前序任务集合
    pre_indices = [index for index, row in enumerate(preor) if row[rtask] != 0]
    print(pre_indices)
    # 找到其后续任务集合
    dom_indices = [index for index, value in enumerate(preor[rtask][:]) if value != 0]
    print(dom_indices)
    # 根据前序找到最小可用任务站
    minstation = 0
    for i in pre_indices:
        minstation = max(int(mindiv.TA[i] / 2), minstation)
    # 根据后序找到最小可用任务站
    maxstation = NS - 1
    for i in dom_indices:
        maxstation = min(int(mindiv.TA[i] / 2), maxstation)
    print(f"任务{rtask}的最小最大工作站为:{minstation},{maxstation}")
    # 随机生成新的工作站
    newstation = 0
    if minstation == maxstation:
        newstation = mindiv.TA[rtask]  # 不换工作站，只随机更换模式
    elif minstation > maxstation:
        print('报个错，min>max')
    else:
        newpairstation = random.randint(minstation, maxstation)
        if (direction[rtask] == 0) | (direction[rtask] == 1):
            newstation = newpairstation * 2 + direction[rtask]
        else:
            newstation = newpairstation * 2 + random.randint(0, 1)
    print(f"新的工作站为:{newstation}")
    # 当前模式
    am = mindiv.AM[rtask]
    # 新工作站是工人模式
    if (SW[newstation] != -1) & (SC[newstation] == -1):
        print(f"任务{rtask}在{newstation}加工，模式更改为0，当前模式工人为{SW[newstation]}")
        # 更改任务模式和工作站
        mindiv.AM[rtask] = 0
        mindiv.TA[rtask] = newstation
    # 如果新工作在是机器人模式且能加工当前任务
    else:
        if (SW[newstation] == -1) & (SC[newstation] != -1) & (TCtime[rtask][SC[newstation]] != -1):
            mindiv.AM[rtask] = 1
            mindiv.TA[rtask] = newstation
            print(f"任务{rtask}在{newstation}加工，模式更改为1，当前模式机器人为{SC[newstation]}")
        # 如果新工作站是合作模式
        else:
            if (SW[newstation] != -1) & (SC[newstation] != -1):
                # 判断任务在当前工作站的可选模式
                flag = [0 for _ in range(3)]
                # 工人模式肯定可选
                flag[0] = 1
                if TCtime[rtask][SC[newstation]] != -1:
                    flag[1] = 1
                if TWCtime[rtask][SW[newstation] * NC + SC[newstation]] != -1:
                    flag[2] = 1
                # 随机选择一个模式赋值给当前任务
                non_zero_indices = [index for index, value in enumerate(flag) if value != 0]
                mindiv.AM[rtask] = random.choice(non_zero_indices)
                mindiv.TA[rtask] = newstation
                print(
                    f"任务{rtask}在{newstation}加工，模式更改为{mindiv.AM[rtask]}，当前模式工人为{SW[newstation]}，机器人为{SC[newstation]}")
    # 判断之前工作站工人/机器是否需要去掉
    print(f"TA:{mindiv.TA}")
    print(f"AM:{mindiv.AM}")

    mindiv.WTS, mindiv.CTS = wcadjust(mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS)

    print(f"调整后任务站安排:{mindiv.TA}")
    print(f"模式选择:{mindiv.AM}")

    return mindiv

#一个工作站必须分配一名工人（未实现）

def AMop(mindiv):
    print(f"任务站安排:{mindiv.TA}")
    print(f"模式选择:{mindiv.AM}")
    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if mindiv.WTS[i] != -1:
            SW[mindiv.WTS[i]] = i
    for i in range(NC):
        if mindiv.CTS[i] != -1:
            SC[mindiv.CTS[i]] = i
    # 随机选择一个任务
    rtask = random.randint(0, Ntask - 1)
    station = mindiv.TA[rtask]
    print(SW)
    print(SC)
    # 判断任务在当前工作站的可选模式
    flag = [0 for _ in range(3)]
    #工作站中不包含工人
    #任务的可选模式为空
    if SW[station] != -1:
        flag[0] = 1
    if (SC[station] != -1) and (TCtime[rtask][SC[station]] != -1):
        flag[1] = 1
    if (SW[station] != -1) and (SC[station] != -1) and (TWCtime[rtask][SW[station] * NC + SC[station]] != -1):
        flag[2] = 1
    # 随机选择一个模式赋值给当前任务
    print(f"任务{rtask}在工作站{station}的可选模式为{flag}")
    non_zero_indices = [index for index, value in enumerate(flag) if value != 0]
    print('----------------')
    print(f"任务{rtask}的可选模式为{non_zero_indices}")
    #修改这里的随机生成可选模式
    if non_zero_indices:
        mindiv.AM[rtask] = random.choice(non_zero_indices)

    # 根据现有模式调整当前工作站工人/机器人
    mindiv.WTS, mindiv.CTS = wcadjust(mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS)

    print(f"任务站安排:{mindiv.TA}")
    print(f"模式选择:{mindiv.AM}")
    return mindiv

def WTSop(mindiv):
    print(f"TA:{mindiv.TA}")
    print(f"AM:{mindiv.AM}")
    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if mindiv.WTS[i] != -1:
            SW[mindiv.WTS[i]] = i
    for i in range(NC):
        if mindiv.CTS[i] != -1:
            SC[mindiv.CTS[i]] = i
    print(f"SW:{SW}")
    print(f"SC:{SC}")
    # 随机选择两个工作站
    station1 = random.randint(0, NS * 2 - 1)
    station2 = random.randint(0, NS * 2 - 1)

    while station2 == station1:
        station2 = random.randint(0, NS * 2 - 1)
    print("选择的工作站为:", station1, station2)
    # 如果两个工作站都有工人，则交换
    if (SW[station1] != -1) & (SW[station2] != -1):
        mindiv.WTS[SW[station1]] = station2
        mindiv.WTS[SW[station2]] = station1
        tp = SW[station1]
        SW[station1] = SW[station2]
        SW[station2] = tp

        print(f"交换后:")
        print(f"SW:{SW}")
        print(f"SC:{SC}")

    else:
        # 如果只有一个工作站有工人，判断是否有工人未分配工作站；如果存在，则交换；否则，不交换
        changestation = -1
        if (SW[station1] != -1) & (SW[station2] == -1):
            changestation = station1
        if (SW[station1] == -1) & (SW[station2] != -1):
            changestation = station2
        if (SW[station1] == -1) & (SW[station2] == -1):
            changestation = -1

        if changestation != -1:
            # 看现在是否有其他工人未分配
            indexes = [index for index, value in enumerate(mindiv.WTS) if value == -1]
            print("存在一个工作站无工人,SW={SW}，当前除工作站{changestation}所有工人外，可用工人为{indexes}")
            if len(indexes) == 1:
                # 当前的工人为SW[changestation];将其WTS设为-1
                mindiv.WTS[SW[changestation]] = -1
                SW[changestation] = indexes[0]
                mindiv.WTS[SW[changestation]] = changestation
                print(f"选择工人{indexes[0]}放入工作站{changestation}中，当前SW={SW}")
            else:
                if len(indexes) != 0:
                    # 当前的工人为SW[changestation];将其WTS设为-1
                    mindiv.WTS[SW[changestation]] = -1
                    # 从中随机选择
                    SW[changestation] = random.choice(indexes)
                    mindiv.WTS[SW[changestation]] = changestation
                    print(f"选择工人{SW[changestation]}放入工作站{changestation}中，SW={SW}")
    # 对发生改变的工作站；判断里面的任务模式是否需要修改
    if SW[station1] != -1:
        for i in range(Ntask):
            if (mindiv.TA[i] == station1) & (mindiv.AM[i] == 2):
                # 如果是工人模式，则不需要修改；如果是合作模式，需要判断当前工人机器组合是否可用；如果不可用，则改为工人模式
                if (TWCtime[i][SW[station1] * NC + SC[station1]] == -1):
                    mindiv.AM[i] = 0

    if SW[station2] != -1:
        for i in range(Ntask):
            if (mindiv.TA[i] == station2) & (mindiv.AM[i] == 2):
                # 如果是工人模式，则不需要修改；如果是合作模式，需要判断当前工人机器组合是否可用；如果不可用，则改为工人模式
                if (TWCtime[i][SW[station2] * NC + SC[station2]] == -1):
                    mindiv.AM[i] = 0

    mindiv.WTS, mindiv.CTS = wcadjust(mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS)

    print(f"各工作站工人分配为:{SW}")
    print(f"各工作站机器人分配为:{SC}")
    print(f"模式选择:{mindiv.AM}")
    return mindiv

def CTSop(mindiv):
    print(f"任务站安排:{mindiv.TA}")
    print(f"模式选择:{mindiv.AM}")
    print(f"机器:{mindiv.CTS}")
    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if mindiv.WTS[i] != -1:
            SW[mindiv.WTS[i]] = i
    for i in range(NC):
        if mindiv.CTS[i] != -1:
            SC[mindiv.CTS[i]] = i
    print(f"各工作站工人分配为:{SW}")
    print(f"各工作站机器人分配为:{SC}")
    # 随机选择两个工作站
    station1 = random.randint(0, NS * 2 - 1)
    station2 = random.randint(0, NS * 2 - 1)
    print("选择的工作站为:", station1, station2)
    while station2 == station1:
        station2 = random.randint(0, NS * 2 - 1)
    # 如果两个工作站都有机器人，则交换
    if (SC[station1] != -1) & (SC[station2] != -1) & (station1 != station2):
        mindiv.CTS[SC[station1]] = station2
        mindiv.CTS[SC[station2]] = station1
        tp = SC[station1]
        SC[station1] = SC[station2]
        SC[station2] = tp

    # 如果只有一个工作站有机器人，判断是否有机器人未分配工作站；如果存在，则交换；否则，不交换
    changestation = -1
    if (SC[station1] != -1) & (SC[station2] == -1):
        changestation = station1
    if (SC[station1] == -1) & (SC[station2] != -1):
        changestation = station2
    if station1 == station2:
        changestation = station2

    if (SC[station1] == -1) & (SC[station2] == -1):
        changestation = -1

    if changestation != -1:
        # 看现在是否有其他机器人未分配
        indexes = [index for index, value in enumerate(mindiv.CTS[0:NC]) if value == -1]
        print(f"未选机器:", indexes)
        if len(indexes) != 0:
            # 当前的机器人为SC[changestation];将其CTS设为-1
            mindiv.CTS[SC[changestation]] = -1
            # 从中随机选择
            SC[changestation] = random.choice(indexes)
            mindiv.CTS[SC[changestation]] = changestation
        else:
            return mindiv
    print(f"各工作站工人分配为:{SW}")
    print(f"各工作站机器人分配为:{SC}")
    print(f"模式选择:{mindiv.AM}")

    adjtask = [-1 for _ in range(Ntask)]
    k = 0
    # 对发生改变的工作站；判断里面的任务模式是否需要修改
    if SC[station1] != -1:
        for i in range(Ntask):
            if mindiv.TA[i] == station1:
                # 如果是合作模式，需要判断当前工人机器组合是否可用；如果不可用，看机器模式是否可用；
                '''print(f"当前{station1}可用工人为{SW[station1]},可用机器为{SC[station1]},对应序号为{SW[station1]*NC+SC[station1]}，i={i}")
                print(mindiv.AM[i])
                print(SW[station1])
                print(TWCtime[i][SW[station1]*NC+SC[station1]])'''
                if (mindiv.AM[i] == 2) & (SW[station1] != -1) & (TWCtime[i][SW[station1] * NC + SC[station1]] == -1):
                    if TCtime[i][SC[station1]] != -1:
                        mindiv.AM[i] == 1
                    else:
                        mindiv.AM[i] == 0

                # 如果是机器模式，看合作模式是否可用
                if (mindiv.AM[i] == 1) & (TCtime[i][SC[station1]] == -1):
                    if (TWCtime[i][SW[station1] * NC + SC[station1]] != -1) & (SW[station1] != -1):
                        mindiv.AM[i] == 2
                    else:
                        if SW[station1] != -1:
                            mindiv.AM[i] == 0
                        else:
                            adjtask[k] = i
                            k = k + 1
                            print(f"任务{i}此时无法在工作站{station1}继续加工")

    if SC[station2] != -1:
        for i in range(Ntask):
            if mindiv.TA[i] == station2:
                '''print(f"当前{station2}可用工人为{SW[station2]},可用机器为{SC[station2]},对应序号为{SW[station2]*NC+SC[station2]}，i={i}")
                print(mindiv.AM[i])
                print(SW[station2])
                print(TWCtime[i][SW[station2]*NC+SC[station2]])'''
                # 如果是合作模式，需要判断当前工人机器组合是否可用；如果不可用，看机器模式是否可用；
                if (mindiv.AM[i] == 2) & (SW[station2] != -1) & (TWCtime[i][SW[station2] * NC + SC[station2]] == -1):
                    if TCtime[i][SC[station2]] != -1:
                        mindiv.AM[i] == 1
                    else:
                        adjtask[k] = i
                        k = k + 1
                        print(f"任务{i}此时无法在工作站{station2}继续加工")
                # 如果是机器模式，看合作模式是否可用
                if (mindiv.AM[i] == 1) & (TCtime[i][SC[station2]] == -1):
                    if (TWCtime[i][SW[station2] * NC + SC[station2]] != -1) & (SW[station2] != -1):
                        mindiv.AM[i] == 2
                    else:
                        if SW[station2] != -1:
                            mindiv.AM[i] == 0
                        else:
                            adjtask[k] = i
                            k = k + 1
                            print(f"任务{i}此时无法在工作站{station2}继续加工")

    print(f"以下任务需要换工作站:", adjtask)
    print(f"当前机器分配SC为:{SC}")
    # 对于需要进一步调整的任务
    for i in range(k):
        curtask = adjtask[i]
        mindiv.TA, mindiv.AM, SC, SW, mindiv.WTS, mindiv.CTS = changemodestation(curtask, mindiv.TA, mindiv.AM, SC, SW)

    mindiv.WTS, mindiv.CTS = wcadjust(mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS)

    print(f"各工作站工人分配为:{SW}")
    print(f"各工作站机器人分配为:{SC}")
    print(f"任务站安排:{mindiv.TA}")
    print(f"模式选择:{mindiv.AM}")
    return mindiv

def critical_cmax(mindiv):
    # 解码，计算当前所有工作站完成时间
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
        ps = mindiv.TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1
    print(PSset)
    print(PSnum)

    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if mindiv.WTS[i] != -1:
            SW[mindiv.WTS[i]] = i
    for i in range(NC):
        if mindiv.CTS[i] != -1:
            SC[mindiv.CTS[i]] = i
    print(f"各工作站工人分配为:{SW}")
    print(f"各工作站机器人分配为:{SC}")

    # 获得每个任务对应分层
    taskl = [-1 for _ in range(Ntask)]
    for i in range(Nlevel):
        for j in range(levelnum[i]):
            taskl[tasklevel[i][j]] = i

    # 对于每个工作站，定义当前时间
    curtime = [0 for _ in range(NS * 2)]

    # 针对每个配对工作站
    for i in range(NS):
        print(f"初步排序前第{i}个配对站，左侧为{PSset[i * 2]},右侧为{PSset[i * 2 + 1]}")
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
        # 只有当两侧都有任务时才进行NEH排序；否则不同排
        if (PSnum[i * 2] != 0) & (PSnum[i * 2 + 1] != 0):
            truesort(left, right, PSnum[i * 2], PSnum[i * 2 + 1], mindiv.AM, mindiv.WTS, mindiv.CTS, i * 2 + 1)
        print(f"最终排序后第{i}个配对站，左侧为{left},右侧为{right}")
        for j in range(PSnum[i * 2]):
            PSset[i * 2][j] = left[j]
        for j in range(PSnum[i * 2 + 1]):
            PSset[i * 2 + 1][j] = right[j]
        # 根据配对站两边任务，计算该配对站两侧目标值相关内容
        point = -1  # 表示左侧
        k1 = 0
        k2 = 0
        print("开始算目标值")
        # 确认当前任务们的处理时间
        Ptime = [0 for _ in range(Ntask)]
        while (k1 < PSnum[i * 2]) | (k2 < PSnum[i * 2 + 1]):
            print(f"左侧已完成{k1}个任务，右侧已完成{k2}个任务")
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
                if point == 1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2][k1]
                am = mindiv.AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2]]
                    Ptime[task] = ptime
                    WTA[SW[i * 2]] = WTA[SW[i * 2]] + ptime
                    print(
                        f"任务{task}选择工人模式，工人为{SW[i * 2]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2]]
                        Ptime[task] = ptime
                        CTA[SC[i * 2]] = CTA[SC[i * 2]] + ptime
                        print(
                            f"任务{task}选择机器人模式，机器人为{SC[i * 2]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2] * NC + SC[i * 2]]
                        Ptime[task] = ptime
                        WTT[SW[i * 2]] = WTT[SW[i * 2]] + ptime
                        CTT[SC[i * 2]] = CTT[SC[i * 2]] + ptime
                        print(
                            f"任务{task}选择合作模式，工人为{SW[i * 2]},机器人为{SC[i * 2]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2]]}，该机器人合作工作时间为{CTT[SC[i * 2]]}")
                curtime[i * 2] = curtime[i * 2] + ptime
                ctask[task] = curtime[i * 2]
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
                if point == -1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2 + 1][k2]
                am = mindiv.AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2 + 1]]
                    Ptime[task] = ptime
                    WTA[SW[i * 2 + 1]] = WTA[SW[i * 2 + 1]] + ptime
                    print(
                        f"任务{task}选择工人模式，工人为{SW[i * 2 + 1]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2 + 1]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2 + 1]]
                        Ptime[task] = ptime
                        CTA[SC[i * 2 + 1]] = CTA[SC[i * 2 + 1]] + ptime
                        print(
                            f"任务{task}选择机器人模式，机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2 + 1]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2 + 1] * NC + SC[i * 2 + 1]]
                        Ptime[task] = ptime
                        WTT[SW[i * 2 + 1]] = WTT[SW[i * 2 + 1]] + ptime
                        CTT[SC[i * 2 + 1]] = CTT[SC[i * 2 + 1]] + ptime
                        print(
                            f"任务{task}选择合作模式，工人为{SW[i * 2 + 1]},机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2 + 1]]}，该机器人合作工作时间为{CTT[SC[i * 2 + 1]]}")
                curtime[i * 2 + 1] = curtime[i * 2 + 1] + ptime
                ctask[task] = curtime[i * 2 + 1]
                k2 = k2 + 1
                Fjk[i * 2 + 1] = Fjk[i * 2 + 1] + AT[task] * ptime
                print(f"加工完任务{task}后,当前时间为{curtime[i * 2 + 1]}")

    # 关键工作站索引
    cmaxindex = 0
    for i in range(NS * 2):
        if curtime[i] > curtime[cmaxindex]:
            cmaxindex = i

    # 对关键工厂中的任务按Ptime从大到小排序
    for i in range(PSnum[cmaxindex]):
        for j in range(i + 1, PSnum[cmaxindex]):
            if Ptime[PSset[cmaxindex][i]] < Ptime[PSset[cmaxindex][j]]:
                tp = Ptime[PSset[cmaxindex][i]]
                Ptime[PSset[cmaxindex][i]] = Ptime[PSset[cmaxindex][j]]
                Ptime[PSset[cmaxindex][j]] = tp

    # 用来标记是否有任务可被插入到其他工厂
    flag = 0
    for i in range(PSnum[cmaxindex]):
        if flag == 1:
            break
        else:
            # 当前要试图插入的任务
            curtask = PSset[cmaxindex][i]
            print("待判断任务为:", curtask)
            # 找到这个任务所有可插入的工厂
            # 找该任务最早最晚可用配对工作站
            pre_indices = [index for index, row in enumerate(preor) if row[curtask] != 0]
            print(f"前序任务为:", pre_indices)
            # 找到其后续任务集合
            dom_indices = [index for index, value in enumerate(preor[curtask][:]) if value != 0]
            print(f"后序任务为:", dom_indices)
            # 根据前序找到最小可用任务站
            minstation = 0
            pairstation = [-1 for _ in range(Ntask)]
            for i in range(Ntask):
                pairstation[i] = int(mindiv.TA[i] / 2)

            for j in pre_indices:
                minstation = max(int(mindiv.TA[j] / 2), minstation)
            # 根据后序找到最小可用任务站
            maxstation = NS - 1
            for j in dom_indices:
                maxstation = min(int(mindiv.TA[j] / 2), maxstation)

            print(f"当前任务分配为:{pairstation}")
            print(f"当前工作站机器分配为:{SW}")
            print(f"当前工作站机器分配为:{SC}")
            print(f"最小工作站为{minstation},最大工作站为{maxstation}")
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
            print(f"可用组合工作站为:{avawctstation[0:k2]}")
            print(f"可用机器工作站为:{avacobotstation[0:k1]}")
            print(f"可用工人工作站为:{avawokerstation[0:k3]}")
            # 如果无可插入工作站,看下一个任务咋样
            if (k1 + k2 + k3) == 0:
                continue
            else:
                # 确认所有可插入索引
                insertindex = avawctstation[0:k2] + avacobotstation[0:k1] + avawokerstation[0:k3]
                # 只要一个工作站，且是当前工作站，则继续下一个任务
                if (len(insertindex) == 1) & (insertindex[0] == mindiv.TA[curtask]):
                    continue
                # 找到完成时间最小的工作站插入,更改模式
                else:
                    insertindex = [x for _, x in sorted(zip(curtime, insertindex), key=lambda pair: pair[0])]
                    mindiv.TA[curtask] = insertindex[0]
                    if insertindex[0] in avawokerstation:
                        mindiv.AM[curtask] = 0
                    else:
                        if insertindex[0] in avacobotstation:
                            mindiv.AM[curtask] = 1
                        else:
                            mindiv.AM[curtask] = 2

                    # 调整工人/机器人分配
                    mindiv.WTS, mindiv.CTS = wcadjust(mindiv.TA, mindiv.AM, mindiv.WTS, mindiv.CTS)
                    break
    return mindiv

def critical_worker(mindiv):
    # 解码，找到最疲劳的工厂
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
        ps = mindiv.TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1
    print(PSset)
    print(PSnum)

    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if mindiv.WTS[i] != -1:
            SW[mindiv.WTS[i]] = i
    for i in range(NC):
        if mindiv.CTS[i] != -1:
            SC[mindiv.CTS[i]] = i
    print(f"各工作站工人分配为:{SW}")
    print(f"各工作站机器人分配为:{SC}")

    # 获得每个任务对应分层
    taskl = [-1 for _ in range(Ntask)]
    for i in range(Nlevel):
        for j in range(levelnum[i]):
            taskl[tasklevel[i][j]] = i

    # 对于每个工作站，定义当前时间
    curtime = [0 for _ in range(NS * 2)]
    taskstart = [0 for _ in range(Ntask)]

    # 针对每个配对工作站
    for i in range(NS):
        print(f"初步排序前第{i}个配对站，左侧为{PSset[i * 2]},右侧为{PSset[i * 2 + 1]}")
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
        # 只有当两侧都有任务时才进行NEH排序；否则不同排
        if (PSnum[i * 2] != 0) & (PSnum[i * 2 + 1] != 0):
            truesort(left, right, PSnum[i * 2], PSnum[i * 2 + 1], mindiv.AM, mindiv.WTS, mindiv.CTS, i * 2 + 1)
        print(f"最终排序后第{i}个配对站，左侧为{left},右侧为{right}")
        for j in range(PSnum[i * 2]):
            PSset[i * 2][j] = left[j]
        for j in range(PSnum[i * 2 + 1]):
            PSset[i * 2 + 1][j] = right[j]
        # 根据配对站两边任务，计算该配对站两侧目标值相关内容
        point = -1  # 表示左侧
        k1 = 0
        k2 = 0
        print("开始算目标值")

        while (k1 < PSnum[i * 2]) | (k2 < PSnum[i * 2 + 1]):
            print(f"左侧已完成{k1}个任务，右侧已完成{k2}个任务")
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
                if point == 1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2][k1]
                am = mindiv.AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2]]
                    WTA[SW[i * 2]] = WTA[SW[i * 2]] + ptime
                    print(
                        f"任务{task}选择工人模式，工人为{SW[i * 2]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2]]
                        CTA[SC[i * 2]] = CTA[SC[i * 2]] + ptime
                        print(
                            f"任务{task}选择机器人模式，机器人为{SC[i * 2]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2] * NC + SC[i * 2]]
                        WTT[SW[i * 2]] = WTT[SW[i * 2]] + ptime
                        CTT[SC[i * 2]] = CTT[SC[i * 2]] + ptime
                        print(
                            f"任务{task}选择合作模式，工人为{SW[i * 2]},机器人为{SC[i * 2]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2]]}，该机器人合作工作时间为{CTT[SC[i * 2]]}")
                curtime[i * 2] = curtime[i * 2] + ptime
                ctask[task] = curtime[i * 2]
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
                if point == -1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2 + 1][k2]
                am = mindiv.AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2 + 1]]
                    WTA[SW[i * 2 + 1]] = WTA[SW[i * 2 + 1]] + ptime
                    print(
                        f"任务{task}选择工人模式，工人为{SW[i * 2 + 1]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2 + 1]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2 + 1]]
                        CTA[SC[i * 2 + 1]] = CTA[SC[i * 2 + 1]] + ptime
                        print(
                            f"任务{task}选择机器人模式，机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2 + 1]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2 + 1] * NC + SC[i * 2 + 1]]
                        WTT[SW[i * 2 + 1]] = WTT[SW[i * 2 + 1]] + ptime
                        CTT[SC[i * 2 + 1]] = CTT[SC[i * 2 + 1]] + ptime
                        print(
                            f"任务{task}选择合作模式，工人为{SW[i * 2 + 1]},机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2 + 1]]}，该机器人合作工作时间为{CTT[SC[i * 2 + 1]]}")
                curtime[i * 2 + 1] = curtime[i * 2 + 1] + ptime
                ctask[task] = curtime[i * 2 + 1]
                k2 = k2 + 1
                Fjk[i * 2 + 1] = Fjk[i * 2 + 1] + AT[task] * ptime
                print(f"加工完任务{task}后,当前时间为{curtime[i * 2 + 1]}")

    # 最大完成时间目标
    cmax = curtime[0]
    for i in range(NS * 2):
        if curtime[i] > cmax:
            cmax = curtime[i]
        Fjk[i] = 1 - math.exp(-Fjk[i])

    # 计算机器人能耗目标
    TCE = 0
    for i in range(NC):
        TCE = TCE + CE[i] * (CTA[i] + CTT[i]) + CIE[i] * (cmax - CTA[i] - CTT[i])

    # 工人疲劳
    Rjk = [0 for _ in range(NS * 2)]
    for i in range(NS * 2):
        if SW[i] == -1:
            Rjk[i] = 0
        else:
            Rjk[i] = Fjk[i] * math.exp(-1 * BW[SW[i]] * (cmax - WTA[i] - WTT[i]))

    # 关键工作站索引
    Rmaxindex = 0
    for i in range(NS * 2):
        if Rjk[i] > Rjk[Rmaxindex]:
            Rmaxindex = i

    # 如果当前工厂无机器人，则从可用机器人中分一个给它
    if (SC[Rmaxindex] == -1):
        # 看现在是否有其他机器人未分配
        indexes = [index for index, value in enumerate(mindiv.CTS) if value == -1]
        if len(indexes) == 1:
            SC[Rmaxindex] = indexes[0]
            mindiv.CTS[SC[Rmaxindex]] = Rmaxindex
        else:
            if len(indexes) != 0:
                # 从中随机选择
                SC[Rmaxindex] = random.choice(indexes)
                mindiv.CTS[SC[Rmaxindex]] = Rmaxindex

    # 对于当前工作站中所有任务，判断其开始时间+合作模式处理时间<所有同配对工作站紧后任务的开始时间，则更改
    changenum = 0  # 标记被修改的任务数
    # 找到当前工人任务索引
    workerindex = [-1 for _ in range(PSnum[Rmaxindex])]
    workertasknum = 0

    for j in range(PSnum[Rmaxindex]):
        curtask = PSset[Rmaxindex][j]
        if mindiv.AM[curtask] == 0:
            workerindex[workertasknum] = curtask
            workertasknum = workertasknum + 1
            # 找到其后续任务集合
            dom_indices = [index for index, value in enumerate(preor[curtask][:]) if value != 0]
            samestationdom = [-1 for _ in range(len(dom_indices))]
            k = 0
            # 找到同配对工作站后续任务集合
            for ttask in dom_indices:
                if (int(mindiv.TA[ttask] / 2) == int(mindiv.TA[curtask] / 2)):
                    samestationdom[k] = ttask
                    k = k + 1
                    # 如果无后续任务，可随意修改
            if k == 0:
                # 若可修改为合作模式，则修改为合作模式
                if (TWCtime[curtask][SW[Rmaxindex] * NC + SC[Rmaxindex]] != -1):
                    mindiv.AM[curtask] = 2
                    changenum = changenum + 1
                else:
                    # 若能修改为机器模式，则修改为机器模式
                    if (TCtime[curtask][SC[Rmaxindex]] != -1):
                        mindiv.AM[curtask] = 1
                        changenum = changenum + 1
            else:
                # 若可修改为合作模式，则修改为合作模式(人机协同的加工效率则高于工人加工，其加工时间短于工人加工,所以不用判断)
                if (TWCtime[curtask][SW[Rmaxindex] * NC + SC[Rmaxindex]] != -1):
                    mindiv.AM[curtask] = 2
                    changenum = changenum + 1
                else:
                    # 找出所有后续中最早的开始时间
                    earstart = taskstart[samestationdom[0]]
                    for i in samestationdom:
                        if i != -1:
                            if taskstart[i] < earstart:
                                earstart = taskstart[i]

                    # 若能修改为机器模式，则修改为机器模式
                    if (TCtime[curtask][SC[Rmaxindex]] != -1) & (
                            taskstart[curtask] + TCtime[curtask][SC[Rmaxindex]] <= earstart):
                        mindiv.AM[curtask] = 1
                        changenum = changenum + 1
    # 没有任务发生更改，随机选择其中一个工人任务插入到可以加工的工作站内
    if (changenum == 0) & (workertasknum != 0):
        changetask = random.choice(workerindex[0:workertasknum])
        TAchange(mindiv, changetask)

    return mindiv

def critical_cobot(mindiv):
    # 解码，找到最疲劳的工人
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
        ps = mindiv.TA[i]
        PSset[ps][PSnum[ps]] = i
        PSnum[ps] = PSnum[ps] + 1
    print(PSset)
    print(PSnum)

    # 机器和工人分配矩阵
    SW = [-1 for _ in range(NS * 2)]
    SC = [-1 for _ in range(NS * 2)]
    for i in range(NW):
        if mindiv.WTS[i] != -1:
            SW[mindiv.WTS[i]] = i
    for i in range(NC):
        if mindiv.CTS[i] != -1:
            SC[mindiv.CTS[i]] = i
    print(f"各工作站工人分配为:{SW}")
    print(f"各工作站机器人分配为:{SC}")

    # 获得每个任务对应分层
    taskl = [-1 for _ in range(Ntask)]
    for i in range(Nlevel):
        for j in range(levelnum[i]):
            taskl[tasklevel[i][j]] = i

    # 对于每个工作站，定义当前时间
    curtime = [0 for _ in range(NS * 2)]
    taskstart = [0 for _ in range(Ntask)]

    # 针对每个配对工作站
    for i in range(NS):
        print(f"初步排序前第{i}个配对站，左侧为{PSset[i * 2]},右侧为{PSset[i * 2 + 1]}")
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
        # 只有当两侧都有任务时才进行NEH排序；否则不同排
        if (PSnum[i * 2] != 0) & (PSnum[i * 2 + 1] != 0):
            truesort(left, right, PSnum[i * 2], PSnum[i * 2 + 1], mindiv.AM, mindiv.WTS, mindiv.CTS, i * 2 + 1)
        print(f"最终排序后第{i}个配对站，左侧为{left},右侧为{right}")
        for j in range(PSnum[i * 2]):
            PSset[i * 2][j] = left[j]
        for j in range(PSnum[i * 2 + 1]):
            PSset[i * 2 + 1][j] = right[j]
        # 根据配对站两边任务，计算该配对站两侧目标值相关内容
        point = -1  # 表示左侧
        k1 = 0
        k2 = 0
        print("开始算目标值")

        while (k1 < PSnum[i * 2]) | (k2 < PSnum[i * 2 + 1]):
            print(f"左侧已完成{k1}个任务，右侧已完成{k2}个任务")
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
                if point == 1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2][k1]
                am = mindiv.AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2]]
                    WTA[SW[i * 2]] = WTA[SW[i * 2]] + ptime
                    print(
                        f"任务{task}选择工人模式，工人为{SW[i * 2]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2]]
                        CTA[SC[i * 2]] = CTA[SC[i * 2]] + ptime
                        print(
                            f"任务{task}选择机器人模式，机器人为{SC[i * 2]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2] * NC + SC[i * 2]]
                        WTT[SW[i * 2]] = WTT[SW[i * 2]] + ptime
                        CTT[SC[i * 2]] = CTT[SC[i * 2]] + ptime
                        print(
                            f"任务{task}选择合作模式，工人为{SW[i * 2]},机器人为{SC[i * 2]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2]]}，该机器人合作工作时间为{CTT[SC[i * 2]]}")
                curtime[i * 2] = curtime[i * 2] + ptime
                ctask[task] = curtime[i * 2]
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
                if point == -1:
                    continue
                # 如果不存在这种任务,则加工当前任务
                task = PSset[i * 2 + 1][k2]
                am = mindiv.AM[task]
                ptime = 0  # 处理时长
                # 工人模式
                if am == 0:
                    ptime = TWtime[task][SW[i * 2 + 1]]
                    WTA[SW[i * 2 + 1]] = WTA[SW[i * 2 + 1]] + ptime
                    print(
                        f"任务{task}选择工人模式，工人为{SW[i * 2 + 1]},处理时间为{ptime},当前该工人单独工作时间为{WTA[SW[i * 2 + 1]]}")
                # 机器人模式
                else:
                    if am == 1:
                        ptime = TCtime[task][SC[i * 2 + 1]]
                        CTA[SC[i * 2 + 1]] = CTA[SC[i * 2 + 1]] + ptime
                        print(
                            f"任务{task}选择机器人模式，机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该机器人单独工作时间为{CTA[SC[i * 2 + 1]]}")
                    # 合作模式
                    else:
                        ptime = TWCtime[task][SW[i * 2 + 1] * NC + SC[i * 2 + 1]]
                        WTT[SW[i * 2 + 1]] = WTT[SW[i * 2 + 1]] + ptime
                        CTT[SC[i * 2 + 1]] = CTT[SC[i * 2 + 1]] + ptime
                        print(
                            f"任务{task}选择合作模式，工人为{SW[i * 2 + 1]},机器人为{SC[i * 2 + 1]},处理时间为{ptime},当前该工人合作工作时间为{WTT[SW[i * 2 + 1]]}，该机器人合作工作时间为{CTT[SC[i * 2 + 1]]}")
                curtime[i * 2 + 1] = curtime[i * 2 + 1] + ptime
                ctask[task] = curtime[i * 2 + 1]
                k2 = k2 + 1
                Fjk[i * 2 + 1] = Fjk[i * 2 + 1] + AT[task] * ptime
                print(f"加工完任务{task}后,当前时间为{curtime[i * 2 + 1]}")

    # 最大完成时间目标
    cmax = curtime[0]
    for i in range(NS * 2):
        if curtime[i] > cmax:
            cmax = curtime[i]
        Fjk[i] = 1 - math.exp(-Fjk[i])

    # 计算机器人能耗目标
    TCE = 0
    for i in range(NC):
        TCE = TCE + CE[i] * (CTA[i] + CTT[i]) + CIE[i] * (cmax - CTA[i] - CTT[i])

    # 统计每个工作站机器人能耗
    stationCE = [0 for _ in range(NS * 2)]
    # print(SW)
    # print(mindiv.CTS)
    for i in range(NS * 2):
        if SC[i] != -1:
            stationCE[i] = CE[SC[i]] * (CTA[SC[i]] + CTT[SC[i]]) + CIE[SC[i]] * (cmax - CTA[SC[i]] - CTT[SC[i]])

    # 关键工作站索引
    Emaxindex = 0
    for i in range(NS * 2):
        if stationCE[i] > stationCE[Emaxindex]:
            Emaxindex = i

    # 如果当前工厂无工人，则从可用工人中分一个给它
    if (SW[Emaxindex] == -1):
        # 看现在是否有其他工人未分配
        indexes = [index for index, value in enumerate(mindiv.WTS) if value == -1]
        if len(indexes) == 1:
            SW[Emaxindex] = indexes[0]
            mindiv.WTS[SW[Emaxindex]] = Emaxindex
        else:
            if len(indexes) != 0:
                # 从中随机选择
                SW[Emaxindex] = random.choice(indexes)
                mindiv.WTS[SW[Emaxindex]] = Emaxindex

    # 对于当前工作站中所有任务，判断其开始时间+合作模式处理时间<所有同配对工作站紧后任务的开始时间，则更改
    changenum = 0  # 标记被修改的任务数
    # 找到当前工人任务索引
    cobotindex = [-1 for _ in range(PSnum[Emaxindex])]
    cobottasknum = 0

    for j in range(PSnum[Emaxindex]):
        curtask = PSset[Emaxindex][j]
        if mindiv.AM[curtask] == 1:
            cobotindex[cobottasknum] = curtask
            cobottasknum = cobottasknum + 1
            # 找到其后续任务集合
            dom_indices = [index for index, value in enumerate(preor[curtask][:]) if value != 0]
            samestationdom = [-1 for _ in range(len(dom_indices))]
            k = 0
            # 找到同配对工作站后续任务集合
            for ttask in dom_indices:
                if (int(mindiv.TA[ttask] / 2) == int(mindiv.TA[curtask] / 2)):
                    samestationdom[k] = ttask
                    k = k + 1
                    # 如果无后续任务，可随意修改
            if k == 0:
                # 若可修改为合作模式，则修改为合作模式
                if (TWCtime[curtask][SW[Emaxindex] * NC + SC[Emaxindex]] != -1):
                    mindiv.AM[curtask] = 2
                    changenum = changenum + 1
                else:
                    # 若能修改为工人模式，则修改为工人模式
                    if (TWtime[curtask][SC[Emaxindex]] != -1):
                        mindiv.AM[curtask] = 0
                        changenum = changenum + 1
            else:
                # 找出所有后续中最早的开始时间
                earstart = taskstart[samestationdom[0]]
                for i in samestationdom:
                    if i != -1:
                        if taskstart[i] < earstart:
                            earstart = taskstart[i]

                # 若可修改为合作模式，则修改为合作模式
                if (TWCtime[curtask][SW[Emaxindex] * NC + SC[Emaxindex]] != -1) & (
                        taskstart[curtask] + TWCtime[curtask][SW[Emaxindex] * NC + SC[Emaxindex]] <= earstart):
                    mindiv.AM[curtask] = 2
                    changenum = changenum + 1
                else:
                    # 若能修改为工人模式，则修改为工人模式
                    if (TCtime[curtask][SC[Emaxindex]] != -1) & (
                            taskstart[curtask] + TWtime[curtask][SC[Emaxindex]] <= earstart):
                        mindiv.AM[curtask] = 0
                        changenum = changenum + 1
    # 没有任务发生更改，随机选择其中一个工人任务插入到可以加工的工作站内
    if (changenum == 0) & (cobottasknum != 0):
        changetask = random.choice(cobotindex[0:cobottasknum])
        TAchange(mindiv, changetask)

    return mindiv

