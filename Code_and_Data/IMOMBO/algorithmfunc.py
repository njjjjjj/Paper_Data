from globalsearch import *
import numpy as np
# 快速非支配排序
# fit表示要排序的序列，length表示长度，返回排序后的index
def fastsort(fit, length):
    # 对于每个fit[i]，计算支配它的项的数目和它支配项的索引
    bedomnum = [0 for _ in range(length)]
    domindex = [[-1 for _ in range(length)] for _ in range(length)]
    domnum = [0 for _ in range(length)]
    for i in range(length):
        for j in range(i + 1, length):
            # 如果i支配j
            if dominates(fit[i], fit[j]):
                # print(f"{i}支配{j}")
                bedomnum[j] = bedomnum[j] + 1
                domindex[i][domnum[i]] = j
                domnum[i] = domnum[i] + 1
            # 如果j支配i
            if dominates(fit[j], fit[i]):
                # print(f"{j}支配{i}")
                bedomnum[i] = bedomnum[i] + 1
                domindex[j][domnum[j]] = i
                domnum[j] = domnum[j] + 1
    print(domnum, bedomnum)
    print(domindex)
    # 分支配等级分层
    index = [[-1 for _ in range(length)] for _ in range(length)]
    k = 0
    # 总层数
    klevel = 0
    # 每一层的数量
    levellength = [0 for _ in range(length)]
    while (k < length):
        # 找到当前bedomnum=0的项目
        zero_indices = [index for index, value in enumerate(bedomnum) if value == 0]
        print(zero_indices)
        # 将这些任务加入当前组
        index[klevel] = zero_indices
        levellength[klevel] = len(zero_indices)

        # 访问这些项目的支配项，对应bedomnum-1
        for i in range(len(zero_indices)):
            bedomnum[zero_indices[i]] = bedomnum[zero_indices[i]] - 1  # 让这个数变成-1，避免再次访问
            for j in range(domnum[zero_indices[i]]):
                bedomnum[domindex[zero_indices[i]][j]] = bedomnum[domindex[zero_indices[i]][j]] - 1
        k = k + levellength[klevel]
        klevel = klevel + 1

    print(klevel, index)
    print(levellength)
    return klevel, levellength, index


#拥挤度计算
def crowd(fit,length,objnum):
    crowdist=[0 for _ in range(length)]
    for i in range(objnum):
        temp = [row[i] for row in fit]
        sorted_indices = sorted(range(len(temp)), key=lambda j: temp[j])
        print(f"目标{i}数据:",temp)
        print(f"目标{i}排序:",sorted_indices)
        maxobj=fit[sorted_indices[length-1]][i]
        minobj=fit[sorted_indices[0]][i]
        for j in range(length):
            if maxobj!=minobj:
                if (j==0)|(j==length-1):
                    crowdist[sorted_indices[j]]=math.inf
                else:
                    crowdist[sorted_indices[j]]=crowdist[sorted_indices[j]]+(fit[sorted_indices[j+1]][i]-fit[sorted_indices[j-1]][i])/(maxobj-minobj)
    return crowdist


# 划分种群
def partpop(klevel, levellength, sortindex,PN):
    # 从第一层随机选择一个个体作为领飞鸟
    Leaderindex = random.choice(sortindex[0][0:levellength[0]])
    print(f"领飞鸟为{Leaderindex}")
    # 顺序查找PN/3*2个任务
    Followindex = [-1 for _ in range(int(PN / 3 * 2))]
    fn = 0
    k = 0
    while fn < int(PN / 3 * 2):
        for i in range(int(levellength[k])):
            print('----------')
            print(k)
            print(i)
            print(sortindex[k][i])
            print(Leaderindex)
            print(fn)
            print(levellength[k])
            print(int(PN / 3 * 2))
            print(Followindex)
            print(sortindex[k])
            if (sortindex[k][i] != Leaderindex) & (sortindex[k][i] != -1):
                Followindex[fn] = sortindex[k][i]
                fn = fn + 1
                if fn >=int(PN / 3 * 2):
                    break
        if fn >= int(PN / 3 * 2):
            break
        else:
            k = k + 1

    # 左侧鸟
    flindex = [Followindex[i] for i in range(len(Followindex)) if i % 2 != 0]
    # 右侧鸟
    frindex = [Followindex[i] for i in range(len(Followindex)) if i % 2 == 0]
    print(flindex, frindex)

    # 剩余的是自由鸟
    flag = [0 for _ in range(PN)]
    for i in Followindex:
        flag[i] = 1
    flag[Leaderindex] = 1

    print(flag)
    freeindex = [index for index, value in enumerate(flag) if value == 0]
    print(freeindex)
    return Leaderindex, flindex, frindex, freeindex, Followindex


def copy(indiv,copyindiv):
    for i in range(Ntask):
        copyindiv.TA[i]=indiv.TA[i]
        copyindiv.AM[i]=indiv.AM[i]
    for i in range(NW):
        copyindiv.WTS[i]=indiv.WTS[i]
    for i in range(NC):
        copyindiv.CTS[i]=indiv.CTS[i]
    for i in range(objnum):
        copyindiv.fitness[i]=indiv.fitness[i]


def calculate_spacing(solutions):
    """
    计算改进的Spacing指标，更好地衡量解集在目标空间中的分布均匀性
    """
    if len(solutions) <= 1:
        return 0

    # 步骤1: 目标归一化 - 避免某一维度主导计算
    normalized_sols = np.copy(solutions)
    for i in range(solutions.shape[1]):  # 对每个目标维度
        dim_min = np.min(solutions[:, i])
        dim_max = np.max(solutions[:, i])
        if dim_max > dim_min:  # 避免除以零
            normalized_sols[:, i] = (solutions[:, i] - dim_min) / (dim_max - dim_min)

    # 步骤2: 计算每个解到其最近邻居的欧几里得距离
    min_distances = []
    for i in range(len(normalized_sols)):
        distances = []
        for j in range(len(normalized_sols)):
            if i != j:
                dist = np.sqrt(np.sum((normalized_sols[i] - normalized_sols[j]) ** 2))
                distances.append(dist)
        min_distances.append(min(distances))

    # 步骤3: 计算标准差作为spacing指标
    mean_dist = np.mean(min_distances)
    spacing = np.sqrt(np.sum((np.array(min_distances) - mean_dist) ** 2) / len(min_distances))

    # 步骤4: 标准化spacing - 使得值越小表示分布越均匀
    # 这里使用平均最小距离进行归一化
    if mean_dist > 0:
        spacing = spacing / mean_dist

    return spacing


# 计算Spacing
# def calculate_spacing(solutions):
#     distances = []
#     for i, sol in enumerate(solutions):
#         # 计算解 i 到其他解的欧几里得距离
#         dist = [np.linalg.norm(sol - solutions[j]) for j in range(len(solutions)) if i != j]
#         distances.append(min(dist))  # 选择最小距离
#     mean_dist = np.mean(distances)
#     spacing = np.sqrt(np.mean((distances - mean_dist) ** 2))
#     return spacing

def sort_indices_by_values(arr):
    # 将数组的元素和它们的索引打包在一起
    indexed_array = list(enumerate(arr))
    # 根据元素的值对打包后的列表进行排序
    sorted_indexed_array = sorted(indexed_array, key=lambda x: x[1])
    # 返回排序后的索引
    return [index for index, value in sorted_indexed_array]


# 判断当前某个解是否能被加入外部档案，MArchive：档案；Archivefit：目标值；curnum:当前档案解个数；newsolution新解,newfit
def archivechange(MArchive, curnum, newsolution, newfit,maxNAP):
    arcfit = [[-1 for _ in range(objnum)] for _ in range(maxNAP + 1)] #修改

    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            for i in range(curnum):
                arcfit[i][0], arcfit[i][1], arcfit[i][2], MArchive[i].FT = decode(MArchive[i].TA, MArchive[i].AM, MArchive[i].WTS,
                                                                  MArchive[i].CTS)
                # arcfit[i][0], arcfit[i][1], MArchive[i].FT = decode1(MArchive[i].TA, MArchive[i].AM, MArchive[i].WTS,
                #                                                   MArchive[i].CTS)
    print("当前档案适应度为", arcfit[0:curnum])
    print("当前档案长度为", curnum)
    print("需要新放入的解目标为", newfit)
    # 判断当前档案是否已满
    # 档案没满
    if curnum < maxNAP:
        # 判断当前解是否被其他解支配
        for i in range(curnum):
            if dominates(arcfit[i], newfit) | (all(x > y - 1e-6 and x < y + 1e-6 for x, y in zip(arcfit[i], newfit))):
                if dominates(arcfit[i], newfit):
                    print("新解被支配")
                else:
                    print("新解和档案中目标值相同")

                # 被其他解支配，则返回-1，代表插入失败
                return MArchive, curnum
        # 如果当前解支配某个解，则去掉被支配的解
        for i in range(curnum):
            if dominates(newfit, arcfit[i]):
                print(f"档案中第{i}个解目标为{arcfit[i]},被新解支配，所以被删掉")
                MArchive[i] = MArchive[curnum - 1]
                i = i - 1
                curnum = curnum - 1
        # 将当前解放入最后一位
        MArchive[curnum] = newsolution
        curnum = curnum + 1
        print("新解已插入")

    else:
        # 判断当前解是否被其他解支配
        for i in range(curnum):
            if dominates(arcfit[i], newfit) | (all(x > y - 1e-6 and x < y + 1e-6 for x, y in zip(arcfit[i], newfit))):
                if dominates(arcfit[i], newfit):
                    print("新解被支配")
                else:
                    print("新解和档案中目标值相同")
                # 被其他解支配，则返回-1，代表插入失败
                return MArchive, curnum
        # 如果当前解支配某个解，则去掉被支配的解
        for i in range(curnum):
            if dominates(newfit, arcfit[i]):
                print(f"档案中第{i}个解目标为{arcfit[i]},被新解支配，所以被删掉")
                MArchive[i] = MArchive[curnum - 1]
                i = i - 1
                curnum = curnum - 1
        if curnum < maxNAP:
            # 删除后若未满，则将当前解放入最后一位
            print("新解将插入位置{curnum}")
            MArchive[curnum] = newsolution
            curnum = curnum + 1

        else:
            # 否则，计算拥挤度，删除拥挤距离最小的个体
            temparcgive = [indiv() for _ in range(maxNAP + 1)]
            for i in range(maxNAP):
                copy(MArchive[i], temparcgive[i])
            temparcgive[maxNAP] = newsolution
            with open(os.devnull, 'w') as f:
                with contextlib.redirect_stdout(f):
                    for i in range(maxNAP + 1):
                        arcfit[i][0], arcfit[i][1], arcfit[i][2], temparcgive[i].FT = decode(temparcgive[i].TA, temparcgive[i].AM,
                                                                          temparcgive[i].WTS, temparcgive[i].CTS)
                        # arcfit[i][0], arcfit[i][1], temparcgive[i].FT = decode1(temparcgive[i].TA, temparcgive[i].AM,
                        #                                                   temparcgive[i].WTS, temparcgive[i].CTS)
            crowdist = crowd(arcfit, maxNAP + 1, objnum)#修改
            # 找到最小的索引
            mincrowindex = 0
            for i in range(maxNAP + 1):
                if crowdist[i] < crowdist[mincrowindex]:
                    mincrowindex = i

            for i in range(maxNAP):
                if i < mincrowindex:
                    copy(temparcgive[i], MArchive[i])
                else:
                    copy(temparcgive[i + 1], MArchive[i])

            curnum = maxNAP
    return MArchive, curnum


def select_top_crowded_solutions(npop, objnum, top_k=10):
    fit = [p.fitness for p in npop]
    crowding_distances = crowd(fit, len(npop), objnum)

    # 将npop中的解和对应的拥挤度距离组成元组
    solutions_with_distances = list(zip(npop, crowding_distances))
    # 根据拥挤度距离对元组进行降序排序
    sorted_solutions = sorted(solutions_with_distances, key=lambda x: x[1], reverse=True)
    # 选择拥挤度距离前10的解
    top_solutions = [solution for solution, _ in sorted_solutions[:top_k]]
    return top_solutions

